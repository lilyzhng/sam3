"""CLI: run SAM3 lane-line inference → LiDAR 3D lifting → save BEV visualizations.

Runs SAM3 directly on camera images and lifts to 3D using full-resolution masks.
No RLE encoding, no bbox cropping, no downsampling — bypasses the perception
team's compact pipeline entirely.

Usage:
   bazel run //autonomy/perception/labels/pseudo_lanelines:visualizer -- \
     -o /tmp/laneline_lift_viz -r 1 --max-frames 3
"""


import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import click
import cv2
import numpy as np
from PIL import Image


from autonomy.perception.datasets.auto_high_beam.constants import P758_LOG_FILTERS
from autonomy.perception.datasets.semantic_segmentation.sam_autolabeled.config import SemanticSegmentationDagsterConfig
from autonomy.perception.datasets.semantic_segmentation.sam_autolabeled.transforms import create_sam_autolabeler
from autonomy.perception.datasets.unified.gold import data_model as gold
from autonomy.perception.datasets.unified_model_interface.data_model import get_point_cloud_data
from autonomy.perception.labels.pseudo_lanelines.lanelines_config import (
    LaneLabelingSAM3Config,
    build_laneline_autolabeling_config,
)
from autonomy.perception.labels.pseudo_lanelines.lift_2d_to_3d import (
    _ensure_2d_point_cloud,
    _transform_lidar_to_vehicle,
    lift_detections_to_3d,
)
from autonomy.perception.labels.pseudo_lanelines.visualize_utils import (
    save_side_by_side_visualization,
    save_visualization,
)
from kits.ml.calibration.camera import CameraCalibrationData
from kits.ml.geometry.se3 import SE3
from kits.scalex.dataset.index.index_reader import DatasetIndexReader
from kits.scalex.dataset.stage_str import get_manifest_from_stage_str
from kits.scalex.hpc.tiered_file_system import tiered_filesystem
from kits.scalex.logging import base_logger
from kits.scalex.pipeline.hydration_v2 import HydrationTransformationV2
from kits.scalex.pipeline.parquet_source import ParquetSource
from kits.scalex.pipeline.pre_signed_urls import check_credentials
from kits.scalex.pipeline.serial import SerialStreamingExecutor
from platforms.lakefs.client import LakeFS


_LOGGER: Final = logging.getLogger(__name__)


def _get_lidar_points(frame, calibrations, platform):
    """Extract LiDAR points in vehicle frame + intensities from a frame.

    Returns:
        (pts_vehicle, intensities) or (None, None) if LiDAR unavailable.
    """
    frame_lidars = getattr(frame, "lidars", None)
    if not frame_lidars:
        return None, None

    lidar_calib = calibrations.get_center_lidar(platform)
    center_lidar = None
    for lidar_obs in frame_lidars:
        if lidar_obs.sensor_name == lidar_calib.sensor_name or lidar_calib.sensor_name.startswith(
            lidar_obs.sensor_name
        ):
            center_lidar = lidar_obs
            break

    if center_lidar is None and len(frame_lidars) == 1:
        center_lidar = frame_lidars[0]

    if center_lidar is None:
        return None, None

    # Get point cloud data.
    pts_raw = None
    try:
        pts_raw = get_point_cloud_data(center_lidar)
    except (ValueError, AttributeError):
        pass
    if pts_raw is None and hasattr(center_lidar, "point_cloud") and center_lidar.point_cloud is not None:
        pts_raw = getattr(center_lidar.point_cloud, "data", None)
    if pts_raw is None:
        return None, None

    pts_raw = _ensure_2d_point_cloud(pts_raw)
    vehicle_se3_lidar = SE3.from_pose(lidar_calib.vehicle_se3_sensor)
    pts_vehicle = _transform_lidar_to_vehicle(pts_raw, vehicle_se3_lidar)
    intensities = pts_raw[:, 3].astype(np.float32) if pts_raw.shape[1] > 3 else np.zeros(pts_raw.shape[0], dtype=np.float32)
    return pts_vehicle, intensities


def _save_lidar_projection_overlay(
    raw_img: np.ndarray,
    pts_vehicle: np.ndarray,
    cam_calib_data: CameraCalibrationData,
    save_path: Path,
) -> None:
    """Project LiDAR onto raw camera image and save. Color by depth (red=near, blue=far)."""
    img = raw_img.copy()
    # Transform to camera frame once, then reuse for both depth filter and projection.
    pts_camera = cam_calib_data.vehicle_se3_sensor.inverse().apply(pts_vehicle.T)  # (3, N)
    depth = pts_camera[2]
    in_front = (depth > 1.0) & (depth < 80.0)
    front_idx = np.where(in_front)[0]
    if front_idx.size == 0:
        _LOGGER.warning("  No front-facing LiDAR pts for projection overlay.")
        return

    projected = cam_calib_data.project_world_onto_camera(
        pts_camera[:, front_idx], valid_points=True, return_transposed=False, in_camera_frame=True,
    )
    px_u = projected[0].astype(np.int32)
    px_v = projected[1].astype(np.int32)
    img_h, img_w = img.shape[:2]
    valid = (px_u >= 0) & (px_u < img_w) & (px_v >= 0) & (px_v < img_h)
    vi = np.where(valid)[0]
    if vi.size == 0:
        _LOGGER.info("  No LiDAR pts in image bounds, skipping overlay.")
        return
    for idx in vi:
        d = depth[front_idx[idx]]
        t = min(d / 80.0, 1.0)
        color = (int(255 * (1 - t)), 0, int(255 * t))  # BGR: red→blue
        cv2.circle(img, (px_u[idx], px_v[idx]), 2, color, -1)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    _LOGGER.info("  Saved LiDAR projection overlay (%d pts): %s", vi.size, save_path.name)


@click.command()
@click.option("-o", "--output-dir", type=Path, default="/tmp/laneline_lift_viz", help="Output directory.")
@click.option("-r", "--rows", type=int, default=1, help="Number of rows to process.")
@click.option("--max-frames", type=int, default=-1, help="Max frames with LiDAR to process per row (-1 = all).")
@click.option("--confidence", type=float, default=0.6, help="SAM3 confidence threshold.")
@click.option(
    "--debug-projection",
    is_flag=True,
    default=False,
    help="Skip SAM3; just overlay LiDAR projection on the raw camera image.",
)
@click.option(
    "--legend-mode",
    is_flag=True,
    default=False,
    help="Render detection labels as a legend in the top-right corner of debug images.",
)
def main(output_dir: Path, rows: int, max_frames: int, confidence: float, debug_projection: bool, legend_mode: bool) -> None:
    """Run SAM3 lane-line inference → LiDAR 3D lifting → save BEV visualizations."""
    check_credentials()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    lakefs = LakeFS()

    laneline_config = LaneLabelingSAM3Config(confidence_threshold=confidence)
    autolabeler_config = build_laneline_autolabeling_config(
        laneline_config=laneline_config,
        debug_image_save_dir=output_dir,
        max_frames_per_row_override=max_frames,
    )

    dagster_config = SemanticSegmentationDagsterConfig()
    input_manifest = get_manifest_from_stage_str(dagster_config.human_labels_gold_reference, lakefs)
    physical_filesystem = tiered_filesystem()

    index_reader = DatasetIndexReader.from_manifest(input_manifest, lakefs)
    references = index_reader.filter_references_to_process(
        list(input_manifest.key_to_data_file.values()),
        filters=P758_LOG_FILTERS,
    )

    # Create SAM3 autolabeler — skip if only debugging projection.
    sam_autolabeler = None
    if not debug_projection:
        sam_autolabeler = create_sam_autolabeler(autolabeler_config, lakefs)

    # Single hydrator for both images and LiDAR — they must be hydrated
    # together to stay in sync. Separate hydrators cause mismatches.
    hydrator = HydrationTransformationV2(process_images=True, process_radar=False, process_lidar=True)

    num_rows_processed = 0
    for file_reference in references:
        if num_rows_processed >= rows:
            break

        _LOGGER.info("Processing file %s...", file_reference.filename)
        source = ParquetSource[gold.Unified](
            data_model=gold.Unified,
            file_path=file_reference.physical_address,
            filesystem=physical_filesystem,
        )
        serial_executor = SerialStreamingExecutor[gold.Unified, gold.Unified](
            source=source,
            transforms_pipeline=None,
            ignore_exception=False,
        )

        for input_row in serial_executor:
            if num_rows_processed >= rows:
                break

            calibrations = input_row.calibrations
            platform = getattr(input_row, "platform_info", None) or getattr(
                input_row.identifiers, "platform", None,
            )

            # Process ALL frames sequentially — no sampling, no skipping.
            # The hydrator expects sequential calls; skipping frames causes
            # LiDAR/camera mismatch.
            frames_processed = 0
            for frame_idx, frame in enumerate(input_row.frames):
                if max_frames >= 0 and frames_processed >= max_frames:
                    break

                frame_id = getattr(frame, "frame_id", None)

                # Hydrate LiDAR through the same hydrator as images.
                if hasattr(frame, "lidars") and frame.lidars:
                    for lidar_obs in frame.lidars:
                        try:
                            hydrator(lidar_obs)
                        except Exception as e:
                            _LOGGER.warning("Failed to hydrate LiDAR (frame %d): %s", frame_idx, e)

                pts_vehicle, intensities = _get_lidar_points(frame, calibrations, platform)
                if pts_vehicle is None:
                    _LOGGER.info("  Frame %d: no LiDAR, skipping.", frame_idx)
                    continue

                _LOGGER.info("  Frame %d (id=%s): %d LiDAR pts.", frame_idx, frame_id, pts_vehicle.shape[0])

                # Process cameras.
                all_frame_results = []
                if frame.cameras is None:
                    continue

                for camera in frame.cameras:
                    if camera.sensor_name not in autolabeler_config.camera_names_to_label:
                        continue

                    cam_proc = autolabeler_config.camera_processing_configs[camera.sensor_name]

                    # Hydrate image.
                    try:
                        hydrator(camera)
                    except Exception as e:
                        _LOGGER.warning("Failed to hydrate camera '%s': %s", camera.sensor_name, e)
                        continue

                    if camera.image.data is None:
                        continue

                    # Get camera calibration.
                    cam_calib_data = None
                    for cam_calib in calibrations.cameras:
                        if cam_calib.sensor_name == camera.sensor_name:
                            cam_calib_data = CameraCalibrationData.from_data_model(cam_calib)
                            break
                    if cam_calib_data is None:
                        _LOGGER.warning("No calibration for camera '%s'", camera.sensor_name)
                        camera.image.data = None
                        continue

                    # Always save LiDAR projection overlay for debugging.
                    proj_path = output_dir / f"{input_row.row_id}_{frame_id}_{camera.sensor_name}_lidar_proj.jpg"
                    _save_lidar_projection_overlay(camera.image.data, pts_vehicle, cam_calib_data, proj_path)

                    if debug_projection:
                        camera.image.data = None
                        continue

                    # Crop + resize — same preprocessing as transform_pad.py.
                    image = Image.fromarray(
                        camera.image.data[
                            cam_proc.crop_ymin:cam_proc.crop_ymax,
                            cam_proc.crop_xmin:cam_proc.crop_xmax,
                        ]
                    ).resize((cam_proc.input_image_width, cam_proc.input_image_height))

                    # Run SAM3.
                    try:
                        detections = sam_autolabeler([image])[0]
                    except Exception as e:
                        _LOGGER.warning("SAM3 failed on camera '%s': %s", camera.sensor_name, e)
                        detections = []

                    _LOGGER.info("  Camera '%s': %d SAM3 detections", camera.sensor_name, len(detections))

                    if not detections:
                        camera.image.data = None
                        continue

                    # Save SAM3 debug image.
                    if autolabeler_config.debug_image_save_dir:
                        from kits.ml.sam.visualization_utils import plot_image_with_detections
                        debug_name = f"{input_row.row_id}_{frame_id}_{camera.sensor_name}.png"
                        plot_image_with_detections(
                            np.array(image), detections,
                            save_path=str(output_dir / debug_name),
                            plot_bounding_boxes=False,
                            legend_mode=legend_mode,
                        )

                    # Lift to 3D.
                    lift_debug_name = f"{input_row.row_id}_{frame_id}_{camera.sensor_name}"
                    results = lift_detections_to_3d(
                        pts_vehicle=pts_vehicle,
                        intensities=intensities,
                        cam_calib_data=cam_calib_data,
                        detections=detections,
                        crop_xmin=cam_proc.crop_xmin,
                        crop_ymin=cam_proc.crop_ymin,
                        rescale_factor=cam_proc.input_image_rescale_factor,
                        lane_classes=laneline_config.lane_classes,
                        min_score=laneline_config.confidence_threshold,
                        debug_name=lift_debug_name,
                    )
                    all_frame_results.extend(results)
                    camera.image.data = None

                if debug_projection:
                    frames_processed += 1
                    continue

                total_pts = sum(lp.num_points for lp in all_frame_results)
                _LOGGER.info("  Frame %d: %d lane types, %d lane points.", frame_idx, len(all_frame_results), total_pts)

                # Side-by-side visualization.
                sam_debug_path = None
                if frame_id:
                    for cam_name in laneline_config.default_camera_names:
                        candidate = output_dir / f"{input_row.row_id}_{frame_id}_{cam_name}.png"
                        if candidate.exists():
                            sam_debug_path = candidate
                            break

                if all_frame_results:
                    if sam_debug_path is not None:
                        viz_path = output_dir / f"row{num_rows_processed}_frame{frame_idx}_side_by_side.jpg"
                        save_side_by_side_visualization(all_frame_results, sam_debug_path, viz_path)
                    else:
                        viz_path = output_dir / f"row{num_rows_processed}_frame{frame_idx}_bev.jpg"
                        save_visualization(all_frame_results, viz_path)
                else:
                    _LOGGER.warning("  No lane points found, skipping visualization.")

                frames_processed += 1

            num_rows_processed += 1

    _LOGGER.info("Done. %d rows processed. Output: %s", num_rows_processed, output_dir)


if __name__ == "__main__":
    base_logger.configure()
    main()
