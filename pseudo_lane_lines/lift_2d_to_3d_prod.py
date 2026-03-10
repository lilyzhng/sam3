"""LiDAR 3D lifting from SAM3 2D lane-line detections."""

import logging
from typing import Final

import numpy as np
import numpy.typing as npt

from autonomy.perception.labels.pseudo_lanelines.data_model import LanePoints3D
from kits.ml.calibration.camera import CameraCalibrationData
from kits.ml.geometry.se3 import SE3
from kits.ml.lidar.recarray import convert_lidar_recarray_to_numpy

_LOGGER: Final = logging.getLogger(__name__)

# Depth bounds for the front-of-camera filter (meters), matching the nuScenes prototype.
_MIN_DEPTH_M: Final[float] = 1.0
_MAX_DEPTH_M: Final[float] = 80.0

#: LiDAR channels to extract from structured point cloud arrays.
_LIDAR_XYZ_CHANNELS: Final[list[str]] = ["x", "y", "z"]
_LIDAR_XYZI_CHANNELS: Final[list[str]] = ["x", "y", "z", "intensity"]


def _transform_lidar_to_vehicle(
    pts_lidar: npt.NDArray[np.float32],
    vehicle_se3_lidar: SE3,
) -> npt.NDArray[np.float64]:
    """Transform LiDAR points from sensor frame to vehicle frame.

    Args:
        pts_lidar: Point cloud in lidar sensor frame, shape (N, 3+).
        vehicle_se3_lidar: Extrinsic transform vehicle_SE3_lidar.

    Returns:
        XYZ coordinates in vehicle frame, shape (N, 3).
    """
    xyz_lidar = pts_lidar[:, :3].astype(np.float64)
    # SE3.apply expects (3, N) and returns (3, N).
    pts_vehicle = vehicle_se3_lidar.apply(xyz_lidar.T).T  # (N, 3)
    return pts_vehicle


def lift_detections_to_3d(
    pts_vehicle: npt.NDArray[np.float64],
    intensities: npt.NDArray[np.float32],
    cam_calib_data: CameraCalibrationData,
    detections: list,  # list[DetectionWithMask] from SAM3
    crop_xmin: int,
    crop_ymin: int,
    rescale_factor: float,
    lane_classes: frozenset[str],
    min_score: float,
    min_depth: float = _MIN_DEPTH_M,
    max_depth: float = _MAX_DEPTH_M,
) -> list[LanePoints3D]:
    """Lift SAM3 full-image masks to 3D lane points via LiDAR projection.

    This function works directly with SAM3's DetectionWithMask output —
    full-resolution boolean masks at inference image size. No RLE encoding,
    no bbox cropping, no downsampling. Just project LiDAR into the camera
    and look up mask[v, u].

    Coordinate mapping (native pixel → mask pixel):
        mask_u = (pixel_u_native - crop_xmin) * rescale_factor
        mask_v = (pixel_v_native - crop_ymin) * rescale_factor

    Args:
        pts_vehicle: LiDAR points in vehicle frame, shape (N, 3).
        intensities: Per-point intensity, shape (N,).
        cam_calib_data: Camera calibration for projection.
        detections: Raw SAM3 output — each has .mask (H,W bool), .label, .score, .box.
        crop_xmin: Native pixel x-offset of the crop applied before SAM3.
        crop_ymin: Native pixel y-offset of the crop applied before SAM3.
        rescale_factor: input_image_rescale_factor (e.g. 0.25). Maps native→inference coords.
        lane_classes: Set of lane label strings to keep.
        min_score: Minimum confidence threshold.
        min_depth: Minimum depth in camera frame (meters).
        max_depth: Maximum depth in camera frame (meters).

    Returns:
        List of LanePoints3D, one per detected lane type.
    """
    if pts_vehicle.shape[0] == 0:
        return []

    # Filter lane detections by class and score.
    lane_dets = [
        d for d in detections
        if d.label.lower() in lane_classes and d.score >= min_score
    ]
    if not lane_dets:
        return []

    # Project vehicle-frame points into camera.
    pts_camera = cam_calib_data.vehicle_se3_sensor.inverse().apply(pts_vehicle.T)  # (3, N)
    depth = pts_camera[2]
    in_front = (depth > min_depth) & (depth < max_depth)
    front_idx = np.where(in_front)[0]
    if front_idx.size == 0:
        return []

    projected = cam_calib_data.project_world_onto_camera(
        pts_vehicle[front_idx].T, valid_points=False, return_transposed=False,
    )
    pixel_u_native = projected[0]
    pixel_v_native = projected[1]

    # Map native pixel coords → inference image coords.
    mask_u = (pixel_u_native - crop_xmin) * rescale_factor
    mask_v = (pixel_v_native - crop_ymin) * rescale_factor

    _LOGGER.info(
        "  Projection: %d front pts, native u=[%.0f,%.0f] v=[%.0f,%.0f], "
        "crop=(%d,%d) rescale=%.4f, mask_u=[%.1f,%.1f] mask_v=[%.1f,%.1f]",
        front_idx.size,
        pixel_u_native.min(), pixel_u_native.max(),
        pixel_v_native.min(), pixel_v_native.max(),
        crop_xmin, crop_ymin, rescale_factor,
        mask_u.min(), mask_u.max(), mask_v.min(), mask_v.max(),
    )

    # Build per-type results by checking each lane detection mask.
    type_hits: dict[str, list[int]] = {}  # lane_type → list of front_idx indices

    for det in lane_dets:
        label = det.label.lower()
        mask = det.mask  # (H, W) bool array at inference resolution
        mask_h, mask_w = mask.shape

        # Filter to points within this mask's bounds.
        in_bounds = (
            (mask_u >= 0) & (mask_u < mask_w)
            & (mask_v >= 0) & (mask_v < mask_h)
        )
        bl = np.where(in_bounds)[0]
        if bl.size == 0:
            continue

        u_int = mask_u[bl].astype(np.int32)
        v_int = mask_v[bl].astype(np.int32)

        # Direct lookup — the entire point of bypassing RLE.
        hit = mask[v_int, u_int]
        hit_local = bl[hit]

        # Diagnostic: understand mask coverage and LiDAR distribution.
        mask_true_count = int(mask.sum())
        mask_true_rows = np.where(mask.any(axis=1))[0]
        mask_true_cols = np.where(mask.any(axis=0))[0]
        _LOGGER.info(
            "  '%s' (score=%.2f): mask %dx%d, %d True pixels (%.1f%%), "
            "True rows=[%d,%d], True cols=[%d,%d]",
            det.label, det.score, mask_h, mask_w,
            mask_true_count, 100.0 * mask_true_count / (mask_h * mask_w),
            mask_true_rows[0] if len(mask_true_rows) > 0 else -1,
            mask_true_rows[-1] if len(mask_true_rows) > 0 else -1,
            mask_true_cols[0] if len(mask_true_cols) > 0 else -1,
            mask_true_cols[-1] if len(mask_true_cols) > 0 else -1,
        )
        # Count LiDAR points in the mask's True bounding region.
        if len(mask_true_rows) > 0 and len(mask_true_cols) > 0:
            in_overlap = (
                (v_int >= mask_true_rows[0]) & (v_int <= mask_true_rows[-1])
                & (u_int >= mask_true_cols[0]) & (u_int <= mask_true_cols[-1])
            )
            overlap_count = int(in_overlap.sum())
            # Check what mask values those overlap points see.
            if overlap_count > 0:
                ov_v = v_int[in_overlap]
                ov_u = u_int[in_overlap]
                ov_vals = mask[ov_v, ov_u]
                _LOGGER.info(
                    "    Overlap region: %d LiDAR pts, mask vals: %d True / %d False, "
                    "dtype=%s, sample mask[%d,%d]=%s",
                    overlap_count, int(ov_vals.sum()), int((~ov_vals).sum()),
                    mask.dtype, ov_v[0], ov_u[0], mask[ov_v[0], ov_u[0]],
                )
            else:
                _LOGGER.info("    Overlap region: 0 LiDAR pts in True bbox")
        _LOGGER.info(
            "    LiDAR u_int range=[%d,%d], v_int range=[%d,%d], %d in-bounds, %d hits",
            u_int.min(), u_int.max(), v_int.min(), v_int.max(),
            bl.size, hit_local.size,
        )

        if hit_local.size > 0:
            if label not in type_hits:
                type_hits[label] = []
            type_hits[label].extend(hit_local.tolist())

    # Deduplicate and build LanePoints3D per type.
    results: list[LanePoints3D] = []
    for lane_type, local_indices in sorted(type_hits.items()):
        unique_local = np.unique(local_indices)
        orig_indices = front_idx[unique_local]
        results.append(
            LanePoints3D(
                lane_type=lane_type,
                points_ego=pts_vehicle[orig_indices].astype(np.float32),
                intensities=intensities[orig_indices],
                pixel_coords=np.column_stack([
                    pixel_u_native[unique_local],
                    pixel_v_native[unique_local],
                ]).astype(np.float32),
                num_points=int(orig_indices.shape[0]),
            )
        )
        _LOGGER.info(
            "  Lane type '%s': %d 3D points", lane_type, orig_indices.shape[0],
        )

    return results


def _ensure_2d_point_cloud(pts_raw: npt.NDArray) -> npt.NDArray[np.float32]:
    """Convert a point cloud to a 2D float32 array of shape (N, K).

    Handles both:
    - Structured/record arrays (1D with named fields like 'x', 'y', 'z')
    - Plain 2D float arrays (N, K)

    Args:
        pts_raw: Raw point cloud, either structured or 2D.

    Returns:
        2D float32 array of shape (N, K) where K >= 3.
    """
    if pts_raw.ndim == 1 and pts_raw.dtype.names is not None:
        # Structured recarray — use convert_lidar_recarray_to_numpy
        # Try xyzi first, fall back to xyz
        try:
            return convert_lidar_recarray_to_numpy(pts_raw, _LIDAR_XYZI_CHANNELS).T.astype(np.float32)
        except (ValueError, KeyError):
            return convert_lidar_recarray_to_numpy(pts_raw, _LIDAR_XYZ_CHANNELS).T.astype(np.float32)
    # Already a 2D array
    return pts_raw.astype(np.float32)
