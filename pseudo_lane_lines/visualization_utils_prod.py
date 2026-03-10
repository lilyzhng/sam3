import logging
from pathlib import Path
from typing import Final, Optional


import cv2
import numpy as np
import numpy.typing as npt


from autonomy.perception.labels.pseudo_lanelines.data_model import LanePoints3D


_LOGGER: Final = logging.getLogger(__name__)


# Colors for lane types (BGR for OpenCV).
LANE_COLORS_BGR: Final[list[tuple[int, int, int]]] = [
   (255, 255, 0),  # cyan
   (0, 255, 0),  # green
   (0, 165, 255),  # orange
   (0, 0, 255),  # red
   (255, 0, 255),  # magenta
   (255, 128, 0),  # blue-ish
   (0, 255, 255),  # yellow
]
_NON_LANE_COLOR_BGR: Final[tuple[int, int, int]] = (128, 128, 128)


_BEV_SIZE_PX: Final[int] = 800
_BEV_FWD_RANGE_M: Final[float] = 80.0
_BEV_LAT_RANGE_M: Final[float] = 40.0




def draw_camera_overlay(
   image_bgr: npt.NDArray[np.uint8],
   lane_points_list: list[LanePoints3D],
   all_pixel_coords: Optional[npt.NDArray[np.float32]] = None,
   point_radius: int = 2,
) -> npt.NDArray[np.uint8]:
   """Draw lane LiDAR points on a camera image.


   Args:
       image_bgr: Camera image in BGR, shape (H, W, 3).
       lane_points_list: Per-type lifted lane points for this camera.
       all_pixel_coords: Optional (M, 2) array of all projected LiDAR pixel
           coords (drawn in gray as background).
       point_radius: Circle radius in pixels.


   Returns:
       Annotated image copy.
   """
   overlay = image_bgr.copy()


   if all_pixel_coords is not None:
       for u, v in all_pixel_coords.astype(np.int32):
           cv2.circle(overlay, (int(u), int(v)), point_radius, _NON_LANE_COLOR_BGR, -1)


   y_offset = 30
   for idx, lp in enumerate(lane_points_list):
       color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
       for u, v in lp.pixel_coords.astype(np.int32):
           cv2.circle(overlay, (int(u), int(v)), point_radius + 1, color, -1)
       label = f"{lp.lane_type}: {lp.num_points} pts"
       cv2.putText(overlay, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
       y_offset += 25


   total_lane = sum(lp.num_points for lp in lane_points_list)
   total_proj = all_pixel_coords.shape[0] if all_pixel_coords is not None else total_lane
   pct = 100 * total_lane / max(total_proj, 1)
   h = overlay.shape[0]
   cv2.putText(
       overlay,
       f"LiDAR: {total_proj} projected, {total_lane} lane ({pct:.1f}%)",
       (10, h - 15),
       cv2.FONT_HERSHEY_SIMPLEX,
       0.5,
       (255, 255, 255),
       1,
   )
   return overlay




def draw_bev(
   lane_points_list: list[LanePoints3D],
   bev_size_px: int = 800,
   fwd_range_m: float = 50.0,  # Fixed forward range
   lat_range_m: float = 30.0,
   point_radius: int = 2,
) -> npt.NDArray[np.uint8]:
   """Render a forward-facing BEV scatter of ego-frame lane points.


   Ego vehicle is at bottom-center, forward is up, left is left.
   Only forward-hemisphere points (x > 0) are rendered.


   Args:
       lane_points_list: Per-type lifted lane points.
       bev_size_px: Height (and width) of the output image.
       fwd_range_m: Forward range in meters (fixed to 50m).
       lat_range_m: Lateral half-range in meters.
       point_radius: Circle radius.


   Returns:
       BEV image, shape (bev_size_px, bev_size_px, 3), BGR.
   """
   bev = np.zeros((bev_size_px, bev_size_px, 3), dtype=np.uint8)


   # Use a uniform scale for both forward and lateral dimensions
   max_range_m = max(fwd_range_m, lat_range_m)
   scale = bev_size_px / (2 * max_range_m)
   cx = bev_size_px / 2


   def _ego_to_bev(pts: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
       bev_x = (cx - pts[:, 1] * scale).astype(np.int32)
       bev_y = (bev_size_px - pts[:, 0] * scale).astype(np.int32)
       return np.column_stack([bev_x, bev_y])


   for idx, lp in enumerate(lane_points_list):
       if lp.num_points == 0:
           continue
       fwd_mask = lp.points_ego[:, 0] > 0
       if not fwd_mask.any():
           continue
       color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
       bev_pts = _ego_to_bev(lp.points_ego[fwd_mask])
       for bx, by in bev_pts:
           if 0 <= bx < bev_size_px and 0 <= by < bev_size_px:
               cv2.circle(bev, (int(bx), int(by)), point_radius + 1, color, -1)


   ego_x = bev_size_px // 2
   ego_y = bev_size_px - 10
   cv2.drawMarker(bev, (ego_x, ego_y), (255, 255, 255), cv2.MARKER_DIAMOND, 15, 2)
   cv2.putText(bev, "EGO", (ego_x + 10, ego_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


   for r_m in [10, 20, 30, 40, 50]:
       r_px = int(r_m * scale)
       if r_px < bev_size_px:
           cv2.circle(bev, (ego_x, ego_y), r_px, (40, 40, 40), 1)
           label_y = ego_y - r_px
           if 0 <= label_y < bev_size_px:
               cv2.putText(bev, f"{r_m}m", (ego_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)


   cv2.putText(bev, f"BEV (front, {fwd_range_m:.0f}m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
   cv2.putText(bev, "fwd", (ego_x - 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
   cv2.putText(bev, "left", (10, bev_size_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
   cv2.putText(bev, "right", (bev_size_px - 45, bev_size_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)


   y_off = 45
   for idx, lp in enumerate(lane_points_list):
       color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
       cv2.putText(bev, f"{lp.lane_type}: {lp.num_points}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
       y_off += 18


   return bev




def compose_visualization(
   lane_points_list: list[LanePoints3D],
   camera_image_bgr: Optional[npt.NDArray[np.uint8]] = None,
   all_pixel_coords: Optional[npt.NDArray[np.float32]] = None,
) -> npt.NDArray[np.uint8]:
   """Compose a combined camera-overlay + BEV image (no disk I/O).


   Args:
       lane_points_list: Per-type lifted lane points.
       camera_image_bgr: Optional camera image for overlay panel.
       all_pixel_coords: Optional (M, 2) all-LiDAR pixel coords for background.


   Returns:
       Combined visualization image (BGR).
   """
   bev = draw_bev(lane_points_list)


   if camera_image_bgr is not None:
       overlay = draw_camera_overlay(camera_image_bgr, lane_points_list, all_pixel_coords)
       cam_h = overlay.shape[0]
       bev_resized = cv2.resize(bev, (cam_h, cam_h))
       return np.hstack([overlay, bev_resized])


   return bev




def save_visualization(lane_points_list: list[LanePoints3D], output_path: Path) -> None:
   """Save a BEV-only visualization to disk.


   Args:
       lane_points_list: List of LanePoints3D, one per lane type.
       output_path: Path to save the output image.
   """
   bev = draw_bev(lane_points_list)
   output_path.parent.mkdir(parents=True, exist_ok=True)
   cv2.imwrite(str(output_path), bev)
   _LOGGER.info("Saved BEV visualization to %s", output_path)




def save_three_row_visualization(
   lane_points_list: list[LanePoints3D],
   sam_debug_image_path: Path,
   lidar_proj_image_path: Path,
   output_path: Path,
) -> None:
   """Save a 3-row stacked visualization:

   Row 1: SAM3 inference image (2D masks)
   Row 2: LiDAR projection overlay on camera image
   Row 3: BEV 3D lifting result (after mask filtering)

   Args:
       lane_points_list: Per-type lifted lane points for this frame.
       sam_debug_image_path: Path to the SAM3 debug image.
       lidar_proj_image_path: Path to the LiDAR projection overlay image.
       output_path: Path to save the combined output image.
   """
   sam_img = cv2.imread(str(sam_debug_image_path))
   lidar_img = cv2.imread(str(lidar_proj_image_path))

   if sam_img is None and lidar_img is None:
       _LOGGER.warning("Could not read SAM or LiDAR images, falling back to BEV only.")
       save_visualization(lane_points_list, output_path)
       return

   # Use LiDAR overlay dimensions as the reference — it's at native camera resolution.
   ref_img = lidar_img if lidar_img is not None else sam_img
   target_w = ref_img.shape[1]
   target_h = ref_img.shape[0]

   title_h = 30
   panels = []

   # Row 1: SAM3 inference — resize to match native camera resolution.
   if sam_img is not None:
       sam_resized = cv2.resize(sam_img, (target_w, target_h))
       panel = np.zeros((target_h + title_h, target_w, 3), dtype=np.uint8)
       panel[title_h:, :] = sam_resized
       cv2.putText(panel, "SAM3 2D Masks", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
       panels.append(panel)

   # Row 2: LiDAR projection overlay — already at native resolution.
   if lidar_img is not None:
       panel = np.zeros((target_h + title_h, target_w, 3), dtype=np.uint8)
       panel[title_h:, :] = lidar_img
       cv2.putText(panel, "LiDAR Projection Overlay", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
       panels.append(panel)

   # Row 3: BEV 3D lifting — shorter height, crop empty far-range space.
   bev = draw_bev(lane_points_list, bev_size_px=target_w)
   bev_crop_top = int(target_w * 0.4)
   bev_cropped = bev[bev_crop_top:, :]
   total_pts = sum(lp.num_points for lp in lane_points_list)
   panel = np.zeros((bev_cropped.shape[0] + title_h, target_w, 3), dtype=np.uint8)
   panel[title_h:, :] = bev_cropped
   cv2.putText(panel, f"BEV 3D Lifting ({total_pts} pts)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   panels.append(panel)

   vis = np.vstack(panels)

   output_path.parent.mkdir(parents=True, exist_ok=True)
   cv2.imwrite(str(output_path), vis)
   _LOGGER.info("Saved 3-row visualization to %s", output_path)


