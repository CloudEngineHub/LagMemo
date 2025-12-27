# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os
import shutil
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import skimage.morphology
from habitat.utils.render_wrapper import append_text_to_image
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision, images_to_video
from natsort import natsorted
from PIL import Image

import lagmemo.utils.pose as pu
from lagmemo.perception.constants import PaletteIndices as PI
from lagmemo.perception.constants import languagenav_2categories_map_color_palette, languagenav_2categories_color_palette
from lagmemo.perception.detection.maskrcnn.coco_categories import (
    coco_categories_color_palette,
)
from lagmemo.utils.visualization import draw_line, get_contour_points

MAP_COLOR_PALETTE = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *coco_categories_color_palette,
    ]
]

MAP_COLOR_PALETTE = languagenav_2categories_map_color_palette

def draw_star(image:np.ndarray, x, y, color):
    center = np.array([x,y])
    
    outer_radius = 100   # 外顶点到中心的距离
    inner_radius = 40    # 内顶点到中心的距离

    # 计算五角星的顶点
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x = center[0] + np.cos(angles) * np.repeat([outer_radius, inner_radius], 5)
    y = center[1] + np.sin(angles) * np.repeat([outer_radius, inner_radius], 5)

    # 将顶点转换为整数
    points = np.array([x, y], dtype=np.int32).T

    # 绘制五角星
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    
    return image


def append_text_to_image_right_align(
    image: np.ndarray, text: List[str], font_size: float = 0.5
) -> np.ndarray:
    """Write lines of text over the top of an image. Text is aligned top-right."""
    h, w, c = image.shape
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            y = textsize[1] + 10

        x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def record_video(
    target_dir: str,
    image_dir: str,
    episode_name: str = "0",
) -> None:
    """Converts a directory of image snapshots into a video."""
    print(f"Recording video {episode_name}")

    # Semantic map vis
    fnames = natsorted(glob.glob(f"{image_dir}/snapshot*.png"))
    imgs = [cv2.imread(fname) for fname in fnames]
    images_to_video(
        [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs],
        target_dir,
        f"{episode_name}",
        fps=10,
        quality=5,
        verbose=True,
    )


class NavVisualizer:
    """
    This class is intended to visualize a single image goal navigation task.
    """

    def __init__(
        self,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ) -> None:
        """
        Arguments:
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            print_images: if True, save visualization as images
            coco_categories_legend: path to the legend image of coco categories
        """
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        if self.print_images:
            os.makedirs(self.default_vis_dir, exist_ok=True)

        self.num_sem_categories = num_sem_categories
        self.map_resolution = map_resolution
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None
        self.ind_frame_height = 450
        # 存储智能体轨迹位置用于渐变绘制
        self.trajectory_positions = []

    def reset(self) -> None:
        self.vis_dir = self.default_vis_dir
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None
        self.trajectory_positions = []  # 重置轨迹位置列表

    def set_vis_dir(self, episode_id: str) -> None:
        self.vis_dir = os.path.join(self.default_vis_dir, str(episode_id))
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        rgb_frame: np.ndarray,
        semantic_frame: np.ndarray,
        timestep: int,
        last_goal_image,
        last_td_map: Dict[str, Any] = None,
        last_collisions: Dict[str, Any] = None,
        semantic_map: Optional[np.ndarray] = None,
        visualize_goal: bool = True,
        metrics: Dict[str, Any] = None,
        been_close_map=None,
        blacklisted_targets_map=None,
        frontier_map: Optional[np.ndarray] = None,
        dilated_obstacle_map: Optional[np.ndarray] = None,
        instance_map: Optional[np.ndarray] = None,
        short_term_goal: Optional[np.ndarray] = None,
        goal_pose = None,
        frontiers = None,
        goal_text = None,
        goal_td_pose: Optional[np.ndarray] = None,
        goal_vis_type = 0,
        explore_info = None,
    ) -> None:
        """Visualize frame input and semantic map.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            found_goal: whether we found the object goal category
            explored_map: (M, M) binary local explored map prediction
            semantic_map: (M, M) local semantic map predictions
            rgb_frame: rgb frame visualization
            semantic_frame: semantic frame visualization
            timestep: time step within the episode
            last_td_map: habitat oracle top down map
            last_collisions: collisions dictionary
            visualize_goal: if True, visualize goal
            metrics: can populate for last frame
        """
        if not self.print_images:
            return

        if last_collisions is None:
            last_collisions = {"is_collision": False}

        if dilated_obstacle_map is not None:
            obstacle_map = dilated_obstacle_map

        goal_frame = self.make_goal(last_goal_image) if last_goal_image is not None else None

        obs_frame = self.make_observations(
            rgb_frame,
            last_collisions["is_collision"],
            found_goal,
            metrics,
        )

        try:
            sem_frame = self.make_sem_observations(
                semantic_frame,
                last_collisions["is_collision"],
                found_goal,
                metrics,
            )
        except:
            sem_frame = self.make_observations(
                semantic_frame,
                last_collisions["is_collision"],
                found_goal,
                metrics,
                text='Semantic'
            )
        map_pred_frame = self.make_map_preds(
            sensor_pose,
            obstacle_map,
            explored_map,
            semantic_map,
            closest_goal_map,
            goal_map,
            visualize_goal,
        )
        
        if last_td_map is not None:
            # 在td map的global pos位置画上一个圆
            if goal_td_pose is not None:
                global_x, global_y = goal_td_pose[:2]
                if 0 <= global_y < last_td_map['map'].shape[0] and 0 <= global_x < last_td_map['map'].shape[1]:
                    # last_td_map['map'] = draw_star(last_td_map['map'], int(global_y), int(global_x), color= (120, 120,120))#(70*(goal_vis_type+1), 0, 0))
                    cv2.circle(last_td_map['map'], (int(global_y), int(global_x)), radius=15, color=(70*(goal_vis_type+1), 0, 0), thickness=-1)
        
        if goal_vis_type == 1:
            explore_type = 'explore'
        elif goal_vis_type == 2:
            explore_type = 'found_goal'
        else:
            explore_type = None
        td_map_frame = None if last_td_map is None else self.make_td_map(last_td_map, explore_type = explore_type, explore_info=explore_info)

        kp_frame = np.ones_like(goal_frame) * 255
        # kp_frame = self.make_keypoint(timestep)
        
        # # wxl, visualize the frontier map
        agent_radius = 0.18 # 0.36
        pixels_per_meter = 20
        area_thresh = 1.0
        kernel_size = pixels_per_meter * agent_radius * 2
        _area_thresh_in_pixels = area_thresh * (pixels_per_meter**2)
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0) - 2
        _navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        obstacle_mask = np.rint(np.array(obstacle_map)) == 1
        navigable_map = 1 - cv2.dilate(
            obstacle_mask.astype(np.uint8),
            _navigable_kernel,
            iterations=1,
        ).astype(np.uint8)
        explored_area = cv2.dilate(
            explored_map.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        obstacle_map_mirror = cv2.dilate(
            1 - obstacle_map.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ).astype(np.uint8)
        
        frontier_frame = cv2.cvtColor(navigable_map * 255, cv2.COLOR_GRAY2BGR)
        frontier_frame[(explored_area > 0)&(obstacle_map_mirror>0)] = (127, 127, 127)
        # for point in frontiers:
        #     point_int = (int(point[0]), int(point[1]))
        #     cv2.circle(frontier_frame, point_int, radius=5, color=(255,0,0), thickness=2)

        # wxl, draw agent arrow for frontier obstacle map
        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)
        
        # 在 frontier_frame 上绘制渐变轨迹
        if len(self.trajectory_positions) >= 2:
            vis_trajectory_frontier = []
            for traj_x, traj_y in self.trajectory_positions:
                # frontier_frame 的坐标系（不翻转 y 轴）
                vis_x = (traj_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0]
                vis_y = (traj_y * 100.0 / self.map_resolution - gy1) * 480 / obstacle_map.shape[1]
                vis_trajectory_frontier.append((vis_x, vis_y))
            
            # 绘制渐变轨迹
            frontier_frame = self.draw_gradient_trajectory(
                frontier_frame,
                vis_trajectory_frontier,
                start_color=(255, 200, 150),  # 浅蓝色 (BGR)
                end_color=(0, 0, 200),        # 深红色 (BGR)
                line_width=2,
            )
        
        # ============ 绘制算法关键节点 ============
        # 坐标转换规则说明（与智能体箭头的 pos 计算保持一致）：
        # 在 planner 中：start = [start_y - gx1, start_x - gy1]（注意 x/y 交换）
        # 在 visualize 中：sensor_pose 解包顺序不同，导致 gx1/gy1 含义交换
        # pos[0] = (curr_x - gx1) * 480 / shape[0]  <- 对应 stg_col 方向
        # pos[1] = (curr_y - gy1) * 480 / shape[1]  <- 对应 stg_row 方向
        # 所以：vis_x 用 shape[0] 缩放，vis_y 用 shape[1] 缩放
        map_h, map_w = obstacle_map.shape[:2]
        scale_x = 480.0 / map_h  # 与 pos[0] 一致
        scale_y = 480.0 / map_w  # 与 pos[1] 一致
        frame_h, frame_w = frontier_frame.shape[:2]
        
        # 1. 绘制 Frontiers（边界探索点）- 蓝色圆圈
        if frontiers is not None and len(frontiers) > 0:
            for frontier_pt in frontiers:
                # frontiers 坐标格式待确认
                fx, fy = frontier_pt[0], frontier_pt[1]
                vis_fx = int(fx)
                vis_fy = int(fy)
                if 0 <= vis_fx < frame_w and 0 <= vis_fy < frame_h:
                    cv2.circle(frontier_frame, (vis_fx, vis_fy), radius=6, color=(255, 100, 0), thickness=2)  # 蓝色圆圈
        
        # 2. 绘制 Goal Map 区域（目标区域）- 绿色半透明区域和绿色星号
        # 注意：只有当 found_goal=True 时 goal_map 才是真正的目标区域
        # 当 found_goal=False 时，goal_map 实际上等于 frontier_map
        if found_goal and goal_map is not None and np.any(goal_map > 0):
            goal_points = np.where(goal_map > 0)
            if len(goal_points[0]) > 0:
                # 创建绿色半透明叠加层来显示目标区域
                overlay = frontier_frame.copy()
                for i in range(len(goal_points[0])):
                    row, col = goal_points[0][i], goal_points[1][i]
                    vis_x = int(col * scale_x)
                    vis_y = int(row * scale_y)
                    if 0 <= vis_x < frame_w and 0 <= vis_y < frame_h:
                        cv2.circle(overlay, (vis_x, vis_y), radius=3, color=(0, 255, 0), thickness=-1)
                # 叠加半透明效果
                cv2.addWeighted(overlay, 0.4, frontier_frame, 0.6, 0, frontier_frame)
                
                # 在目标区域中心绘制绿色星号
                goal_center_row = int(np.mean(goal_points[0]))
                goal_center_col = int(np.mean(goal_points[1]))
                vis_goal_center_x = int(goal_center_col * scale_x)
                vis_goal_center_y = int(goal_center_row * scale_y)
                if 0 <= vis_goal_center_x < frame_w and 0 <= vis_goal_center_y < frame_h:
                    self._draw_star(frontier_frame, vis_goal_center_x, vis_goal_center_y, 
                                   radius=12, color=(0, 255, 0), thickness=2)
        
        # 3. 绘制 Closest Goal（最近目标点）- 红色实心圆
        # closest_goal_map 是与 obstacle_map 相同大小的局部地图
        if closest_goal_map is not None and np.any(closest_goal_map > 0):
            goal_points = np.where(closest_goal_map > 0)
            if len(goal_points[0]) > 0:
                # goal_points[0] 是 rows, goal_points[1] 是 cols
                goal_center_row = int(np.mean(goal_points[0]))
                goal_center_col = int(np.mean(goal_points[1]))
                # 使用与 pos 相同的缩放方式
                vis_goal_x = int(goal_center_col * scale_x)
                vis_goal_y = int(goal_center_row * scale_y)
                if 0 <= vis_goal_x < frame_w and 0 <= vis_goal_y < frame_h:
                    cv2.circle(frontier_frame, (vis_goal_x, vis_goal_y), radius=10, color=(0, 0, 255), thickness=-1)  # 红色实心圆
        # ============ 关键节点绘制结束 ============
        
        pos = (
                    (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
                    (curr_y * 100.0 / self.map_resolution - gy1) * 480 / obstacle_map.shape[1],
                    np.deg2rad(curr_o),
                )
        agent_arrow = get_contour_points(pos, origin=(0, 0))
        color = MAP_COLOR_PALETTE[9:12][::-1]

        cv2.drawContours(frontier_frame, [agent_arrow], 0, color, -1)
        
        # 为 frontier_frame 添加边框和标签
        frontier_frame = self._add_frontier_frame_label(frontier_frame)

        # 添加object/description map, wxl, 2025.2.21
        if td_map_frame is None and goal_frame is not None:
            obs_frame = self.pad_frame_height(obs_frame, sem_frame.shape[0])
            goal_frame = self.pad_frame_height(goal_frame, sem_frame.shape[0])
            map_pred_frame = self.pad_frame_height(map_pred_frame, frontier_frame.shape[0])
            upper_frame = np.concatenate([goal_frame, obs_frame, sem_frame], axis=1)
            lower_frame = np.concatenate([map_pred_frame, frontier_frame], axis=1)
            if lower_frame.shape[1] > upper_frame.shape[1]:
                upper_frame = self.pad_frame(
                    upper_frame,
                    lower_frame.shape[1]
                )
            else:
                lower_frame = self.pad_frame(
                    lower_frame,
                    upper_frame.shape[1]
                )
            frame = np.concatenate([upper_frame, lower_frame], axis=0)
            # frame = np.concatenate(
            #     [goal_frame, obs_frame, map_pred_frame, frontier_frame], axis=1
            # )
        elif goal_frame is None:
            # import ipdb; ipdb.set_trace()
            if td_map_frame is None:
                obs_frame = self.pad_frame_height(obs_frame, sem_frame.shape[0])
                upper_frame = np.concatenate([obs_frame, sem_frame], axis=1)
                map_pred_frame = self.pad_frame_height(map_pred_frame, frontier_frame.shape[0])
                lower_frame = np.concatenate([map_pred_frame, frontier_frame], axis=1)
                if lower_frame.shape[1] > upper_frame.shape[1]:
                    upper_frame = self.pad_frame(
                        upper_frame,
                        lower_frame.shape[1]
                    )
                else:
                    lower_frame = self.pad_frame(
                        lower_frame,
                        upper_frame.shape[1]
                    )
                frame = np.concatenate([upper_frame, lower_frame], axis=0)
                # frame = np.concatenate(
                #     [obs_frame, map_pred_frame, frontier_frame], axis=1
                # )
            else:
                obs_frame = self.pad_frame_height(obs_frame, sem_frame.shape[0])
                upper_frame = np.concatenate([obs_frame, sem_frame], axis=1)
                map_pred_frame = self.pad_frame_height(map_pred_frame, frontier_frame.shape[0])
                td_map_frame = self.pad_frame_height(td_map_frame, frontier_frame.shape[0])
                lower_frame = np.concatenate([map_pred_frame, td_map_frame, frontier_frame], axis=1)
                if lower_frame.shape[1] > upper_frame.shape[1]:
                    upper_frame = self.pad_frame(
                        upper_frame,
                        lower_frame.shape[1]
                    )
                else:
                    lower_frame = self.pad_frame(
                        lower_frame,
                        upper_frame.shape[1]
                    )
                frame = np.concatenate([upper_frame, lower_frame], axis=0)
        else:
            obs_frame = self.pad_frame_height(obs_frame, sem_frame.shape[0])
            goal_frame = self.pad_frame_height(goal_frame, sem_frame.shape[0])
            upper_frame = np.concatenate([goal_frame, obs_frame, sem_frame], axis=1)
            map_pred_frame = self.pad_frame_height(map_pred_frame, frontier_frame.shape[0])
            td_map_frame = self.pad_frame_height(td_map_frame, frontier_frame.shape[0])
            lower_frame = np.concatenate([map_pred_frame, td_map_frame, frontier_frame], axis=1)
            # lower_frame = self.pad_frame(
            #     np.concatenate([map_pred_frame, td_map_frame, frontier_frame], axis=1),
            #     upper_frame.shape[1],
            # )
            if lower_frame.shape[1] > upper_frame.shape[1]:
                upper_frame = self.pad_frame(
                    upper_frame,
                    lower_frame.shape[1]
                )
            else:
                lower_frame = self.pad_frame(
                    lower_frame,
                    upper_frame.shape[1]
                )

            frame = np.concatenate([upper_frame, lower_frame], axis=0)
        # end
        
        if goal_text is not None:
            frame = self._put_text_on_image(frame, str(goal_text))
        
        try:
            nframes = 1 if metrics is None else 5
        except Exception as e:
            import pdb; pdb.set_trace()
        for i in range(nframes):
            name = f"snapshot_{timestep}_{i}.png"
            cv2.imwrite(os.path.join(self.vis_dir, name), frame)
    
    # 在图片上添加文本，2025.2.24，wxl
    def _put_text_on_image(
        self,
        vis_image,
        text: str,
        font_scale: int = 0.4,
    ):
        """
        Place text at the center of the given bounding box.
        """
        h, w = vis_image.shape[:2]
        # import ipdb; ipdb.set_trace()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (20, 20, 20)  # BGR
        text_thickness = 1

        textsize = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        bbox_x_len = w  # 文本框宽度与图像宽度一致
        bbox_y_len = int(textsize[1] * 1.5)  # 文本框高度为文本高度的1.5倍

        # 计算文本框位置
        bbox_x_start = (w - bbox_x_len) // 2  # 水平居中
        bbox_y_start = h - bbox_y_len  # 位于图像底部
        # The x coordinate at which the left edge of text needs to be placed
        textX = (bbox_x_len - textsize[0]) // 2 + bbox_x_start
        # The height at which base needs to be placed
        textY = (bbox_y_len + textsize[1]) // 2 + bbox_y_start
        return cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )
    
    # 水平拼接时纵向对齐，wxl
    def pad_frame_height(self, frame: np.ndarray, height: int) -> np.ndarray:
        """Pad the width of a frame to `width` centered white sides."""
        h = frame.shape[0]
        w = frame.shape[1]
        left_bar = np.ones(((height - h)//2, w, 3), dtype=np.uint8) * 255
        right_bar = (
            np.ones(((height - h - left_bar.shape[0]), w, 3), dtype=np.uint8) * 255
        )
        return np.concatenate([left_bar, frame, right_bar], axis=0)
    
    def pad_frame(self, frame: np.ndarray, width: int) -> np.ndarray:
        """Pad the width of a frame to `width` centered white sides."""
        h = frame.shape[0]
        w = frame.shape[1]
        left_bar = np.ones((h, (width - w) // 2, 3), dtype=np.uint8) * 255
        right_bar = (
            np.ones((h, (width - w - left_bar.shape[1]), 3), dtype=np.uint8) * 255
        )
        return np.concatenate([left_bar, frame, right_bar], axis=1)

    def make_keypoint(self, timestep: int) -> np.ndarray:
        """Create the keypoint-matching sub-frame."""
        fname = os.path.join(self.vis_dir, f"superglue_{timestep}.png")
        assert os.path.exists(fname), f"keypoint frame does not exist at `{fname}`."

        border_size = 10
        text_bar_height = 50 - border_size
        kp_img = cv2.imread(fname)
        os.remove(fname)

        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int((new_h / kp_img.shape[0]) * kp_img.shape[1])
        kp_img = cv2.resize(kp_img, (new_w, new_h))

        kp_img = self._add_border(kp_img, border_size)

        w = kp_img.shape[1]
        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, kp_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Keypoint Matching"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def make_goal(self, goal_img: np.ndarray) -> np.ndarray:
        """make the goal image sub-frame."""
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        goal_img = cv2.resize(goal_img, (new_h, new_h))
        goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR)
        goal_img = self._add_border(goal_img, border_size)
        w = goal_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, goal_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Goal Image"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def make_observations(
        self,
        sem_img: np.ndarray,
        collision: bool,
        found_goal: bool,
        metrics: Dict[str, float],
        text = 'Observation',
    ) -> np.ndarray:
        """
        make the egocentric RGB observation sub-frame. Overlay a goal detected banner
        and a collision border.
        """
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / sem_img.shape[0] * sem_img.shape[1])
        sem_img = cv2.resize(sem_img, (new_w, new_h))

        if found_goal:
            sem_img = self._found_goal_detection(sem_img)

        sem_img = self._write_metrics(sem_img, metrics)

        if collision:
            sem_img = draw_collision(sem_img)

        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
        sem_img = self._add_border(sem_img, border_size)
        w = sem_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, sem_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        # text = "Observation"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame


    def make_sem_observations(
        self,
        sem_img: np.ndarray,
        collision: bool,
        found_goal: bool,
        metrics: Dict[str, float],
    ) -> np.ndarray:
        """
        make the egocentric RGB observation sub-frame. Overlay a goal detected banner
        and a collision border.
        """

        semantic_map_vis = Image.new(
            "P", (sem_img.shape[1], sem_img.shape[0])
        )

        # sem_img = sem_img % 255

        semantic_map_vis.putpalette(languagenav_2categories_color_palette)
        semantic_map_vis.putdata(sem_img.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")

        sem_img = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]

        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / sem_img.shape[0] * sem_img.shape[1])

        sem_img = cv2.resize(sem_img, (new_w, new_h))

        if found_goal:
            sem_img = self._found_goal_detection(sem_img)

        sem_img = self._write_metrics(sem_img, metrics)

        if collision:
            sem_img = draw_collision(sem_img)

        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
        sem_img = self._add_border(sem_img, border_size)
        w = sem_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, sem_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Observation"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def draw_gradient_trajectory(
        self,
        image: np.ndarray,
        positions: List[tuple],
        start_color: tuple = (200, 200, 255),  # 浅色 (浅蓝色)
        end_color: tuple = (0, 0, 180),        # 深色 (深红色)
        line_width: int = 2,
    ) -> np.ndarray:
        """
        在地图上绘制渐变颜色的轨迹路径。
        
        Args:
            image: 要绘制的图像
            positions: 轨迹位置列表，每个元素为 (x, y) 像素坐标
            start_color: 起始颜色 (BGR格式)，用于最早的位置
            end_color: 结束颜色 (BGR格式)，用于当前位置
            line_width: 轨迹线宽度
        
        Returns:
            绘制了轨迹的图像
        """
        if len(positions) < 2:
            return image
        
        n_points = len(positions)
        
        # 绘制轨迹线段，每段使用不同的透明度/颜色
        for i in range(1, n_points):
            # 计算插值比例 (0.0 到 1.0)
            alpha = i / (n_points - 1) if n_points > 1 else 1.0
            
            # 线性插值计算当前颜色
            color = tuple(
                int(start_color[c] + alpha * (end_color[c] - start_color[c]))
                for c in range(3)
            )
            
            # 获取前后两个点
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            
            # 绘制线段
            cv2.line(image, pt1, pt2, color, line_width, cv2.LINE_AA)
        
        # 在当前位置绘制一个小圆点，用深色标识
        if len(positions) > 0:
            curr_pos = (int(positions[-1][0]), int(positions[-1][1]))
            cv2.circle(image, curr_pos, line_width + 2, end_color, -1)
        
        return image

    def make_map_preds(
        self,
        sensor_pose: np.ndarray,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        closest_goal_map: np.ndarray,
        goal_map: np.ndarray,
        visualize_goal: bool,
    ) -> np.ndarray:
        """make the predicted map sub-frame."""
        if semantic_map is None:
            fill_val = self.num_sem_categories - 1
            semantic_map = np.zeros_like(obstacle_map) + fill_val

        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)

        # Update visited map with last visited area
        if self.last_xy is not None:
            last_x, last_y = self.last_xy
            last_pose = [
                int(last_y * 100.0 / self.map_resolution - gy1),
                int(last_x * 100.0 / self.map_resolution - gx1),
            ]
            last_pose = pu.threshold_poses(last_pose, obstacle_map.shape)
            curr_pose = [
                int(curr_y * 100.0 / self.map_resolution - gy1),
                int(curr_x * 100.0 / self.map_resolution - gx1),
            ]
            curr_pose = pu.threshold_poses(curr_pose, obstacle_map.shape)
            self.visited_map_vis[gy1:gy2, gx1:gx2] = draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2]
            )
        self.last_xy = (curr_x, curr_y)
        
        # 记录当前位置用于渐变轨迹绘制
        # 存储的是全局坐标，在绘制时会转换为可视化坐标
        self.trajectory_positions.append((curr_x, curr_y))

        semantic_map += PI.SEM_START

        # Obstacles, explored, and visited areas
        no_category_mask = semantic_map == PI.SEM_START + self.num_sem_categories - 1
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = self.visited_map_vis[gy1:gy2, gx1:gx2] == 1
        semantic_map[no_category_mask] = PI.EMPTY_SPACE
        semantic_map[np.logical_and(no_category_mask, explored_mask)] = PI.EXPLORED
        semantic_map[np.logical_and(no_category_mask, obstacle_mask)] = PI.OBSTACLES
        semantic_map[visited_mask] = PI.VISITED

        # Goal
        if visualize_goal:
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(goal_map, selem) != 1
            goal_mask = goal_mat == 1
            semantic_map[goal_mask] = PI.REST_OF_GOAL
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem) != 1
                )
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = PI.CLOSEST_GOAL

        # Semantic categories
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )

        semantic_map_vis.putpalette(MAP_COLOR_PALETTE)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        semantic_map_vis = cv2.resize(
            semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )

        border_size = 10
        text_bar_height = 50 - border_size
        old_h, old_w = semantic_map_vis.shape[:2]
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / semantic_map_vis.shape[0] * semantic_map_vis.shape[1])
        semantic_map_vis = cv2.resize(semantic_map_vis, (new_w, new_h))

        # 绘制渐变轨迹 - 将全局坐标转换为可视化坐标
        if len(self.trajectory_positions) >= 2:
            vis_trajectory = []
            for traj_x, traj_y in self.trajectory_positions:
                # 转换坐标到480x480地图坐标系，然后再缩放到当前可视化尺寸
                vis_x = (traj_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0]
                vis_y = (obstacle_map.shape[1] - traj_y * 100.0 / self.map_resolution + gy1) * 480 / obstacle_map.shape[1]
                # 缩放到最终可视化尺寸
                vis_x = vis_x * new_w / old_w
                vis_y = vis_y * new_h / old_h
                vis_trajectory.append((vis_x, vis_y))
            
            # 绘制渐变轨迹 (浅蓝色 -> 深红色)
            semantic_map_vis = self.draw_gradient_trajectory(
                semantic_map_vis,
                vis_trajectory,
                start_color=(255, 200, 150),  # 浅蓝色 (BGR)
                end_color=(0, 0, 200),        # 深红色 (BGR)
                line_width=2,
            )

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        pos = (pos[0] * new_w / old_w, pos[1] * new_h / old_h, pos[2])
        agent_arrow = get_contour_points(pos, origin=(0, 0))
        color = MAP_COLOR_PALETTE[9:12][::-1]
        cv2.drawContours(semantic_map_vis, [agent_arrow], 0, color, -1)

        # semantic_map_vis = cv2.cvtColor(semantic_map_vis, cv2.COLOR_RGB2BGR)

        # add map outline
        color = [100, 100, 100]
        h, w = semantic_map_vis.shape[:2]
        semantic_map_vis[0, 0:] = color
        semantic_map_vis[h - 1, 0:] = color
        semantic_map_vis[0:, 0] = color
        semantic_map_vis[0:, w - 1] = color

        semantic_map_vis = self._add_border(semantic_map_vis, border_size)
        w = semantic_map_vis.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, semantic_map_vis.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Predicted Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def make_td_map(self, top_down_map: np.ndarray, explore_type = None, explore_info = None) -> np.ndarray:
        """
        In Habitat Simulation, an oracle top-down map may be provided.
        Visualize that sub-frame.
        """
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size

        td_map = maps.colorize_draw_agent_and_fit_to_height(top_down_map, new_h)
        td_map = cv2.cvtColor(td_map, cv2.COLOR_RGB2BGR)
        
        if explore_type is not None:
            td_map = self._found_goal_detection(td_map, _type = explore_type, addtional_info=explore_info)

        # add map outline
        color = [100, 100, 100]
        h, w = td_map.shape[:2]
        td_map[0, 0:] = color
        td_map[h - 1, 0:] = color
        td_map[0:, 0] = color
        td_map[0:, w - 1] = color

        td_map = self._add_border(td_map, border_size)
        w = td_map.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, td_map.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Oracle Top-Down Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def _write_metrics(
        self, frame: np.ndarray, metrics: Dict[str, float]
    ) -> np.ndarray:
        """If metrics are provided, write them on the RGB frame."""
        if metrics is None:
            return frame

        lines = []
        for k, v in {"success": "SR", "spl": "SPL"}.items():
            if k in metrics:
                lines.append(f"{v}: {metrics[k]:.3f}")

        return append_text_to_image_right_align(frame, lines, font_size=0.8)

    def _add_border(self, frame: np.ndarray, border_size: int) -> np.ndarray:
        """Add a white border to a frame."""
        h, w = frame.shape[:2]
        side = np.ones((h, border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([side, frame, side], axis=1)
        top = np.ones((border_size, w + 2 * border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top, frame, top], axis=0)
        return frame

    def _draw_star(
        self, 
        image: np.ndarray, 
        cx: int, 
        cy: int, 
        radius: int = 10, 
        color: tuple = (0, 255, 0), 
        thickness: int = 2
    ) -> None:
        """
        在图像上绘制一个五角星。
        
        Args:
            image: 要绘制的图像
            cx, cy: 星号中心坐标
            radius: 外顶点到中心的距离
            color: 颜色 (BGR)
            thickness: 线条粗细，-1为填充
        """
        outer_radius = radius
        inner_radius = radius * 0.4
        
        # 计算五角星的顶点（从顶部开始，顺时针）
        points = []
        for i in range(10):
            angle = np.pi / 2 + i * np.pi / 5  # 从顶部开始
            r = outer_radius if i % 2 == 0 else inner_radius
            x = int(cx + r * np.cos(angle))
            y = int(cy - r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        if thickness == -1:
            cv2.fillPoly(image, [points], color=color)

    def _add_frontier_frame_label(self, frontier_frame: np.ndarray) -> np.ndarray:
        """为 frontier_frame 添加边框、标签和图例 (Navigable Map)"""
        border_size = 10
        text_bar_height = 50 - border_size
        
        # 调整大小以匹配其他地图帧
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / frontier_frame.shape[0] * frontier_frame.shape[1])
        frontier_frame = cv2.resize(frontier_frame, (new_w, new_h))
        
        # ============ 在地图内部右上角添加图例 ============
        font = cv2.FONT_HERSHEY_SIMPLEX
        legend_font_scale = 0.35
        legend_thickness = 1
        
        legend_items = [
            ((255, 100, 0), "Frontiers", "circle"),         # 蓝色圆圈
            ((0, 255, 0), "Goal Region", "star"),           # 绿色星号（目标区域）
            ((0, 0, 255), "Closest Goal", "filled_circle"), # 红色实心圆
        ]
        
        # 计算图例背景大小
        line_height = 16
        legend_padding = 5
        max_text_width = 0
        for _, item_text, _ in legend_items:
            text_width = cv2.getTextSize(item_text, font, legend_font_scale, legend_thickness)[0][0]
            max_text_width = max(max_text_width, text_width)
        
        legend_width = max_text_width + 25 + legend_padding * 2  # 符号宽度 + 文字 + padding
        legend_height = line_height * len(legend_items) + legend_padding * 2
        
        # 在右上角绘制半透明背景
        legend_x = new_w - legend_width - 5
        legend_y = 5
        
        # 创建半透明白色背景
        overlay = frontier_frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frontier_frame, 0.3, 0, frontier_frame)
        
        # 绘制图例边框
        cv2.rectangle(frontier_frame, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (150, 150, 150), 1)
        
        # 绘制图例项
        for i, (item_color, item_text, item_shape) in enumerate(legend_items):
            item_y = legend_y + legend_padding + i * line_height + line_height // 2 + 2
            symbol_x = legend_x + legend_padding + 6
            
            if item_shape == "circle":
                cv2.circle(frontier_frame, (symbol_x, item_y), 4, item_color, 1)
            elif item_shape == "star":
                self._draw_star(frontier_frame, symbol_x, item_y, radius=5, color=item_color, thickness=1)
            elif item_shape == "filled_circle":
                cv2.circle(frontier_frame, (symbol_x, item_y), 4, item_color, -1)
            
            # 绘制图例文字
            text_x = symbol_x + 10
            cv2.putText(
                frontier_frame, item_text, (text_x, item_y + 3),
                font, legend_font_scale, (30, 30, 30), legend_thickness, cv2.LINE_AA
            )
        # ============ 图例绘制结束 ============
        
        # 添加地图边框线
        color_outline = [100, 100, 100]
        h, w = frontier_frame.shape[:2]
        frontier_frame[0, 0:] = color_outline
        frontier_frame[h - 1, 0:] = color_outline
        frontier_frame[0:, 0] = color_outline
        frontier_frame[0:, w - 1] = color_outline
        
        # 添加白色边框
        frontier_frame = self._add_border(frontier_frame, border_size)
        w = frontier_frame.shape[1]
        
        # 添加标题栏
        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, frontier_frame.astype(np.uint8)], axis=0)
        
        # 绘制标题文字
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2
        
        text = "Navigable Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        
        return frame

    # def _found_goal_detection(self, view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    #     """overlay a green goal detected banner"""
    #     strip_width = view.shape[0] // 15
    #     mask = np.ones(view.shape)
    #     mask[strip_width:-strip_width] = 0
    #     mask = mask == 1
    #     view[mask] = (alpha * np.array([0, 255, 0]) + (1.0 - alpha) * view)[mask]
    #     return append_text_to_image(view, ["Goal Detected"], font_size=0.5)

    def _found_goal_detection(self, view: np.ndarray, alpha: float = 0.4, _type: str = 'found_goal', addtional_info = None) -> np.ndarray:
        """overlay a green goal detected banner"""
        strip_width = view.shape[0] // 15
        mask = np.ones(view.shape)
        mask[strip_width:-strip_width] = 0
        mask = mask == 1
        if _type == 'found_goal':
            view[mask] = (alpha * np.array([0, 255, 0]) + (1.0 - alpha) * view)[mask]
            text = "Goal Detected"
            if addtional_info is not None:
                text += f' | {addtional_info}'
            return append_text_to_image(view, [text], font_size=0.4)
        
        elif _type == 'explore':
            view[mask] = (alpha * np.array([200, 0, 0]) + (1.0 - alpha) * view)[mask]
            text = "Expolre Interesting Area"
            if addtional_info is not None:
                text += f' | {addtional_info}'
            return append_text_to_image(view, [text], font_size=0.4)
        
        else:
            return view
        