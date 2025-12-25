import os
import sys
import time
import argparse
import numpy as np
import open3d as o3d
import spatialmath as sm
import mujoco

# --- anygrasp tracking dependencies ---
from PIL import Image
from graspnetAPI import GraspGroup
from tracker import AnyGraspTracker

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('../../manipulator_grasp')
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'utils'))
from graspnet_baseline.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector

from manipulator_grasp.env.dynamic_y import Yenv
from manipulator_grasp.env.rotate_env import RotateEnv

class anygrasp:
    def __init__(self):
        self.checkpoint_path = os.path.join(ROOT_DIR, 'anygrasp', 'ckp', 'checkpoint_tracking.tar')
        self.filter = 'oneeuro'
        self.debug = False
        # init network
        self.tracker = AnyGraspTracker(self)
        self.tracker.load_net()

        # tracking state
        self.grasp_ids = [0]
        self._has_init = False
        self.max_init_grasps = 1
        self.max_draw_grasps = 6

    def _collision_filter(self, gg: GraspGroup, cloud_xyz: np.ndarray):
        if len(gg) == 0 or cloud_xyz is None or len(cloud_xyz) == 0:
            return gg
        try:
            mfcd = ModelFreeCollisionDetector(cloud_xyz, voxel_size=0.01)
            collision_mask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=0.01)
            return gg[~collision_mask]
        except Exception:
            return gg

    def generate_grasps(self, imgs, num_point=20000):
        """
        Returns:
            cloud_pcd: open3d PointCloud (for visualization)
            target_gg: GraspGroup (tracked grasps, sorted)
            grippers:  list[open3d.geometry.Geometry] (gripper meshes to draw)
            T_co_best: (4,4) np.ndarray or None (best grasp pose in camera frame)
        """
        color = imgs['img'] / 255.0
        depth = imgs['depth']
        height = 256
        width = 256
        fovy = np.pi / 4
        intrinsic = np.array([
            [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
            [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        factor_depth = 1.0

        camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud_img = create_point_cloud_from_depth_image(depth, camera, organized=True)

        mask = (depth > 1e-6) & (depth < 2.0)
        points_full = cloud_img[mask].astype(np.float32)
        colors_full = color[mask].astype(np.float32)

        rgb = color.astype(np.float32)            # (H,W,3) 0~1
        d   = depth.astype(np.float32)            # (H,W)
        valid = (d > 1e-6) & (d < 2.0)

        seg = imgs.get("seg", None)
        obj_gid = int(imgs.get("capsule_gid", -1))
        # 'box_gid'
        # 'cylinder_gid'
        # 'capsule_gid'
        obj_mask = None

        if seg is not None and obj_gid >= 0:
            seg = np.asarray(seg)

            if seg.ndim == 3 and seg.shape[-1] >= 2:
                c0 = seg[..., 0]
                c1 = seg[..., 1]

                m0 = valid & (c0 == obj_gid)
                m1 = valid & (c1 == obj_gid)

                if np.count_nonzero(m0) >= np.count_nonzero(m1):
                    obj_id, obj_type = c0, c1
                else:
                    obj_id, obj_type = c1, c0

                mask_id = valid & (obj_id == obj_gid)

                try:
                    import mujoco
                    GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
                    mask_type = mask_id & (obj_type == GEOM)
                    obj_mask = mask_type if np.count_nonzero(mask_type) > 0.5 * np.count_nonzero(mask_id) else mask_id
                except Exception:
                    obj_mask = mask_id

            else:
                obj_mask = valid & (seg == obj_gid)

        if obj_mask is not None and np.count_nonzero(obj_mask) > 10:
            points_use = cloud_img[obj_mask].astype(np.float32)
            colors_use = rgb[obj_mask].astype(np.float32)
        else:
            empty = GraspGroup(np.zeros((0,17), dtype=np.float32))
            cloud_pcd = o3d.geometry.PointCloud()
            return cloud_pcd, empty, [], None

        cloud_pcd = o3d.geometry.PointCloud()
        if len(points_full) == 0:
            empty = GraspGroup(np.zeros((0, 17), dtype=np.float32))
            return cloud_pcd, empty, [], None

        cloud_pcd.points = o3d.utility.Vector3dVector(points_full)
        cloud_pcd.colors = o3d.utility.Vector3dVector(colors_full)

        # AnyGrasp tracking update using FULL points
        target_gg, curr_gg, target_grasp_ids, corres_preds = self.tracker.update(
            points_use, colors_use, self.grasp_ids
        )

        curr_gg = self._collision_filter(curr_gg, points_full)
        target_gg = self._collision_filter(target_gg, points_full)

        need_reinit = (not self._has_init) or (target_grasp_ids is None) or (len(target_grasp_ids) == 0)

        if need_reinit:
            # NMS + sort
            try: curr_gg.nms()
            except Exception: pass
            try: curr_gg.sort_by_score()
            except Exception: pass

            n = len(curr_gg)
            if n == 0:
                self.grasp_ids = [0]
                self._has_init = False
                return cloud_pcd, curr_gg, [], None
            k = min(self.max_init_grasps, n)
            self.grasp_ids = list(range(k))
            target_gg = curr_gg[self.grasp_ids]
            self._has_init = True
        else:
            self.grasp_ids = list(target_grasp_ids)

        try: target_gg.nms()
        except Exception: pass
        try: target_gg.sort_by_score()
        except Exception: pass
        target_gg = target_gg[: self.max_draw_grasps]

        if len(target_gg) > self.max_draw_grasps:
            target_gg = target_gg[:self.max_draw_grasps]

        grippers = []
        T_co_best = None
        if len(target_gg) > 0:
            # GraspGroup -> Open3D gripper geometry list :contentReference[oaicite:3]{index=3}
            grippers = target_gg.to_open3d_geometry_list()

            R = target_gg.rotation_matrices[0]
            t = target_gg.translations[0]
            T_co_best = np.eye(4, dtype=np.float32)
            T_co_best[:3, :3] = R
            T_co_best[:3, 3] = t

            try:
                grippers[0].paint_uniform_color([0, 1, 0])  # best grasp candidate in green
            except Exception:
                pass

        return cloud_pcd, target_gg, grippers, T_co_best

def _get_T_wc() -> sm.SE3:
    n_wc = np.array([0.0, -1.0, 0.0], dtype=float)
    o_wc = np.array([-1.0, 0.0, -0.5], dtype=float)
    t_wc = np.array([1.0, 0.6, 2.0], dtype=float)
    return sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

def _ikRTB(robot, T_wt: sm.SE3, seed_q: np.ndarray | None = None, verbose: bool = False):
    """RTB ikine_LM"""
    if seed_q is None:
        seed_q = robot.get_joint()
    seed_q = np.asarray(seed_q, dtype=float).reshape(-1)

    ik_result = robot.robot.ikine_LM(T_wt, q0=seed_q)
    if not ik_result.success:
        if verbose:
            print("[IK] RTB ikine_LM failed. residual:", ik_result.residual, "iterations:", ik_result.iterations)
        return None

    q = np.asarray(ik_result.q, dtype=float).reshape(-1)
    if q.size < 6 or (not np.all(np.isfinite(q))):
        return None
    return q

def _build_T_co_from_gg(gg: GraspGroup) -> sm.SE3 | None:
    if gg is None or len(gg) == 0:
        return None
    R = np.asarray(gg.rotation_matrices[0], dtype=float)   # (3,3)
    t = np.asarray(gg.translations[0], dtype=float).reshape(3)
    if R.shape != (3, 3) or (not np.all(np.isfinite(R))) or (not np.all(np.isfinite(t))):
        return None
    return sm.SE3.Trans(t) * sm.SE3(sm.SO3.TwoVectors(x=R[:, 0], y=R[:, 1]))

def interpolate(env: TestEnv, robot, q_from, q_to, steps: int, action: np.ndarray, grip: float):
    """Interpolate motion planning"""
    q_from = np.asarray(q_from, dtype=float).reshape(-1)
    q_to   = np.asarray(q_to,   dtype=float).reshape(-1)
    for k in range(steps):
        a = (k + 1) / float(steps)
        q = (1.0 - a) * q_from + a * q_to
        robot.move_joint(q)
        action[:6] = q
        action[6] = grip
        env.step(action)

def joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.01):
    if q_des is not None:
        dq = np.clip(q_des - q_cmd, -max_step, max_step)
        q_cmd = q_cmd + dq

    robot.move_joint(q_cmd)
    action[:6] = q_cmd
    action[6] = grip
    env.step(action)
    return q_cmd

def main():
    env = RotateEnv()
    env.reset()

    Anygrasp = anygrasp()

    # constants
    dt = float(env.mj_model.opt.timestep)         # 0.002
    grasp_hz = 10.0
    steps_per_update = max(1, int(round(1.0 / (grasp_hz * dt))))  # 10Hz -> 50 steps
    action = np.zeros(7, dtype=float)

    # gipper state
    grip_open = 0.0
    grip_close = 255.0

    # pre-grasp pose
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=float)

    # camera extrinsic (world <- camera)
    T_wc = _get_T_wc()

    lookahead = 0.18
    beta = 0.6
    dist_close = 0.035
    # pre_offset = -0.1 # works well for box and cylinder
    pre_offset = -0.08  # works well for capsule
    attempt_time = 8.0

    close_steps = int(round(0.20 / dt))
    hold_steps  = int(round(0.20 / dt))
    lost_max = 5

    try:
        # move to pre-grasp pose
        robot = env.robot
        q0 = robot.get_joint()
        action[:6] = q0
        action[6] = grip_open
        interpolate(env, robot, q0, q1, int(round(2.0 / dt)), action, grip_open)

        q_cmd = robot.get_joint()

        mode = "track"
        # Grasp flag
        grasp_success = False
    
        grip = grip_open
        close_i = 0
        hold_i = 0
        lost = 0

        R_lock = None
        q_des = None
        last_t = None
        v_filt = np.zeros(3)

        for _ in range(int(attempt_time * grasp_hz)):

            imgs = env.render()

            _, target_gg, _, _ = Anygrasp.generate_grasps(imgs, num_point=20000)
            T_co = _build_T_co_from_gg(target_gg)

            if T_co is None:
                lost += 1
                for _ in range(steps_per_update):
                    q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.01)
                if lost >= lost_max:
                    break
                continue
            lost = 0

            T_wo  = T_wc * T_co

            t = np.array(T_wo.t).reshape(3)
            if last_t is not None:
                v = (t - last_t) * grasp_hz   # dt = 1/grasp_hz
                v_filt = beta * v_filt + (1 - beta) * v
            last_t = t

            t_pred = t + v_filt * lookahead

            if R_lock is None:
                R_lock = sm.SO3(T_wo.R)
            T_wo_pred = sm.SE3.Rt(R_lock.R, t_pred)
            # track -> approach -> close
            if mode == "track":
                offset = pre_offset
                T_tar = T_wo_pred * sm.SE3.Tx(offset)
            elif mode == "approach":
                offset = 0.0
                T_tar = T_wo_pred * sm.SE3.Tx(offset)
            else:
                # close/hold
                offset = 0.0
                T_tar = T_wo_pred * sm.SE3.Tx(offset)

            # IK
            q_new = _ikRTB(robot, T_tar, seed_q=q_cmd)
            if q_new is not None:
                q_des = q_new

            # close
            if mode == "track":
                ee = robot.get_cartesian()
                ee_t = np.array(ee.t).reshape(3)
                # approach
                t_pre = np.array((T_wo_pred * sm.SE3.Tx(pre_offset)).t).reshape(3)
                if np.linalg.norm(ee_t - t_pre) < dist_close:
                    mode = "approach"
                    # Wait for approching complete
                    close_i = 0
            elif mode == "approach":
                ee = robot.get_cartesian()
                ee_t = np.array(ee.t).reshape(3)
                t_grasp = np.array(T_wo_pred.t).reshape(3)
                if np.linalg.norm(ee_t - t_grasp) < dist_close:
                    mode = "close"
                    close_i = 0

            for _ in range(steps_per_update):
                if mode == "track":
                    grip = grip_open
                elif mode == "close":
                    close_i += 1
                    a = min(1.0, close_i / float(close_steps))
                    grip = grip_open + (grip_close - grip_open) * a
                    if close_i >= close_steps:
                        mode = "hold"
                        grip = grip_close
                        hold_i = 0
                elif mode == "hold":
                    hold_i += 1
                    grip = grip_close

                q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.01)

                if mode == "hold" and hold_i >= hold_steps:
                    grasp_success = True
                    break

            if mode == "hold" and hold_i >= hold_steps:
                grasp_success = True
                break

            print("[mode]", mode)
        # go back to q0 after grasp
        if grasp_success:
            q_curr = q_cmd.copy()

            interpolate(env, robot, q_curr, q1, int(round(1.0 / dt)), action, grip_close)
            # hold for 5s
            for _ in range(int(round(5.0 / dt))):
                action[:6] = q1
                action[6]  = grip_close
                env.step(action)

    finally:
        env.close()

def visual_anygrasp():   # For visualizing AnyGrasp tracking only
    """AnyGrasp tracking"""
    env = RotateEnv()
    # env = Yenv()
    env.reset()

    # AnyGrasp
    Anygrasp = anygrasp()
    gripper_geoms = []
    last_T_co_best = None  # For motion planner

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud", width=1280, height=720, visible=True)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd, reset_bounding_box=True)

    opt = vis.get_render_option()
    opt.point_size = 2.0

    pcd_hz = 10.0
    steps_per_update = max(1, int(env.sim_hz / pcd_hz))
    period = 1.0 / pcd_hz
    next_t = time.time()

    first_frame = True

    try:
        while vis.poll_events():
            # advance sim
            for _ in range(steps_per_update):
                env.step()

            imgs = env.render()
            cloud_pcd, target_gg, grippers, T_co_best = Anygrasp.generate_grasps(imgs, num_point=20000)

            # update point cloud
            pcd.points = cloud_pcd.points
            pcd.colors = cloud_pcd.colors
            vis.update_geometry(pcd)

            # update grippers (remove old, add new)
            for g in gripper_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)  # keep viewpoint :contentReference[oaicite:1]{index=1}
            gripper_geoms = []
            for g in grippers:
                vis.add_geometry(g, reset_bounding_box=False)     # keep viewpoint :contentReference[oaicite:2]{index=2}
                vis.update_geometry(g)
                gripper_geoms.append(g)

            # first frame: reset view once after non-empty cloud
            if first_frame and len(cloud_pcd.points) > 0:
                vis.reset_view_point(True)  # recompute bbox + view :contentReference[oaicite:3]{index=3}
                first_frame = False

            vis.update_renderer()

            # keep fixed update rate
            last_T_co_best = T_co_best  # For planner, T_wo = T_wc * T_co
            next_t += period
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        vis.destroy_window()
        env.close()

if __name__ == "__main__":
    main()
    # visual_anygrasp()