"""
PBVS track and grasp using anygrasp
"""
import os
import sys
import time
import argparse
import numpy as np
import open3d as o3d
import spatialmath as sm
import mujoco

from PIL import Image
from graspnetAPI import GraspGroup
from tracker import AnyGraspTracker

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('../../manipulator_grasp')
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'utils'))
from graspnet_baseline.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from manipulator_grasp.env.dynamic_yz import YZEnv

class anygrasp:
    def __init__(self):
        self.checkpoint_path = os.path.join(ROOT_DIR, 'anygrasp', 'ckp', 'checkpoint_tracking.tar')
        self.filter = 'oneeuro'
        self.debug = False

        # --- init network ---
        self.tracker = AnyGraspTracker(self)
        self.tracker.load_net()

        # --- tracking state ---
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

    def generate_grasps(self, imgs):
        """
        Returns:
            cloud_pcd: open3d PointCloud (for visualization)
            target_gg: GraspGroup (tracked grasps, sorted)
            grippers:  list[open3d.geometry.Geometry] (gripper meshes to draw)
            T_co_best: (4,4) np.ndarray or None (best grasp pose in camera frame)
        """
        # --- RGB-D -> points/colors ---
        color = imgs["img"].astype(np.float32) / 255.0
        depth = imgs["depth"].astype(np.float32)

        height, width = depth.shape[:2]
        fovy = np.pi / 4
        K = imgs.get("K", None)
        if K is not None:
            fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        else:
            fovy = imgs.get("fovy", np.pi/4)
            fy = height / (2.0 * np.tan(fovy / 2.0))
            fx = fy
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0

        camera = CameraInfo(height, width, fx, fy, cx, cy, 1.0)
        cloud_img = create_point_cloud_from_depth_image(depth, camera, organized=True)

        mask = (depth > 1e-6) & (depth < 2.0)
        points_full = cloud_img[mask].astype(np.float32)
        colors_full = color[mask].astype(np.float32)

        rgb = color.astype(np.float32)
        d   = depth.astype(np.float32)
        valid = (d > 1e-6) & (d < 2.0)

        seg = imgs.get("seg", None)
        box_gid = int(imgs.get("box_gid", -1))
        obj_mask = None

        if seg is not None and box_gid >= 0:
            seg = np.asarray(seg)
            if seg.ndim == 3 and seg.shape[-1] >= 2:
                c0 = seg[..., 0]
                c1 = seg[..., 1]
                m0 = valid & (c0 == box_gid)
                m1 = valid & (c1 == box_gid)
                if np.count_nonzero(m0) >= np.count_nonzero(m1):
                    obj_id, obj_type = c0, c1
                else:
                    obj_id, obj_type = c1, c0
                mask_id = valid & (obj_id == box_gid)
                try:
                    import mujoco
                    GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
                    mask_type = mask_id & (obj_type == GEOM)
                    obj_mask = mask_type if np.count_nonzero(mask_type) > 0.5 * np.count_nonzero(mask_id) else mask_id
                except Exception:
                    obj_mask = mask_id
            else:
                obj_mask = valid & (seg == box_gid)

        if obj_mask is not None and np.count_nonzero(obj_mask) > 20:    # 80
            points_use = cloud_img[obj_mask].astype(np.float32)
            colors_use = rgb[obj_mask].astype(np.float32)
        else:
            empty = GraspGroup(np.zeros((0, 17), dtype=np.float32))
            cloud_pcd = o3d.geometry.PointCloud()
            return cloud_pcd, empty, [], None
        
        cloud_pcd = o3d.geometry.PointCloud()
        if len(points_full) == 0:
            empty = GraspGroup(np.zeros((0, 17), dtype=np.float32))
            return cloud_pcd, empty, [], None

        cloud_pcd.points = o3d.utility.Vector3dVector(points_full)
        cloud_pcd.colors = o3d.utility.Vector3dVector(colors_full)

        # --- AnyGrasp tracking update ---
        target_gg, curr_gg, target_grasp_ids, corres_preds = self.tracker.update(
            points_use, colors_use, self.grasp_ids
        )

        curr_gg   = self._collision_filter(curr_gg, points_use)
        target_gg = self._collision_filter(target_gg, points_use)

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
            grippers = target_gg.to_open3d_geometry_list()
            # translations only
            t = target_gg.translations[0]
            T_co_best = np.eye(4, dtype=np.float32)
            T_co_best[:3, 3] = t
            try:
                grippers[0].paint_uniform_color([0, 1, 0])  # best grasp candidate in green
            except Exception:
                pass

        return cloud_pcd, target_gg, grippers, T_co_best

def _build_T_co_from_gg(gg: GraspGroup) -> sm.SE3 | None:
    if gg is None or len(gg) == 0:
        return None
    t = np.asarray(gg.translations[0], dtype=float).reshape(3)
    if not np.all(np.isfinite(t)):
        return None
    return sm.SE3.Trans(t)

def interpolate(env, robot, q_from, q_to, steps: int, action: np.ndarray, grip: float):
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
    q_cmd = np.asarray(robot.get_joint(), dtype=float).reshape(-1)

    if q_des is not None:
        q_des = np.asarray(q_des, dtype=float).reshape(-1)
        if (q_des.shape[0] < 6) or (not np.all(np.isfinite(q_des))):
            q_des = None

    if q_des is not None:
        dq = np.clip(q_des - q_cmd, -max_step, max_step)
        q_cmd = q_cmd + dq

    if not np.all(np.isfinite(q_cmd)):
        q_cmd = np.asarray(robot.get_joint(), dtype=float).reshape(-1)

    robot.move_joint(q_cmd)
    action[:6] = q_cmd
    action[6] = grip

    try:
        env.step(action)
    except mujoco.FatalError as e:
        import traceback
        print("\n[MUJOCO FatalError]", e)
        print("action finite:", np.all(np.isfinite(action)), "action:", action)
        print("qpos finite:", np.all(np.isfinite(env.mj_data.qpos)))
        print("qvel finite:", np.all(np.isfinite(env.mj_data.qvel)))
        traceback.print_exc()
        raise
    return q_cmd

def so3_logvec(R: np.ndarray) -> np.ndarray:
    # axis-angle vector: u*theta
    tr = np.trace(R)
    cosang = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cosang)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / np.sin(theta))
    return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

def pbvs_rmrc_step(robot,
                   T_wc: sm.SE3,
                   T_cg_des: sm.SE3,
                   q_seed: np.ndarray,
                   dt_cmd: float,
                   kp_t: float = 1.5,
                   kp_R: float = 1.0,
                   lam: float = 0.06,
                   v_max: float = 0.25,
                   w_max: float = 1.2,
                   rot_enable: bool = True,
                   v_ff_w: np.ndarray | None = None,
                   w_ff_w: np.ndarray | None = None):
    q = np.asarray(q_seed, dtype=float).reshape(-1)
    T_we = robot.robot.fkine(q)

    # desired end-effector pose in world
    T_wg_des = T_wc * T_cg_des

    # position error in world
    t_err = np.asarray(T_wg_des.t - T_we.t).reshape(3)

    v_fb = kp_t * t_err
    if v_ff_w is not None:
        v_fb = v_fb + np.asarray(v_ff_w, dtype=float).reshape(3)
    v_fb = np.clip(v_fb, -v_max, v_max)

    J = robot.robot.jacob0(q)

    if not rot_enable:
        Jv = J[:3, :]                          # linear part
        A = Jv @ Jv.T + (lam ** 2) * np.eye(3)
        try:
            qdot = Jv.T @ np.linalg.solve(A, v_fb)
        except np.linalg.LinAlgError:
            return q, t_err, np.zeros(3)
        q_next = q + qdot * dt_cmd
        return q_next, t_err, np.zeros(3)

    # original 6D branch (keep)
    R_err = np.asarray(T_wg_des.R @ T_we.R.T)
    r_err = so3_logvec(R_err)

    w_fb = np.clip(kp_R * r_err, -w_max, w_max)
    if w_ff_w is not None:
        w_fb = w_fb + np.asarray(w_ff_w, dtype=float).reshape(3)

    xi_w = np.hstack((v_fb, w_fb))

    A = J @ J.T + (lam ** 2) * np.eye(6)
    try:
        qdot = J.T @ np.linalg.solve(A, xi_w)
    except np.linalg.LinAlgError:
        return q, t_err, r_err

    q_next = q + qdot * dt_cmd
    return q_next, t_err, r_err

def rmrc_step(robot, T_tar: sm.SE3, q_seed: np.ndarray, dt_cmd: float,
              kp_p: float = 3.0, kp_R: float = 2.0, lam: float = 0.06,
              rot_enable: bool = True, v_ff: np.ndarray | None = None):
    """
    Resolved-rate pose servo (world/base frame), output a small q_des step.
    - kp_p: 位置伺服增益 (1/s)
    - kp_R: 姿态伺服增益 (1/s)
    - lam:  DLS 阻尼 (越大越稳但越慢)
    """
    q = np.asarray(q_seed, dtype=float).reshape(-1)

    T_we = robot.robot.fkine(q)
    p_e = np.asarray(T_we.t).reshape(3)
    p_d = np.asarray(T_tar.t).reshape(3)
    e_p = (p_d - p_e)

    R_e = np.asarray(T_we.R)
    R_d = np.asarray(T_tar.R)

    # 小角度姿态误差：0.5*vee(R_e^T R_d - R_d^T R_e) 的等价写法（数值更稳）
    e_R = 0.5 * (np.cross(R_e[:, 0], R_d[:, 0]) +
                 np.cross(R_e[:, 1], R_d[:, 1]) +
                 np.cross(R_e[:, 2], R_d[:, 2]))
    if not rot_enable:
        e_R[:] = 0.0

    if v_ff is None:
        v_ff = np.zeros(3)
    else:
        v_ff = np.asarray(v_ff, dtype=float).reshape(3)

    v_ff = np.clip(v_ff, -0.3, 0.3)

    xi = np.hstack((v_ff + kp_p * e_p, kp_R * e_R))
    if not np.all(np.isfinite(xi)):
        return q  # hold

    J = robot.robot.jacob0(q)
    if not np.all(np.isfinite(J)):
        return q  # hold

    A = J @ J.T + (lam ** 2) * np.eye(6)
    try:
        qdot = J.T @ np.linalg.solve(A, xi)
    except np.linalg.LinAlgError:
        return q  # hold

    if not np.all(np.isfinite(qdot)):
        return q  # hold

    return q + np.clip(qdot, -3.0, 3.0) * dt_cmd

def main():
    # env = MeteEnv()
    env = YZEnv()
    env.reset()

    pad_gids = set()
    for n in ("right_pad1", "right_pad2", "left_pad1", "left_pad2"):
        gid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            pad_gids.add(int(gid))

    gripper_gids = getattr(env, "_gripper_geom_ids", None)
    if gripper_gids is None:
        gripper_gids = getattr(env, "gripper_geom_ids", set())

    left_pad_ids = []
    right_pad_ids = []
    for n in ("left_pad1", "left_pad2"):
        gid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            left_pad_ids.append(int(gid))
    for n in ("right_pad1", "right_pad2"):
        gid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            right_pad_ids.append(int(gid))

    Anygrasp = anygrasp()

    # --- constants ---
    dt = float(env.mj_model.opt.timestep)         # 0.002
    grasp_hz = 20.0
    steps_per_update = max(1, int(round(1.0 / (grasp_hz * dt))))  # 10Hz -> 50 steps
    action = np.zeros(7, dtype=float)

    dt_meas = steps_per_update * dt   # ≈ 1/grasp_hz

    # gipper state
    grip_open = 0.0
    grip_close = 255.0

    # pre-grasp pose
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=float)

    # camera extrinsic (world <- camera)
    T_wc = env.T_wc

    lookahead = 0.18    # 0.18

    # Close threshold
    dist_close = 0.035  # 0.035
    t_tol_close = 0.023 

    pre_offset = 0.0    # original shoulde be fine with 0.0
    # pre_offset = -0.08
    attempt_time = 20.0
    close_steps = int(round(0.10 / dt))
    hold_steps  = int(round(0.20 / dt))
    lost_max = 5
    blind_lost_max = 30
    jump_thresh = 0.25

    stable_need = 1 # 3
    stable_cnt = 0

    try:
        # --- move to pre-grasp pose ---
        robot = env.robot
        q0 = robot.get_joint()
        action[:6] = q0
        action[6] = grip_open
        interpolate(env, robot, q0, q1, int(round(2.0 / dt)), action, grip_open)

        q_cmd = robot.get_joint()
        R_lock = sm.SO3(robot.robot.fkine(q_cmd).R)

        mode = "track"
        # Grasp flag
        grasp_success = False
        gap = False

        grip = grip_open
        close_i = 0
        hold_i = 0
        lost = 0
        lost_blind = 0
        blind_close = False

        T_wo_pred_last = None
        q_des = None
        last_t = None
        v_filt = np.zeros(3)

        for _ in range(int(attempt_time * grasp_hz)):

            imgs = env.render()

            _, target_gg, _, _ = Anygrasp.generate_grasps(imgs)
            T_co = _build_T_co_from_gg(target_gg)

            if T_co is None:
                # blind_close: 遮挡但已“入袋” -> 不依赖视觉，保持关节并闭合
                if blind_close and mode in ("close", "hold"):
                    for _ in range(steps_per_update):
                        if mode == "close":
                            close_i += 1
                            a = min(1.0, close_i / float(close_steps))
                            grip = grip_open + (grip_close - grip_open) * a
                            if close_i >= close_steps:
                                mode = "hold"
                                grip = grip_close
                                hold_i = 0
                        else:  # hold
                            hold_i += 1
                            grip = grip_close

                        q_cmd = joint_servo_step(env, robot, q_cmd, None, action, grip, max_step=0.015)

                        if mode == "hold" and hold_i >= hold_steps:
                            grasp_success = True
                            break

                    if grasp_success:
                        break
                    continue

                # --- PATCH 1: hold 阶段丢观测 -> 不 abort，继续 hold ---
                if (not blind_close) and mode == "hold":
                    lost = min(lost + 1, lost_max)
                    lost_blind = 0
                    print(f"[lost] T_co None, lost={lost}/{lost_max}, mode={mode} (keep hold)")

                    for _ in range(steps_per_update):
                        hold_i += 1
                        grip = grip_close
                        q_cmd = joint_servo_step(env, robot, q_cmd, None, action, grip, max_step=0.015)
                        if hold_i >= hold_steps:
                            grasp_success = True
                            break

                    if grasp_success:
                        break
                    continue

                # --- PATCH 2: 仅 close 阶段丢观测才 abort grasp ---
                if (not blind_close) and mode == "close" and (T_wo_pred_last is not None):
                    lost = min(lost + 1, lost_max)
                    lost_blind = 0
                    print(f"[lost] T_co None, lost={lost}/{lost_max}, mode={mode} -> abort grasp")

                    mode = "track"
                    close_i = 0
                    hold_i = 0
                    stable_cnt = 0
                    grip = grip_open

                    for _ in range(steps_per_update):
                        R_use = (R_lock.R if (R_lock is not None) else np.asarray(T_wo_pred_last.R, dtype=float))
                        t_last = np.asarray(T_wo_pred_last.t).reshape(3)
                        T_wo_pred_last = sm.SE3.Rt(R_use, t_last + v_filt * dt)

                        T_tar_i = T_wo_pred_last * sm.SE3.Tx(pre_offset)
                        q_des = rmrc_step(
                            robot, T_tar_i,
                            q_seed=robot.get_joint(),
                            dt_cmd=dt,
                            kp_p=3.0, kp_R=2.0, lam=0.06,
                            rot_enable=False,
                            v_ff=v_filt,
                        )
                        q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip_open, max_step=0.015)
                    continue

                lost = min(lost + 1, lost_max)
                if mode == "track":
                    lost_blind = min(lost_blind + 1, blind_lost_max)
                else:
                    lost_blind = 0
                # blind close to avoid infinite loop
                if mode == "track" and lost_blind >= blind_lost_max:
                    target_gid = int(imgs.get("box_gid", -1))
                    if 0 <= target_gid < env.mj_model.ngeom:
                        T_we_now = robot.robot.fkine(robot.get_joint())
                        t_tcp = np.asarray(T_we_now.t).reshape(3)
                        R_we  = np.asarray(T_we_now.R)

                        t_obj = np.asarray(env.mj_data.geom_xpos[target_gid]).reshape(3)
                        e_tcp = float(np.linalg.norm(t_obj - t_tcp))

                        # 是否与 TCP 附近的“夹爪几何”发生接触（说明被拦住/在袋内）
                        has_contact = False
                        for ci in range(env.mj_data.ncon):
                            c = env.mj_data.contact[ci]
                            g1 = int(c.geom1); g2 = int(c.geom2)
                            if g1 == target_gid:
                                g_other = g2
                            elif g2 == target_gid:
                                g_other = g1
                            else:
                                continue

                            gpos = np.asarray(env.mj_data.geom_xpos[g_other]).reshape(3)
                            if np.linalg.norm(gpos - t_tcp) < 0.12:
                                has_contact = True
                                break

                        # 单侧接触异常 -> 不盲闭合
                        if (e_tcp < dist_close) and has_contact:
                            print("[WARN] lost>=lost_max but object likely captured -> blind_close")
                            blind_close = True
                            mode = "close"
                            close_i = 0
                            hold_i = 0
                            stable_cnt = 0
                            lost_blind = 0

                            # 立刻执行一段闭合（本周期就推进），避免还要等下一帧
                            for _ in range(steps_per_update):
                                close_i += 1
                                a = min(1.0, close_i / float(close_steps))
                                grip = grip_open + (grip_close - grip_open) * a
                                if close_i >= close_steps:
                                    mode = "hold"
                                    grip = grip_close
                                    hold_i = 0
                                q_cmd = joint_servo_step(env, robot, q_cmd, None, action, grip, max_step=0.015)
                            continue

                if lost >= lost_max:
                    gap = True
                    stable_cnt = 0
                    Anygrasp._has_init = False
                    Anygrasp.grasp_ids = [0]
                print(f"[lost] T_co None, lost={lost}/{lost_max}, mode={mode}")
                # stable_cnt = 0

                if T_wo_pred_last is not None:
                    stable_cnt = 0

                    for _ in range(steps_per_update):
                        offset_i = pre_offset
                        grip = grip_open
                        T_tar_i = T_wo_pred_last * sm.SE3.Tx(offset_i)

                        q_des = rmrc_step(
                            robot, T_tar_i, robot.get_joint(), dt,
                            rot_enable=False, v_ff=None
                        )
                        q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.015)
                    continue

                q_des = None
                for _ in range(steps_per_update):
                    q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip_open, max_step=0.01)
                continue

            # Whenever we get T_co from anygrasp
            lost = 0
            lost_blind = 0
            blind_close = False
            T_wo  = T_wc * T_co

            t = np.array(T_wo.t).reshape(3)

            t_prev = last_t

            if gap:
                # [关键] 恢复第一帧只重置基准，不追
                last_t = t.copy()
                v_filt[:] = 0.0
                stable_cnt = 0
                q_des = None
                gap = False
                for _ in range(steps_per_update):
                    q_cmd = joint_servo_step(
                        env,
                        robot,
                        q_cmd,
                        q_des,
                        action,
                        (grip_close if mode in ("close", "hold") else grip_open),
                        max_step=0.01)
                continue

            if t_prev is not None and np.linalg.norm(t - t_prev) > jump_thresh:
                q_des = None
                gap = True
                grip = grip_close if mode in ("close", "hold") else grip_open
                close_i = 0
                hold_i = 0

                lost += 1
                # reset prediction states
                last_t = None
                v_filt[:] = 0.0

                # force anygrasp tracker re-init next frame
                Anygrasp._has_init = False
                Anygrasp.grasp_ids = [0]

                for _ in range(steps_per_update):
                    q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.01)

                if lost >= lost_max:
                    print("[warn] jump lost_max reached -> reinit tracker and keep running")
                    lost = 0
                    stable_cnt = 0
                    gap = True
                    Anygrasp._has_init = False
                    Anygrasp.grasp_ids = [0]
                    continue
                continue

            if t_prev is None:
                v_filt[:] = 0.0
            else:
                v_meas = (t - t_prev) / dt_meas
                alpha = 0.6  # 越大越平滑，越小越跟得紧
                v_filt[:] = alpha * v_filt + (1.0 - alpha) * v_meas

            last_t = t

            t_pred = t + v_filt * lookahead

            T_wo_pred = sm.SE3.Rt(R_lock.R, t_pred)
            T_wo_pred_last = T_wo_pred
            t_pred0 = t_pred.copy()

            la_hold = lookahead
            # every sim step
            for i in range(steps_per_update):

                t_base_i = t + v_filt * la_hold
                t_pred_i = t_base_i + v_filt * ((i + 1) * dt)
                T_wo_pred_i = sm.SE3.Rt(R_lock.R, t_pred_i)

                # gripper state machine（保留你原来的）
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

                if mode == "track":
                    offset_i = pre_offset
                    T_cg_des_i = T_wc.inv() * T_wo_pred_i
                    if offset_i != 0.0:
                        T_cg_des_i = T_cg_des_i * sm.SE3.Tx(offset_i)

                    q_des, t_err, r_err = pbvs_rmrc_step(
                        robot, T_wc, T_cg_des_i,
                        q_seed=robot.get_joint(), dt_cmd=dt,
                        kp_t=1.8,
                        kp_R=0.0,
                        lam=0.06,
                        v_max=0.8,
                        w_max=0.0,
                        rot_enable=False,
                        v_ff_w=v_filt,
                    )
                    q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip_open, max_step=0.015)

                    # --- track -> close ---
                    if i == steps_per_update - 1:
                        T_we_now = robot.robot.fkine(robot.get_joint())
                        t_tcp = np.asarray(T_we_now.t).reshape(3)
                        R_we  = np.asarray(T_we_now.R)

                        target_gid = int(imgs.get("box_gid", -1))
                        if 0 <= target_gid < env.mj_model.ngeom:
                            t_obj = np.asarray(env.mj_data.geom_xpos[target_gid]).reshape(3)
                        else:
                            # fallback：别用 t_err（可能是旧的），用当前预测目标位置
                            t_obj = np.asarray(T_wo_pred_i.t).reshape(3)

                        # AnyGrasp candidate（当前预测 grasp 点）在 TCP 坐标系下的位置
                        t_cand = np.asarray(T_wo_pred_i.t).reshape(3)
                        p_cand = R_we.T @ (t_cand - t_tcp)

                        # --- bag test: object center must lie between left/right pads ---
                        bag_ok = True
                        dist_ref = None  # distance to "mouth center" (pad midpoint)

                        if len(left_pad_ids) > 0 and len(right_pad_ids) > 0:
                            pL = np.mean(np.asarray(env.mj_data.geom_xpos[left_pad_ids]), axis=0)
                            pR = np.mean(np.asarray(env.mj_data.geom_xpos[right_pad_ids]), axis=0)
                            dLR = pR - pL
                            w = float(np.linalg.norm(dLR))
                            if w > 1e-6:
                                u = dLR / w
                                mid = 0.5 * (pL + pR)            # gripper mouth center
                                s = float(abs(np.dot(t_obj - mid, u)))  # lateral offset across jaws
                                # 0.9 留一点边界，避免“刚擦边”被当成袋内
                                bag_ok = (s < 0.5 * w * 0.9)
                                dist_ref = float(np.linalg.norm(t_obj - mid))

                        # --- close trigger: prefer GT object center; do NOT OR with p_cand ---
                        if dist_ref is None:
                            # fallback: still use TCP frame constraint if pads not found
                            p_obj = R_we.T @ (t_obj - t_tcp)
                            ok_pose = (
                                (abs(p_obj[0]) < t_tol_close) and
                                (abs(p_obj[1]) < t_tol_close) and
                                (abs(p_obj[2]) < dist_close)
                            )
                        else:
                            ok_pose = (dist_ref < dist_close) and bag_ok

                        if ok_pose:
                            stable_cnt += 1
                        else:
                            stable_cnt = 0

                        # if can grasp
                        if stable_cnt >= stable_need:
                            mode = "close"
                            stable_cnt = 0
                            close_i = 0
                            hold_i = 0

                    continue

                # ========== CLOSE / HOLD ==========
                T_tar_i = T_wo_pred_i * sm.SE3.Tx(0.0)
                q_des = rmrc_step(
                    robot, T_tar_i,
                    q_seed=robot.get_joint(),
                    dt_cmd=dt,
                    kp_p=3.0, kp_R=2.0, lam=0.06,
                    rot_enable=False,
                    v_ff=v_filt,
                )
                q_cmd = joint_servo_step(env, robot, q_cmd, q_des, action, grip, max_step=0.015)

                if mode == "hold" and hold_i >= hold_steps:
                    grasp_success = True
                    break

            if grasp_success:
                break

            print("[mode]", mode)

        # --- go back to q0 after grasp ---
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

if __name__ == "__main__":
    main()
