"""
Env: box moving along yz plane
"""
import os.path
import sys

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj
from manipulator_grasp.arm.geometry.shape.brick import Brick

class YZEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()
        self.cam_name = "cam"
        self.cam_id = -1
        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096
        self.mj_seg_renderer: mujoco.Renderer = None
        self.cam_id = -1
        self.box_geom_id = -1

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'dynamic_yz.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        self.robot_q = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, -np.pi / 2 * 0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q))
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        self.cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        # for segmentation
        self.box_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Box")
        if self.cam_id < 0:
            raise RuntimeError(f"Camera '{self.cam_name}' not found in XML")
        self.fovy = np.deg2rad(float(self.mj_model.cam_fovy[self.cam_id]))

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_seg_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        
        self.mj_renderer.update_scene(self.mj_data, camera=self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, camera=self.cam_id)
        self.mj_depth_renderer.enable_depth_rendering()
        self.mj_seg_renderer.enable_segmentation_rendering()
        # update scene with the correct camera
        self.mj_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_seg_renderer.update_scene(self.mj_data, self.cam_id)

        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # === camera intrinsics ===
        fy = self.height / (2.0 * np.tan(self.fovy / 2.0))
        fx = fy
        cx = (self.width  - 1) / 2.0
        cy = (self.height - 1) / 2.0
        self.camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # observation = self._get_obs()
        observation = None

        # --- motion config ---
        self.cyl_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Box")
        self.cyl_qposadr = int(self.mj_model.jnt_qposadr[self.cyl_joint_id])  # free joint -> 7 qpos

        self.grasp_update = 10.0    # in hz
        self.cyl_speed = 0.02 * self.grasp_update   # v=0.2m/s(10Hz)

        # YZ plane motion config
        adr = self.cyl_qposadr
        self.cyl_x0 = float(self.mj_data.qpos[adr + 0])
        self.cyl_quat0 = self.mj_data.qpos[adr + 3: adr + 7].copy()

        # y range
        # y range
        self.cyl_y_max = 1.0
        self.cyl_y_min = 0.2
        # free joint qpos layout: [x, y, z, qw, qx, qy, qz] :contentReference[oaicite:2]{index=2}
        eps = 0.02  # optional: avoid spawning exactly at the boundary
        y_lo = self.cyl_y_min + eps
        y_hi = self.cyl_y_max - eps
        if y_hi <= y_lo:
            y_lo, y_hi = self.cyl_y_min, self.cyl_y_max
        self.mj_data.qpos[adr + 1] = float(np.random.uniform(y_lo, y_hi))

        # make sure derived states are consistent after directly editing qpos :contentReference[oaicite:3]{index=3}
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.cyl_dir_y = -1.0  # start moving toward y_min

        # z range
        self.cyl_z_min = 0.8    # table surface
        self.cyl_z_max = 1.0

        # speed |v| = self.cyl_speed
        # default is 45° vy = vz = v/sqrt(2)
        v = float(self.cyl_speed)
        self.cyl_speed_y = v / np.sqrt(2.0)
        self.cyl_speed_z = v / np.sqrt(2.0)
        self.cyl_dir_z = 1.0  # start moving toward z_max

        self.mj_data.qpos[adr + 2] = float(np.clip(self.mj_data.qpos[adr + 2],
                                                   self.cyl_z_min, self.cyl_z_max))
        self.cyl_qveladr = int(self.mj_model.jnt_dofadr[self.cyl_joint_id])
        self.cyl_motion_enabled = True
        self.cyl_grasped = False
        self._grasp_hold = 0
        dt = float(self.mj_model.opt.timestep)
        self._grasp_hold_need = max(1, int(0.12 / dt))
        # ---- release-to-physics (no intermediate state) ----
        self.release_dist = 0.06          # 距离阈值：末端(2f85_base)到box中心 (m)，可调 0.05~0.08
        self.grip_close_cmd_th = 200.0    # 夹爪“闭合指令”阈值（0~255），可调 180~230
        self._release_hold = 0
        self._release_hold_need = max(1, int(0.05 / dt))  # 满足条件持续 50ms 才触发（可调 0.03~0.08）

        self.cyl_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Box")
        self.gripper_base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "2f85_base")
        self.gripper_geom_ids = self._collect_subtree_geoms(self.gripper_base_body_id)

        t_wc = self.mj_data.cam_xpos[self.cam_id].copy()
        R_wc_mj = self.mj_data.cam_xmat[self.cam_id].reshape(3, 3).copy()

        # MuJoCo camera: X right, Y up, look=-Z  ->  CV camera: X right, Y down, Z forward
        R_mj_to_cv = np.diag([1.0, -1.0, -1.0])

        R_wc_cv = R_wc_mj @ R_mj_to_cv
        self.T_wc = sm.SE3.Rt(R_wc_cv, t_wc)

        R = self.T_wc.R
        print("[T_wc] [should be +1] det(R)=", np.linalg.det(R), "[should be small] orth_err=", np.linalg.norm(R.T @ R - np.eye(3)))
        print("[T_wc] t=", self.T_wc.t)
        print("[T_wc] cam forward(world)=", (self.T_wc.R @ np.array([0,0,1.0])).ravel())  # +Z (CV forward) in world

        return observation

    def _get_grip_cmd(self, action) -> float:
        """Return gripper command (0~255). Prefer the action passed in."""
        if action is not None and len(action) >= 7:
            return float(action[6])
        # fallback: read from mujoco control
        if self.mj_model is not None and self.mj_model.nu >= 7:
            return float(self.mj_data.ctrl[6])
        return 0.0

    def _ee_obj_dist(self) -> float:
        """Distance between gripper base body and object geom center."""
        obj = self.mj_data.geom_xpos[self.cyl_geom_id]  # Object position (3,)
        ee = self.mj_data.geom_xpos[self.gripper_base_body_id]  # Gripper base position (3,)
        return float(np.linalg.norm(obj - ee))

    def _collect_subtree_geoms(self, root_body_id: int) -> set[int]:
        """Collect all geom ids whose body is in the subtree of root_body_id."""
        m = self.mj_model

        def is_descendant(bid: int) -> bool:
            # walk parents until world(0) or hit root
            while bid > 0:
                if bid == root_body_id:
                    return True
                bid = int(m.body_parentid[bid])
            return False

        geoms = set()
        for gid in range(m.ngeom):
            bid = int(m.geom_bodyid[gid])
            if is_descendant(bid):
                geoms.add(gid)
        return geoms

    def _is_object_grasped_by_contacts(self) -> bool:
        """Object geom contacts with >=2 distinct gripper geoms in same step."""
        if self.cyl_geom_id < 0 or self.gripper_base_body_id < 0:
            return False
        hit = set()
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.cyl_geom_id and g2 in self.gripper_geom_ids:
                hit.add(g2)
            elif g2 == self.cyl_geom_id and g1 in self.gripper_geom_ids:
                hit.add(g1)
            if len(hit) >= 2:
                return True
        return False

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()
        if self.mj_seg_renderer is not None:
            self.mj_seg_renderer.close()

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        dt = float(self.mj_model.opt.timestep)
        adr = self.cyl_qposadr

        # Only move the object if not grasped
        if self.cyl_motion_enabled:
            y = float(self.mj_data.qpos[adr + 1])
            z = float(self.mj_data.qpos[adr + 2])

            y_next = y + self.cyl_dir_y * self.cyl_speed_y * dt
            z_next = z + self.cyl_dir_z * self.cyl_speed_z * dt

            # y bounce
            if y_next <= self.cyl_y_min:
                y_next = self.cyl_y_min
                self.cyl_dir_y = 1.0
            elif y_next >= self.cyl_y_max:
                y_next = self.cyl_y_max
                self.cyl_dir_y = -1.0

            # z bounce
            if z_next <= self.cyl_z_min:
                z_next = self.cyl_z_min
                self.cyl_dir_z = 1.0
            elif z_next >= self.cyl_z_max:
                z_next = self.cyl_z_max
                self.cyl_dir_z = -1.0

            # keep x fixed, update y/z, keep quat fixed
            self.mj_data.qpos[adr + 0] = self.cyl_x0
            self.mj_data.qpos[adr + 1] = y_next
            self.mj_data.qpos[adr + 2] = z_next
            self.mj_data.qpos[adr + 3: adr + 7] = self.cyl_quat0

            # clear free-joint velocity to avoid drift
            vadr = self.cyl_qveladr
            self.mj_data.qvel[vadr:vadr + 6] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)

        # grasp detect: contact(Cylinder geom) with gripper subtree geoms
        # ---- release to physics when "close cmd" AND "near" ----
        if not self.cyl_grasped:
            grip_cmd = self._get_grip_cmd(action)
            near = (self._ee_obj_dist() <= self.release_dist)
            closing = (grip_cmd >= self.grip_close_cmd_th)

            if self.cyl_motion_enabled and closing and near:
                self._release_hold += 1
                if self._release_hold >= self._release_hold_need:
                    self.cyl_grasped = True
                    self.cyl_motion_enabled = False
                    print(f"[YZEnv] Box released to physics (closing+near). grip_cmd={grip_cmd:.1f}")
            else:
                self._release_hold = 0

        self.mj_viewer.sync()

    def hold(self):
        if self.mj_viewer is None:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        try:
            dt = float(self.mj_model.opt.timestep)
            while self.mj_viewer.is_running():
                self.step()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass

    def render(self):
        self.mj_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_seg_renderer.update_scene(self.mj_data, self.cam_id)

        return {
            "img": self.mj_renderer.render(),
            "depth": self.mj_depth_renderer.render(),
            "seg": self.mj_seg_renderer.render(),
            "box_gid": int(self.box_geom_id),
            "K": self.camera_matrix.copy(),
        }

if __name__ == '__main__':
    env = YZEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
