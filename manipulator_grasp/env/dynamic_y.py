"""
Env: box moving along y-axis
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

class Yenv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'test.xml')
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

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.enable_depth_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # observation = self._get_obs()
        observation = None

        # --- Cylinder motion config (AnyGrasp-like tracking friendly) ---
        self.cyl_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Box")
        self.cyl_qposadr = int(self.mj_model.jnt_qposadr[self.cyl_joint_id])  # free joint -> 7 qpos

        self.cyl_y_max = 1.0
        self.cyl_y_min = 0.2
        self.cyl_dir = -1.0  # start moving toward y_min

        # 推荐：按你的 grasp-update 频率选速度（默认假设 10Hz）
        self.grasp_update_hz = 10.0
        self.cyl_speed = 0.02 * self.grasp_update_hz   # 1次更新走 ~2cm -> v=0.2m/s(10Hz)
        # 想更像 demo “30帧一段”，可用：self.cyl_speed = (1.0-0.2) / (30.0/self.grasp_update_hz)

        # 固定 x/z/quat，只动 y（避免物体滚转影响 tracking）
        adr = self.cyl_qposadr
        self.cyl_x0 = float(self.mj_data.qpos[adr + 0])
        self.cyl_z0 = float(self.mj_data.qpos[adr + 2])
        self.cyl_quat0 = self.mj_data.qpos[adr + 3: adr + 7].copy()
        self.cyl_qveladr = int(self.mj_model.jnt_dofadr[self.cyl_joint_id])

        self.cyl_motion_enabled = True
        self.cyl_grasped = False
        self._grasp_hold = 0
        dt = float(self.mj_model.opt.timestep)
        self._grasp_hold_need = max(1, int(0.12 / dt))

        self.cyl_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Box")
        self.gripper_base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "2f85_base")
        self.gripper_geom_ids = self._collect_subtree_geoms(self.gripper_base_body_id)

        return observation

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
        """Heuristic: Cylinder geom contacts with >=2 distinct gripper geoms in same step."""
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

            # 提前退出：足够的“夹持接触”
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

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        dt = float(self.mj_model.opt.timestep)
        adr = self.cyl_qposadr

        # Only move the cylinder if not grasped
        if self.cyl_motion_enabled:
            y = float(self.mj_data.qpos[adr + 1])
            y_next = y + self.cyl_dir * self.cyl_speed * dt

            if y_next <= self.cyl_y_min:
                y_next = self.cyl_y_min
                self.cyl_dir = 1.0
            elif y_next >= self.cyl_y_max:
                y_next = self.cyl_y_max
                self.cyl_dir = -1.0

            # keep x/z/quat fixed (only y moves)
            self.mj_data.qpos[adr + 0] = self.cyl_x0
            self.mj_data.qpos[adr + 1] = y_next
            self.mj_data.qpos[adr + 2] = self.cyl_z0
            self.mj_data.qpos[adr + 3: adr + 7] = self.cyl_quat0

            # clear free-joint velocity to avoid drift
            vadr = self.cyl_qveladr
            self.mj_data.qvel[vadr:vadr + 6] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)

        # grasp detect: contact(Cylinder geom) with gripper subtree geoms
        if (not self.cyl_grasped) and self._is_object_grasped_by_contacts():
            self._grasp_hold += 1
            if self._grasp_hold >= self._grasp_hold_need:
                self.cyl_grasped = True
                self.cyl_motion_enabled = False
                print("[TestEnv] Cylinder grasped -> stop scripted motion")
        else:
            self._grasp_hold = 0

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
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }

if __name__ == '__main__':
    env = Yenv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()