"""
Env: three objects rotating
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

class RotateEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        # 添加目标物体ID
        self.box_id = None
        self.cylinder_id = None
        self.capsule_id = None
        self.box_joint_id = None
        self.cylinder_joint_id = None
        self.capsule_joint_id = None

        # 物体的旋转参数
        self.box_rotation_speed = 0.6
        self.cylinder_rotation_speed = 0.7
        self.capsule_rotation_speed = 0.7

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

        self.obj_grasped = False
        self.obj_motion_enabled = True
        self._grasp_hold = 0

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'rotate.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 获取目标物体的ID
        self.box_id = self.mj_model.body("Box").id
        self.cylinder_id = self.mj_model.body("Cylinder").id
        self.capsule_id = self.mj_model.body("Capsule").id

        # 获取关节ID
        self.box_joint_id = self.mj_model.joint("box_joint").id
        self.cylinder_joint_id = self.mj_model.joint("cylinder_joint").id
        self.capsule_joint_id = self.mj_model.joint("capsule_joint").id

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

        self.cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_seg_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.enable_depth_rendering()
        self.mj_seg_renderer.enable_segmentation_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.mj_seg_renderer.update_scene(self.mj_data, 0)

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        self._graspable_bodies = ["Box", "Cylinder", "Capsule"]
        self.initialize_object_rotations()
        self.step_num = 0
        # observation = self._get_obs()
        observation = None
        # 一次性收集夹具下所有 geom id
        self._gripper_geom_ids = self._collect_subtree_geoms(
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "2f85_base")
        )
        for pad_name in ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]:
            try:
                pad_gid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
                self._gripper_geom_ids.add(int(pad_gid))
            except Exception:
                pass
        return observation

    def _is_object_grasped(self) -> bool:
        """检测目标物体是否被夹具抓住（至少两个接触点）"""
        # geom id
        obj_geom_ids = [
            int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in self._graspable_bodies
        ]
        hit = set()
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in obj_geom_ids and g2 in self._gripper_geom_ids:
                hit.add((g1, g2))
            elif g2 in obj_geom_ids and g1 in self._gripper_geom_ids:
                hit.add((g2, g1))
            if len(hit) >= 1:
                return True
        return False
    
    def _collect_subtree_geoms(self, root_body_id: int) -> set[int]:
        m = self.mj_model
        def is_descendant(bid: int) -> bool:
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

    def initialize_object_rotations(self): #初始化物体的旋转速度
        box_qvel = self.mj_model.jnt_dofadr[self.box_joint_id]
        cylinder_qvel = self.mj_model.jnt_dofadr[self.cylinder_joint_id]
        capsule_qvel = self.mj_model.jnt_dofadr[self.capsule_joint_id]
        # Box: 绕Y轴旋转
        self.mj_data.qvel[box_qvel + 4] = self.box_rotation_speed
        # Cylinder: 绕Y轴旋转
        self.mj_data.qvel[cylinder_qvel + 4] = self.cylinder_rotation_speed
        # Capsule: 绕X轴旋转
        self.mj_data.qvel[capsule_qvel + 4] = self.capsule_rotation_speed
        mujoco.mj_forward(self.mj_model, self.mj_data)

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

        # 如果 motion enabled，则更新旋转；否则不动（自由物理）
        if self.obj_motion_enabled:
            box_qveladr = self.mj_model.jnt_dofadr[self.box_joint_id]
            cyl_qveladr = self.mj_model.jnt_dofadr[self.cylinder_joint_id]
            cap_qveladr = self.mj_model.jnt_dofadr[self.capsule_joint_id]

            self.mj_data.qvel[box_qveladr+4] = self.box_rotation_speed
            self.mj_data.qvel[cyl_qveladr+4] = self.cylinder_rotation_speed
            self.mj_data.qvel[cap_qveladr+3] = self.capsule_rotation_speed
            
            self.mj_data.qvel[box_qveladr+0:box_qveladr+3] = 0.0
            self.mj_data.qvel[cyl_qveladr+0:cyl_qveladr+3] = 0.0
            self.mj_data.qvel[cap_qveladr+0:cap_qveladr+3] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()

        if not self.obj_grasped:
            # 是否有至少两个接触点
            contact_ok = self._is_object_grasped()
            # 末端与物体中心距离
            ee_pos = self.mj_data.body("2f85_base").xpos
            # 选第一个检测到接触的抓取对象作为参考
            obj_pos = None
            for name in self._graspable_bodies:
                if self._is_object_grasped():  # 若当前抓取判定成功
                    obj_pos = self.mj_data.body(name).xpos
                    break
            dist = np.linalg.norm(ee_pos - obj_pos) if obj_pos is not None else 0.0
            # grip command (action 中 grip)
            grip_cmd = action[6] if action is not None else self.mj_data.ctrl[6]
            # 近距离 & grip 足够大
            near = (dist <= 0.06)
            closing = (grip_cmd >= 180.0)
            if contact_ok and closing:
                self._grasp_hold += 1
            else:
                self._grasp_hold = 0
            # hold 一定步数才认为是稳定抓住
            if self._grasp_hold >= int(0.1 / float(self.mj_model.opt.timestep)):
                self.obj_grasped = True
                self.obj_motion_enabled = False
                print("[RotateEnv] Object grasp detected -> stop scripted motion")

    def render(self):
        self.mj_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_seg_renderer.update_scene(self.mj_data, self.cam_id)

        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render(),
            'seg': self.mj_seg_renderer.render(),
            'box_gid':   int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Box")),
            'cylinder_gid': int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Cylinder")),
            'capsule_gid': int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Capsule")),
        }

if __name__ == '__main__':
    env = RotateEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
