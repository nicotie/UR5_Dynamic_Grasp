from typing import List
import copy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from arm.geometry import Geometry3D, Capsule
from arm.utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class UR5e(Robot):

    def __init__(self) -> None:
        super().__init__()
        d1 = 0.163
        d4 = 0.134
        d5 = 0.1
        d6 = 0.1

        a3 = 0.425
        a4 = 0.392

        self._dof = 6
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.joint_limits = np.array([
            [-2.8,  2.8],
            [-2.8,  2.8],
            [-2.8,  2.8],
            [-2.8,  2.8],
            [-2.8,  2.8],
            [-2.8,  2.8],
        ], dtype=float)
        
        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        d_array = [d1, 0.0, 0.0, d4, d5, d6]
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
        sigma_array = [0, 0, 0, 0, 0, 0]

        m1 = 3.7
        r1 = np.array([0.0, 0.0, 0.0])
        I1 = np.diag([0.0102675, 0.0102675, 0.00666])
        Jm1 = 0.1

        m2 = 8.393
        r2 = np.array([0.2125, 0.0, 0.138])
        I2 = np.diag([0.0151074, 0.133886, 0.133886])
        Jm2 = 0.1

        m3 = 2.275
        r3 = np.array([0.196, 0.0, 0.007])
        I3 = np.diag([0.004095, 0.0311796, 0.0311796])
        Jm3 = 0.1

        m4 = 1.219
        r4 = np.array([0.0, 0.0, 0.0])
        I4 = np.diag([0.0025599, 0.0021942, 0.0025599])
        Jm4 = 0.1

        m5 = 1.219
        r5 = np.array([0.0, 0.0, 0.0])
        I5 = np.diag([0.0025599, 0.0025599, 0.0021942])
        Jm5 = 0.1

        m6 = 0.1889
        r6 = np.array([0.0, 0.0, -0.0228317])
        I6 = np.diag([9.90863e-05, 9.90863e-05, 0.000132134])
        Jm6 = 0.1

        ms = [m1, m2, m3, m4, m5, m6]
        rs = [r1, r2, r3, r4, r5, r6]
        Is = [I1, I2, I3, I4, I5, I6]
        Jms = [Jm1, Jm2, Jm3, Jm4, Jm5, Jm6]

        links = []
        for i in range(6):
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True,
                                    m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        self.robot = rtb.DHRobot(links)

        self.alpha_array = alpha_array
        self.a_array = a_array
        self.d_array = d_array
        self.theta_array = theta_array
        self.sigma_array = sigma_array

        T = SE3()
        for i in range(self.dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            self._Ms.append(Ti.A)
            T: SE3 = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = Is[i]
            Gm[3:, 3:] = ms[i] * np.eye(3)
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))
            self._Gs.append(AdT.T @ Gm @ AdT)
            self._Jms.append(Jms[i])

        self._Ms.append(np.eye(4))

        self.robot_config = RobotConfig()
        self.robot_config.wrist = -1

    def _ensure_kin_params(self):
        d1, d4, d5, d6 = 0.163, 0.134, 0.1, 0.1
        a3, a4 = 0.425, 0.392
        if not hasattr(self, "alpha_array"):
            self.alpha_array = [0.0, -np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2]
        if not hasattr(self, "a_array"):
            self.a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        if not hasattr(self, "d_array"):
            self.d_array = [d1, 0.0, 0.0, d4, d5, d6]
        if not hasattr(self, "theta_array"):
            self.theta_array = [0.0, -np.pi/2, 0.0, np.pi/2, 0.0, 0.0]
        if not hasattr(self, "sigma_array"):
            self.sigma_array = [0, 0, 0, 0, 0, 0]
        if not hasattr(self, "q0"):
            self.q0 = [0.0] * 6
        if not hasattr(self, "robot_config") or self.robot_config is None:
            self.robot_config = RobotConfig()
            self.robot_config.wrist = -1
            self.robot_config.inline = 1
            self.robot_config.overhead = 1
        if not hasattr(self, "_dof"):
            self._dof = 6

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_kin_params()

    def __getstate__(self):
        return dict(self.__dict__)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))
        new_obj._ensure_kin_params()
        return new_obj

    def ikine(self, Tep: SE3) -> np.ndarray:
        self._ensure_kin_params()
        T06: SE3 = self._base.inv() * Tep * self._tool.inv()

        wpc = self.cal_wpc(T06)

        thetas = [0.0 for _ in range(self.dof)]

        # solve theta1
        theta1_condition = np.power(wpc[1], 2) + np.power(wpc[0], 2) - np.power(self.d_array[3], 2)
        if theta1_condition < 0:
            return np.array([])
        if MathUtils.near_zero(np.linalg.norm([wpc[1], wpc[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]  # overhead singularity
        else:
            # if self.robot_config.overhead == 0:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], np.sqrt(theta1_condition))
            # elif self.robot_config.overhead == 1:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], -np.sqrt(theta1_condition))
            # else:
            #     return np.array([])
            thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3],
                                                                self.robot_config.overhead * np.sqrt(theta1_condition))

        # solve theta5
        theta5_condition = -T06.a[0] * np.sin(thetas[0]) + T06.a[1] * np.cos(thetas[0])
        if np.abs(theta5_condition) > 1:
            return np.array([])
        # if self.robot_config.wrist == 0:
        #     thetas[4] = np.arccos(theta5_condition)
        # elif self.robot_config.wrist == 1:
        #     thetas[4] = -np.arccos(theta5_condition)
        # else:
        #     return np.array([])
        thetas[4] = self.robot_config.wrist * np.arccos(theta5_condition)

        # solve theta6
        m1 = - T06.n[0] * np.sin(thetas[0]) + T06.n[1] * np.cos(thetas[0])
        n1 = - T06.o[0] * np.sin(thetas[0]) + T06.o[1] * np.cos(thetas[0])
        if MathUtils.near_zero(np.sin(thetas[4])):
            thetas[5] = self.q0[5] - self.theta_array[5]
        else:
            thetas[5] = np.arctan2(-n1 / np.sin(thetas[4]), m1 / np.sin(thetas[4]))

        # solve theta3
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0],
                                     0.0)
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], thetas[4],
                                     self.sigma_array[4],
                                     0.0)
        T56 = get_transformation_mdh(self.alpha_array[5], self.a_array[5], self.d_array[5], thetas[5],
                                     self.sigma_array[5],
                                     0.0)
        T46 = T45 * T56
        T14 = T01.inv() * T06 * T46.inv()
        x = T14.t[0]
        y = T14.t[2]

        theta3_condition = (np.power(x, 2) + np.power(y, 2) - np.power(self.a_array[2], 2)
                            - np.power(self.a_array[3], 2)) / (2 * self.a_array[2] * self.a_array[3])
        if np.abs(theta3_condition) > 1.0:
            return np.array([])
        # if self.robot_config.inline == 0:
        #     thetas[2] = np.arccos(theta3_condition)
        # elif self.robot_config.inline == 1:
        #     thetas[2] = - np.arccos(theta3_condition)
        # else:
        #     return np.array([])
        thetas[2] = self.robot_config.inline * np.arccos(theta3_condition)

        # solve theta2
        M = np.array([
            [self.a_array[3] * np.cos(thetas[2]) + self.a_array[2], -self.a_array[3] * np.sin(thetas[2])],
            [self.a_array[3] * np.sin(thetas[2]), self.a_array[3] * np.cos(thetas[2]) + self.a_array[2]]
        ])
        XY = np.array([
            [x],
            [-y]
        ])
        CS = np.linalg.inv(M) @ XY
        thetas[1] = np.arctan2(CS[1, 0], CS[0, 0])

        # solve theta4
        thetas[3] = np.arctan2(-T14.o[0], T14.n[0]) - thetas[1] - thetas[2]

        qs = np.array([wrap(thetas[i] - self.theta_array[i])[0] for i in range(self.dof)])

        q0_s = list(map(wrap, self.q0))

        for i in range(self.dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def set_robot_config(self, q):
        self._ensure_kin_params()
        T = self.fkine(q)
        wpc = self.cal_wpc(T)
        thetas = [q[i] + self.theta_array[i] for i in range(self.dof)]

        if wrap(np.arctan2(wpc[1], wpc[0]) - wrap(thetas[0])[0])[0] <= np.pi / 2:
            self.robot_config.overhead = 1
        else:
            self.robot_config.overhead = -1

        # inline
        if wrap(thetas[2])[0] >= 0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        # wrist
        if wrap(thetas[4])[0] >= 0:
            self.robot_config.wrist = 1
        else:
            self.robot_config.wrist = -1

    def cal_wpc(self, T: SE3) -> np.ndarray:
        t = T.t - self.d_array[5] * T.a
        t[2] -= self.d_array[0]
        return t

    def get_geometries(self) -> List[Geometry3D]:
        self._ensure_kin_params()
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                           self.sigma_array[i], self.q0[i])
            Ts.append(T)

        T1 = Ts[0] * SE3.Trans(0, 0, -0.04)
        geometry1 = Capsule(T1, 0.06, 0.12)

        T2 = Ts[1] * SE3.Trans(0.2, 0, 0.138) * SE3.Ry(-np.pi / 2)
        geometry2 = Capsule(T2, 0.05, 0.4)

        T3 = Ts[2] * SE3.Trans(0.2, 0, 0.007) * SE3.Ry(-np.pi / 2)
        geometry3 = Capsule(T3, 0.038, 0.38)

        T4 = Ts[3] * SE3.Trans(0.0, 0.0, -0.07)
        geometry4 = Capsule(T4, 0.04, 0.14)

        T5 = Ts[4] * SE3.Trans(0.0, 0.0, -0.06)
        geometry5 = Capsule(T5, 0.04, 0.12)

        T6 = Ts[5] * SE3.Trans(0.0, 0.0, -0.06)
        geometry6 = Capsule(T6, 0.04, 0.12)

        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]


if __name__ == '__main__':
    ur_robot = UR5e()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
