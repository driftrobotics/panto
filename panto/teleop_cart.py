"""Cartesian end-effector offset for the YAM follower (scripts/teleop_yam.py).

panto drives two YAM joints joint-to-joint. The operator can *also* nudge the
end effector in the arm's vertical plane -- radial (in/out along the base yaw
direction) and z -- with the arrow keys. That nudge is kept as a Cartesian
offset and folded into the two mapped joints every tick by a 2-DOF Newton IK on
the YAM's own MuJoCo model, so the joint-space coupling (lead, force feedback)
is untouched: the leader spring targets ``q_measured - dq_offset``.

Only the two mapped joints move; the other joints stay at their hold values.
"""

from __future__ import annotations

import numpy as np

_R_MAX = 0.30      # m, |offset| clamp -- a nudge, not a second teleop channel


class PlanarOffset:
    """Radial/z EE offset -> joint offset on two joints, via MuJoCo FK + Jacobian."""

    def __init__(self, model, idx: tuple[int, int], *, ee_body: str = "gripper",
                 yaw_joint: int = 0, speed_m_s: float = 0.05) -> None:
        import mujoco
        self._mj = mujoco
        self._m = model
        self._d = mujoco.MjData(model)
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self._body < 0:
            raise ValueError(f"no body {ee_body!r} in model")
        self.idx = list(idx)
        self._yaw = yaw_joint
        self.speed = float(speed_m_s)
        self.offset = np.zeros(2)          # (radial, z) metres
        self.dq = np.zeros(2)              # joint offset on idx, warm-started
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---------------------------------------------------------------- kinematics

    def _fk(self, q: np.ndarray) -> np.ndarray:
        n = min(len(q), self._m.nq)
        self._d.qpos[:n] = q[:n]
        self._mj.mj_forward(self._m, self._d)
        return self._d.xpos[self._body].copy()

    def _basis(self, q: np.ndarray) -> np.ndarray:
        """Rows: radial unit vector (horizontal, along the base yaw) and z."""
        yaw = float(q[self._yaw])
        return np.array([[np.cos(yaw), np.sin(yaw), 0.0], [0.0, 0.0, 1.0]])

    def ee_rz(self, q: np.ndarray) -> np.ndarray:
        return self._basis(q) @ self._fk(q)

    def _jac_rz(self, q: np.ndarray) -> np.ndarray:
        self._fk(q)
        self._mj.mj_jacBody(self._m, self._d, self._jacp, self._jacr, self._body)
        return self._basis(q) @ self._jacp[:, self.idx]          # 2x2

    # ---------------------------------------------------------------- per tick

    def step(self, axes: tuple[float, float], dt: float) -> np.ndarray:
        """Integrate held arrow keys (radial, z in -1..1) into the offset."""
        self.offset += self.speed * np.asarray(axes, float) * max(dt, 0.0)
        norm = float(np.linalg.norm(self.offset))
        if norm > _R_MAX:
            self.offset *= _R_MAX / norm
        return self.offset.copy()

    def clear(self) -> None:
        self.offset[:] = 0.0
        self.dq[:] = 0.0

    def solve(self, q_full: np.ndarray, q_map: np.ndarray, iters: int = 3) -> np.ndarray:
        """Joint targets on ``idx`` such that EE(q_map + dq) = EE(q_map) + offset.

        ``q_full`` supplies the held joints; ``q_map`` is the two mapped joints'
        target before the offset. Returns q_map + dq. Newton on the 2x2 (r,z)
        Jacobian, warm-started from last tick; a singular Jacobian (arm fully
        stretched) freezes dq rather than jumping."""
        if not self.offset.any():
            self.dq[:] = 0.0
            return np.asarray(q_map, float).copy()
        q = np.asarray(q_full, float).copy()
        q[self.idx] = q_map
        goal = self.ee_rz(q) + self.offset
        for _ in range(iters):
            q[self.idx] = q_map + self.dq
            err = goal - self.ee_rz(q)
            if np.linalg.norm(err) < 1e-4:
                break
            J = self._jac_rz(q)
            if abs(np.linalg.det(J)) < 1e-6:
                break
            self.dq += np.linalg.solve(J, err)
        return q_map + self.dq
