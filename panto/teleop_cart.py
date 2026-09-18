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
_DQ_MAX = np.radians(45.0)   # a nudge never re-poses a joint by more than this
_TOL = 0.005       # m, IK residual accepted as converged


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
        """Integrate held arrow keys (radial, z in -1..1) into the offset. The
        previous (known-reachable) offset is kept so ``solve`` can back out."""
        self._prev_offset = self.offset.copy()
        self.offset += self.speed * np.asarray(axes, float) * max(dt, 0.0)
        norm = float(np.linalg.norm(self.offset))
        if norm > _R_MAX:
            self.offset *= _R_MAX / norm
        return self.offset.copy()

    _prev_offset = None
    rejected = 0        # ticks whose nudge step was backed out (unreachable / diverging)

    def clear(self) -> None:
        self.offset[:] = 0.0
        self.dq[:] = 0.0

    def solve(self, q_full: np.ndarray, q_map: np.ndarray, iters: int = 8) -> np.ndarray:
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
        dq = self.dq.copy()
        for _ in range(iters):
            q[self.idx] = q_map + dq
            err = goal - self.ee_rz(q)
            if np.linalg.norm(err) < 1e-4:
                break
            J = self._jac_rz(q)
            if abs(np.linalg.det(J)) < 1e-6:
                break
            dq += np.linalg.solve(J, err)
            if np.any(np.abs(dq) > _DQ_MAX):
                break
        q[self.idx] = q_map + dq
        ok = np.linalg.norm(goal - self.ee_rz(q)) < _TOL
        # 2026-09-18 00:28 incident: an unreachable 21.7 cm nudge made Newton diverge, the
        # garbage dq re-posed the YAM by 70 deg and dragged panto to its joint limits.
        # Never emit a non-converged or oversized dq: back the nudge out to the last
        # reachable offset and keep the last good dq.
        if not ok or np.any(np.abs(dq) > _DQ_MAX):
            self.rejected += 1
            if self._prev_offset is not None:
                self.offset = self._prev_offset.copy()
            return q_map + self.dq
        self.dq = dq
        return q_map + self.dq
