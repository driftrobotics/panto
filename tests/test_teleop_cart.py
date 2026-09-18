"""PlanarOffset: arrow-key EE nudge folded into two YAM joints (needs mujoco + i2rt)."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
i2rt_utils = pytest.importorskip("i2rt.robots.utils")

from panto.teleop_cart import PlanarOffset  # noqa: E402


@pytest.fixture(scope="module")
def model():
    path = i2rt_utils.combine_arm_and_gripper_xml(i2rt_utils.ArmType.YAM, i2rt_utils.GripperType.LINEAR_4310)
    return mujoco.MjModel.from_xml_path(path)


def test_offset_moves_ee_by_the_requested_rz(model):
    po = PlanarOffset(model, (1, 2))
    q = np.radians([0.0, 65.0, 65.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    before = po.ee_rz(q)
    po.step((1.0, 0.0), 1.0)            # +5 cm radial
    po.step((0.0, -1.0), 0.6)           # -3 cm z
    q2 = q.copy()
    q2[[1, 2]] = po.solve(q, q[[1, 2]])
    after = po.ee_rz(q2)
    assert np.allclose(after - before, [0.05, -0.03], atol=1e-3)
    assert q2[0] == q[0] and np.all(q2[3:] == q[3:])          # only the two mapped joints moved


def test_offset_follows_yaw(model):
    po = PlanarOffset(model, (1, 2))
    q = np.radians([90.0, 65.0, 65.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p0 = po._fk(q)
    po.step((1.0, 0.0), 1.0)
    q2 = q.copy()
    q2[[1, 2]] = po.solve(q, q[[1, 2]])
    d = po._fk(q2) - p0
    assert abs(d[1] - 0.05) < 1e-3 and abs(d[0]) < 2e-3            # radial is +y at yaw 90


def test_zero_offset_is_identity_and_clamp(model):
    po = PlanarOffset(model, (1, 2))
    q = np.radians([0.0, 65.0, 65.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(po.solve(q, q[[1, 2]]), q[[1, 2]])
    for _ in range(200):
        po.step((1.0, 1.0), 0.1)
    assert np.linalg.norm(po.offset) <= 0.30 + 1e-9


def test_unreachable_nudge_is_backed_out_not_emitted(model):
    po = PlanarOffset(model, (1, 2), speed_m_s=1.0)
    q = np.radians([0.0, 65.0, 65.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    last_good = None
    for _ in range(40):                              # try to walk 40 cm straight up, 1 cm per tick
        po.step((0.0, 1.0), 0.01)
        out = po.solve(q, q[[1, 2]])
        assert np.all(np.abs(out - q[[1, 2]]) <= np.radians(45.0) + 1e-9)
        if po.rejected == 0:
            last_good = out.copy()
    assert po.rejected > 0                           # it did hit the reach limit
    assert last_good is not None and np.allclose(po.solve(q, q[[1, 2]]), last_good, atol=1e-6)  # frozen there
