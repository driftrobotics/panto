"""Pure kinematics checks — no I/O. Guards the FK/IK/Jacobian consistency the
haptics layer assumes. (Does NOT validate the angle conventions against the
q(DD) monster math doc — that's a hardware bring-up task.)"""

import numpy as np
import pytest

from panto.kinematics import (
    LinkGeometry,
    Unreachable,
    forward,
    inverse,
    jacobian,
    min_singular_value,
)

GEO = LinkGeometry.panto_v0()


@pytest.mark.parametrize("q1", np.linspace(-1.0, 1.0, 5))
@pytest.mark.parametrize("q2", np.linspace(0.3, 2.5, 5))
def test_ik_inverts_fk(q1, q2):
    q = np.array([q1, q2])
    recovered = inverse(forward(q, GEO), GEO, elbow="up")
    assert np.allclose(recovered, q, atol=1e-9)


def test_jacobian_matches_finite_difference():
    q = np.array([0.4, 1.1])
    eps = 1e-6
    numeric = np.column_stack(
        [
            (forward(q + [eps, 0], GEO) - forward(q - [eps, 0], GEO)) / (2 * eps),
            (forward(q + [0, eps], GEO) - forward(q - [0, eps], GEO)) / (2 * eps),
        ]
    )
    assert np.allclose(numeric, jacobian(q, GEO), atol=1e-6)


def test_elbow_branches_differ_but_reach_same_point():
    xy = np.array([0.15, 0.05])
    up = inverse(xy, GEO, elbow="up")
    down = inverse(xy, GEO, elbow="down")
    assert not np.allclose(up, down)
    assert np.allclose(forward(up, GEO), xy, atol=1e-9)
    assert np.allclose(forward(down, GEO), xy, atol=1e-9)


def test_unreachable_raises():
    with pytest.raises(Unreachable):
        inverse(np.array([0.30, 0.0]), GEO)   # > l1 + l2


def test_singularity_has_small_sigma_min():
    near_full_extension = np.array([0.2, 1e-3])
    assert min_singular_value(near_full_extension, GEO) < 0.02
