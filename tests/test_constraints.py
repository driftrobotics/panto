"""Pure constraint-projection checks — no I/O. Mirrors tests/test_kinematics.py
style: parametrized, np.allclose, analytic expectations.

Covers: anchor correctness on/off each constraint; bilateral pull from both
sides; the wall being free on the free side and stiff only into the surface;
wall anchor reprojection under tangential slide (no drag); workspace boundary
firing near the extension / fold singularities and staying quiet well inside;
grid snapping; degenerate poses exactly on the anchor / line / node.
"""

import numpy as np
import pytest

from panto.constraints import (
    Line,
    Point,
    Projection,
    SnapGrid,
    Wall,
    WorkspaceBoundary,
    is_active,
)
from panto.kinematics import LinkGeometry

GEO = LinkGeometry.panto_v0()


# --- Point ----------------------------------------------------------------
def test_point_anchor_is_the_target_and_penetration_is_distance():
    c = Point(at=np.array([0.1, 0.05]))
    p = c.project(np.array([0.1, 0.0]))
    assert np.allclose(p.anchor, [0.1, 0.05])
    assert p.penetration == pytest.approx(0.05)
    assert not p.unilateral
    assert np.allclose(p.normal, [0.0, 1.0])            # pose -> anchor


@pytest.mark.parametrize(
    "pose, expected_normal",
    [
        ([0.1, 0.0], [0.0, 1.0]),
        ([0.1, 0.1], [0.0, -1.0]),
        ([0.05, 0.05], [1.0, 0.0]),
        ([0.15, 0.05], [-1.0, 0.0]),
    ],
)
def test_point_pulls_from_every_side(pose, expected_normal):
    c = Point(at=np.array([0.1, 0.05]))
    p = c.project(np.array(pose))
    assert np.allclose(p.normal, expected_normal)
    # normal and (anchor - pose) are colinear and same sense
    assert np.allclose(p.normal * p.penetration, p.anchor - np.array(pose))


def test_point_pose_exactly_on_anchor_is_degenerate_but_safe():
    c = Point(at=np.array([0.1, 0.05]))
    p = c.project(np.array([0.1, 0.05]))
    assert p.penetration == pytest.approx(0.0)
    assert np.allclose(p.normal, [0.0, 0.0])
    assert is_active(p)                                  # bilateral, always on


# --- Line ---------------------------------------------------------------
def test_line_anchor_is_foot_of_perpendicular():
    c = Line(a=np.array([0.0, 0.0]), d=np.array([1.0, 0.0]))
    p = c.project(np.array([0.3, 0.2]))
    assert np.allclose(p.anchor, [0.3, 0.0])
    assert p.penetration == pytest.approx(0.2)
    assert np.allclose(p.normal, [0.0, -1.0])
    assert not p.unilateral


def test_line_pulls_from_both_sides():
    c = Line(a=np.array([0.0, 0.0]), d=np.array([1.0, 0.0]))
    above = c.project(np.array([0.3, 0.2]))
    below = c.project(np.array([0.3, -0.2]))
    assert np.allclose(above.anchor, below.anchor)
    assert np.allclose(above.normal, -below.normal)


def test_line_direction_need_not_be_unit():
    unit = Line(a=np.array([0.0, 0.0]), d=np.array([1.0, 1.0]))
    scaled = Line(a=np.array([0.0, 0.0]), d=np.array([5.0, 5.0]))
    pose = np.array([1.0, 0.0])
    pu, ps = unit.project(pose), scaled.project(pose)
    assert np.allclose(pu.anchor, [0.5, 0.5])
    assert np.allclose(pu.anchor, ps.anchor)
    assert pu.penetration == pytest.approx(np.sqrt(0.5))


def test_line_pose_exactly_on_line_is_degenerate():
    c = Line(a=np.array([0.1, 0.1]), d=np.array([0.0, 1.0]))
    p = c.project(np.array([0.1, 0.42]))
    assert np.allclose(p.anchor, [0.1, 0.42])
    assert p.penetration == pytest.approx(0.0)
    assert np.allclose(p.normal, [0.0, 0.0])


# --- Wall -------------------------------------------------------------------
def test_wall_is_free_on_the_free_side():
    w = Wall(a=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]))
    p = w.project(np.array([0.05, 0.02]))               # +y == free
    assert p.penetration < 0
    assert np.allclose(p.anchor, [0.05, 0.02])          # anchor collapses to pose
    assert p.unilateral
    assert not is_active(p)                             # runtime gates it off


def test_wall_is_stiff_into_the_surface():
    w = Wall(a=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]))
    p = w.project(np.array([0.05, -0.01]))              # past the surface
    assert p.penetration == pytest.approx(0.01)
    assert np.allclose(p.anchor, [0.05, 0.0])           # foot on the surface
    assert np.allclose(p.normal, [0.0, 1.0])            # push points to free side
    assert is_active(p)
    # restoring force direction is +normal, straight out, no tangential term
    force_dir = p.anchor - np.array([0.05, -0.01])
    assert np.allclose(force_dir, p.penetration * p.normal)


@pytest.mark.parametrize("x", [-0.08, -0.02, 0.0, 0.03, 0.11])
def test_wall_anchor_reprojects_under_tangential_slide_no_drag(x):
    """As the pose slides along the wall at fixed depth, the anchor tracks it
    tangentially — the pull stays purely normal, with no tangential drag."""
    w = Wall(a=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]))
    depth = 0.006
    p = w.project(np.array([x, -depth]))
    assert p.anchor[0] == pytest.approx(x)             # no tangential offset
    assert p.anchor[1] == pytest.approx(0.0)
    assert p.penetration == pytest.approx(depth)       # depth unchanged by slide


def test_wall_diagonal_projects_onto_plane():
    n = np.array([1.0, 1.0]) / np.sqrt(2)
    w = Wall(a=np.array([0.0, 0.0]), normal=n)
    # free side
    free = w.project(np.array([0.02, 0.02]))
    assert not is_active(free)
    # blocked side: symmetric point maps onto the plane through the origin
    blk = w.project(np.array([-0.01, -0.01]))
    assert np.allclose(blk.anchor, [0.0, 0.0], atol=1e-12)
    assert blk.penetration == pytest.approx(np.sqrt(2) * 0.01)


def test_wall_offset_surface_uses_point_a():
    w = Wall(a=np.array([0.0, 0.1]), normal=np.array([0.0, 1.0]))
    assert not is_active(w.project(np.array([0.0, 0.15])))
    p = w.project(np.array([0.0, 0.08]))
    assert p.penetration == pytest.approx(0.02)
    assert np.allclose(p.anchor, [0.0, 0.1])


# --- WorkspaceBoundary ----------------------------------------------------
def test_workspace_quiet_well_inside():
    wb = WorkspaceBoundary(geo=GEO, elbow="up", sigma_min_threshold=0.03)
    for pose in ([0.10, 0.05], [0.12, 0.03], [0.15, 0.05], [0.05, 0.12]):
        p = wb.project(np.array(pose))
        assert p.penetration == pytest.approx(0.0)
        assert np.allclose(p.anchor, pose)
        assert not is_active(p)


def test_workspace_fires_near_full_extension():
    wb = WorkspaceBoundary(geo=GEO, elbow="up", sigma_min_threshold=0.03)
    p = wb.project(np.array([0.249, 0.0]))              # ~ at max reach
    assert is_active(p)
    assert p.penetration > 0
    assert np.linalg.norm(p.anchor) < 0.249            # pulled inward
    assert wb._valid(p.anchor)
    assert np.allclose(p.normal, [-1.0, 0.0], atol=1e-6)  # push back inward


def test_workspace_fires_near_the_fold_singularity():
    wb = WorkspaceBoundary(geo=GEO, elbow="up", sigma_min_threshold=0.03)
    p = wb.project(np.array([0.005, 0.0]))             # folded, near base
    assert is_active(p)
    assert np.linalg.norm(p.anchor) > 0.005            # pushed outward
    assert wb._valid(p.anchor)


def test_workspace_handles_unreachable_target():
    wb = WorkspaceBoundary(geo=GEO, elbow="up", sigma_min_threshold=0.03)
    p = wb.project(np.array([0.4, 0.0]))               # past the annulus
    assert is_active(p)
    assert p.penetration > 0.1
    assert wb._valid(p.anchor)
    assert np.allclose(p.normal, [-1.0, 0.0], atol=1e-6)


def test_workspace_threshold_widens_the_valid_region():
    strict = WorkspaceBoundary(geo=GEO, sigma_min_threshold=0.05)
    loose = WorkspaceBoundary(geo=GEO, sigma_min_threshold=0.01)
    pose = np.array([0.235, 0.0])
    assert is_active(strict.project(pose))
    assert not is_active(loose.project(pose))


def test_workspace_from_config_reads_the_right_fields():
    class FakeConfig:
        geo = GEO
        elbow = "up"
        sigma_min_threshold = 0.03
        workspace_polygon = None

    wb = WorkspaceBoundary.from_config(FakeConfig())
    assert wb.geo is GEO
    assert wb.sigma_min_threshold == 0.03
    assert not is_active(wb.project(np.array([0.12, 0.03])))


def test_workspace_polygon_clips_the_region():
    poly = np.array([[0.0, -0.08], [0.18, -0.08], [0.18, 0.08], [0.0, 0.08]])
    wb = WorkspaceBoundary(
        geo=GEO, sigma_min_threshold=0.03, workspace_polygon=poly
    )
    # inside polygon and well-conditioned -> quiet
    assert not is_active(wb.project(np.array([0.10, 0.02])))
    # conditioned but outside the polygon (x < 0) -> pushed back in
    p = wb.project(np.array([-0.05, 0.02]))
    assert is_active(p)
    assert p.anchor[0] >= -1e-6
    assert wb._valid(p.anchor)


# --- SnapGrid -----------------------------------------------------------
@pytest.mark.parametrize(
    "pose, node",
    [
        ([0.023, -0.017], [0.02, -0.02]),
        ([0.0, 0.0], [0.0, 0.0]),
        ([0.049, 0.031], [0.05, 0.03]),
        ([-0.006, 0.014], [-0.01, 0.01]),
    ],
)
def test_snapgrid_snaps_to_nearest_node(pose, node):
    g = SnapGrid(pitch=0.01, origin=np.array([0.0, 0.0]))
    p = g.project(np.array(pose))
    assert np.allclose(p.anchor, node)
    assert p.penetration == pytest.approx(np.linalg.norm(np.array(node) - pose))
    assert not p.unilateral


def test_snapgrid_respects_origin_offset():
    g = SnapGrid(pitch=0.02, origin=np.array([0.005, 0.005]))
    p = g.project(np.array([0.028, 0.006]))
    assert np.allclose(p.anchor, [0.025, 0.005])


def test_snapgrid_pose_exactly_on_node_is_degenerate():
    g = SnapGrid(pitch=0.01, origin=np.array([0.0, 0.0]))
    p = g.project(np.array([0.03, -0.02]))
    assert np.allclose(p.anchor, [0.03, -0.02])
    assert p.penetration == pytest.approx(0.0)
    assert np.allclose(p.normal, [0.0, 0.0])


# --- protocol / helper ---------------------------------------------------
def test_every_primitive_returns_a_projection_with_unit_normal():
    cases = [
        Point(at=np.array([0.1, 0.0])).project(np.array([0.12, 0.02])),
        Line(a=np.zeros(2), d=np.array([1.0, 2.0])).project(np.array([0.1, 0.0])),
        Wall(a=np.zeros(2), normal=np.array([0.0, 1.0])).project(np.array([0.0, -0.01])),
        SnapGrid(pitch=0.01, origin=np.zeros(2)).project(np.array([0.033, 0.011])),
    ]
    for p in cases:
        assert isinstance(p, Projection)
        assert np.linalg.norm(p.normal) == pytest.approx(1.0)


def test_is_active_gates_only_unilateral_terms():
    bilateral = Point(at=np.zeros(2)).project(np.array([0.1, 0.0]))
    free_wall = Wall(a=np.zeros(2), normal=np.array([0.0, 1.0])).project(
        np.array([0.0, 0.05])
    )
    stiff_wall = Wall(a=np.zeros(2), normal=np.array([0.0, 1.0])).project(
        np.array([0.0, -0.05])
    )
    assert is_active(bilateral)
    assert not is_active(free_wall)
    assert is_active(stiff_wall)


# ------------------------------------------------------------ finite segments

def test_finite_wall_does_not_push_past_its_ends():
    wall = Wall(a=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), b=np.array([0.1, 0.0]))
    inside = wall.project(np.array([0.05, -0.01]))
    assert inside.penetration == pytest.approx(0.01)
    assert np.allclose(inside.anchor, [0.05, 0.0])
    beyond = wall.project(np.array([0.15, -0.01]))
    assert beyond.penetration == 0.0 and not is_active(beyond)
    before = wall.project(np.array([-0.02, -0.01]))
    assert before.penetration == 0.0
    free = wall.project(np.array([0.05, 0.01]))
    assert free.penetration < 0.0 and not is_active(free)


def test_finite_line_clamps_to_its_endpoints():
    seg = Line(a=np.array([0.0, 0.0]), d=np.array([1.0, 0.0]), b=np.array([0.1, 0.0]))
    assert np.allclose(seg.project(np.array([0.05, 0.02])).anchor, [0.05, 0.0])
    assert np.allclose(seg.project(np.array([0.30, 0.02])).anchor, [0.10, 0.0])
    assert np.allclose(seg.project(np.array([-0.30, 0.02])).anchor, [0.0, 0.0])
    # direction pointing away from b still clamps to the same segment
    seg2 = Line(a=np.array([0.0, 0.0]), d=np.array([-1.0, 0.0]), b=np.array([0.1, 0.0]))
    assert np.allclose(seg2.project(np.array([0.30, 0.02])).anchor, [0.10, 0.0])
