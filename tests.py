import os
import numpy as np

from fitting_3DMM import non_rigid_mathching, rigid_ICP, rigid_matching
from utils import hausdorff, load_bfm_from_dict, load_scan

bfm_params = np.load("data/BFM_params.npz")
check = np.load("data/check.npz")


def test_simple_rigid_matching():
    pts_1 = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        * 5
        + 3
    )

    pts_2 = np.array(
        [
            [0, 0, 0],
            [-1, 0, 0],
            [0, 0, -1],
        ]
    )

    R, t, scale = rigid_matching(
        pts_1,
        pts_2,
        need_scale=True,
    )
    pts_1 = pts_1 @ R * scale + t

    assert np.isclose(pts_1, pts_2).all()


def test_rigid_matching():
    bfm_model = load_bfm_from_dict(bfm_params)
    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan()

    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    assert hausdorff(scan_verts, bfm_model.mean_shape) < 0.15


def test_J_rigid():
    bfm_model = load_bfm_from_dict(bfm_params)

    assert bfm_model.get_J_t().shape == (check["J_t"].shape)
    assert np.isclose(bfm_model.get_J_t(), check["J_t"], atol=1e-4).all()

    inp = check["rot_inp"]
    assert bfm_model.get_J_rot(inp).shape == (check["J_rot"].shape)
    assert np.isclose(bfm_model.get_J_rot(inp), check["J_rot"], atol=1e-4).all()


def test_rigid_ICP():
    bfm_model = load_bfm_from_dict(bfm_params)
    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan()

    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    rigid_ICP(bfm_model, scan_verts, scan_normals)

    assert hausdorff(scan_verts, bfm_model.rigid_inference()) < 0.14


def test_J_bfm():
    bfm_model = load_bfm_from_dict(bfm_params)

    assert bfm_model.get_J_exp().shape == (check["J_exp"].shape)
    assert np.isclose(bfm_model.get_J_exp(), check["J_exp"], atol=1e-4).all()

    assert bfm_model.get_J_id().shape == (check["J_id"].shape)
    assert np.isclose(bfm_model.get_J_id(), check["J_id"], atol=1e-4).all()


def test_bfm_sample():
    bfm_model = load_bfm_from_dict(bfm_params)
    bfm_model.exp_coeff = check["exp_coeff"]
    bfm_model.id_coeff = check["id_coeff"]
    print(bfm_model.inference().shape)
    assert bfm_model.inference().shape == (check["sample"].shape)
    assert np.isclose(bfm_model.inference(), check["sample"], atol=1e-4).all()


def test_non_rigid_mathching():
    bfm_model = load_bfm_from_dict(bfm_params)
    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan()

    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    rigid_ICP(bfm_model, scan_verts, scan_normals)
    non_rigid_mathching(bfm_model, scan_verts, iters=20)
    assert hausdorff(scan_verts, bfm_model.inference()) < 0.12


def test_bouns_point2plane():
    bfm_model = load_bfm_from_dict(bfm_params)
    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan()

    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    rigid_ICP(
        bfm_model,
        scan_verts,
        scan_normals,
        loss_type="Point2Plane",
    )
    non_rigid_mathching(
        bfm_model, scan_verts, scan_normals, iters=20, loss_type="Point2Plane"
    )

    assert hausdorff(scan_verts, bfm_model.inference()) < 0.06


def test_make_screens():
    import polyscope as ps

    os.makedirs("./img_saves", exist_ok=True)

    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan()
    bfm_model = load_bfm_from_dict(bfm_params)

    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    rigid_ICP(bfm_model, scan_verts, scan_normals)
    non_rigid_mathching(bfm_model, scan_verts, iters=50)

    ps.init()
    pred_mesh = ps.register_surface_mesh("BFM", bfm_model.inference(), bfm_model.faces)
    pred_mesh.set_enabled(True)

    center = [0, 0, 0]
    dirs = [[0, 1, 4], [2, 0, 4], [-2, 0, 4], [0, 0, 4]]
    for idx, a_dir in enumerate(dirs):
        ps.look_at(a_dir, center)
        ps.screenshot(f"./img_saves/img_{idx}.jpg")
