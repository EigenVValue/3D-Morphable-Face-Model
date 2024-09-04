import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from utils import hausdorff, load_bfm, load_scan


def rigid_matching(X, Y, need_scale=True):
    """
    Finds a rigid transform R, t such that X @ R * s + t = Y

    Args:
        X (np.array [N, 3]): 3D coordinates of N points in space
        Y (np.array [N, 3]): 3D coordinates of N points in space

    Returns:
        R (np.array [3, 3]): Rotation
        t (np.array [3, ]): Translation
        s (np.array [1, ]): Scale

    """
    ############ TODO: YOUR CODE HERE ############
    # 1. Find the mean of X and Y

    # 2. Center X and Y
    # X_centered = ...
    # Y_centered = ...

    # 3. Find the scale factor:
    #    norm_X = np.sqrt(np.sum(X_centered**2))
    #    norm_Y = np.sqrt(np.sum(Y_centered**2))
    #  Rescale center:
    #   X_centered /= norm_X
    #   Y_centered /= norm_Y

    # 4. Find the rotation
    # 5. Find the translation

    # 6. If need_scale is True, then also find scale and fix translation:
    #   scale = norm_Y / (np.sum(s) * norm_X)
    #   t = y_mean - scale * x_mean.reshape(1, 3) @ R

    # 6. Return R, t, scale

    # 1. Find the mean of X and Y
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    # 2. Center X and Y
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # 3. Find the scale factor:
    scale = np.ones(1)
    norm_X = np.sqrt(np.sum(X_centered**2))
    norm_Y = np.sqrt(np.sum(Y_centered**2))
    #  Rescale center:
    X_centered /= norm_X
    Y_centered /= norm_Y

    # 4. Find the rotation
    H = X_centered.T @ Y_centered
    U, S, Vh = np.linalg.svd(H)
    R = U @ Vh

    # 5. Find the translation
    t = Y_mean - X_mean @ R

    # 6. If need_scale is True, then also find scale and fix translation:
    if need_scale:
        scale = norm_Y / (np.sum(S) * norm_X)
        t = Y_mean - scale * X_mean.reshape(1, 3) @ R

    #R, t, scale = np.eye(3), np.zeros(3), np.ones(1)  # < Dummy line for display
    ############ END OF YOUR CODE #################
    return R, t, scale


def rigid_ICP(
    bfm_model, scan_verts, scan_normals=None, loss_type="Point2Point", iters=15
):
    """
    Args:
        bfm_model (BFM): BFM model (BFM_layer class)
        scan_verts (np.array [N, 3]): 3D coordinates of N points in space
        scan_normals (np.array [N, 3]): 3D normals of the N points
        loss_type (str): Point2Point or Point2Plane
        iters (int): number of iterations

    Returns:
        bfm_model (BFM): BFM model with updated parameters

    """
    tree = KDTree(scan_verts)

    for idx in tqdm(range(iters)):
        bfm_verts = bfm_model.rigid_inference()
        _, idx_chmf = tree.query(bfm_verts, k=1)
    ############ TODO: YOUR CODE HERE ############
    # 1. Find the residuals (should have size [3N,], N = number of vertices)
    # residuals = ...

    # 2. Find the translation Jacobian (write implementation in BFM_layer.py)
    # J_t = bfm_model.get_J_t()

    # 3. Find the rotation Jacobian. Note that it depends on the current bfm_verts (write implementation in BFM_layer.py)
    # J_rot = bfm_model.get_J_rot(bfm_verts)

    # 4. Find full Jacobian
    # J = concatenated [J_rot, J_t]

    # 5. Find the update (delta). Solve linear least squares problem (don't forget about regularization)
    # delta = ...

    # 5. Update the bfm_model
    # bfm_model.update_r_t(delta)
        
    # 1. Find the residuals (should have size [3N,], N = number of vertices)
    scan_matched = scan_verts[idx_chmf]
    residuals = scan_matched - bfm_verts

    # 2. Find the translation Jacobian (write implementation in BFM_layer.py)
    J_t = bfm_model.get_J_t()

    # 3. Find the rotation Jacobian. Note that it depends on the current bfm_verts (write implementation in BFM_layer.py)
    J_rot = bfm_model.get_J_rot(bfm_verts)

    # 4. Find full Jacobian
    J = np.hstack((J_rot, J_t))

    # 5. Find the update (delta). Solve linear least squares problem (don't forget about regularization)
    delta = np.linalg.lstsq(J.T @ J + 0.1 * np.eye(J.shape[1]), J.T @ residuals.flatten(), rcond=None)[0]

    # 6. Update the bfm_model
    bfm_model.update_r_t(delta)

    ############ END OF YOUR CODE #################
    return bfm_model


def non_rigid_mathching(
    bfm_model, scan_verts, scan_normals=None, loss_type="Point2Point", iters=20
):
    """
    Args:
        bfm_model (BFM): BFM model (BFM_layer class)
        scan_verts (np.array [N, 3]): 3D coordinates of N points in space
        scan_normals (np.array [N, 3]): 3D normals of the N points
        loss_type (str): Point2Point or Point2Plane
        iters (int): number of iterations

    Returns:
        bfm_model (BFM): BFM model with updated parameters
    """
    scan_tree = KDTree(scan_verts)

    for i in tqdm(range(iters)):
        bfm_verts = bfm_model.inference()
        _, idx_chmf = scan_tree.query(bfm_verts, k=1)

    ############ TODO: YOUR CODE HERE ############
    # 1. Find the residuals (should have size [3N,], N = number of vertices)
    scan_matched = scan_verts[idx_chmf]
    residuals = scan_matched - bfm_verts

    # 2. Find the Jacobian of the identity non-rigid parameters (write implementation in BFM_layer.py)
    J_id = bfm_model.get_J_id()

    # 5. Find the Jacobian of the expression non-rigid parameters (write implementation in BFM_layer.py)
    J_exp = bfm_model.get_J_exp()

    # Use the Jacobians for rotations and translations as provided below (uncomment)
    # If their implementation for rigid ICP is correct, then here it'll work without any changes
    J_t = bfm_model.get_J_t(special_mode=True)
    J_rot = bfm_model.get_J_rot(np.tile(bfm_model.t, (len(scan_verts[idx_chmf]), 1)), special_mode=True) \
         - bfm_model.get_J_rot(scan_verts[idx_chmf], special_mode=True)

    # 6. Find full Jacobian
    J = np.hstack((J_rot, J_t, J_id, J_exp))

    # 7. Find the update (delta). Solve linear least squares problem (don't forget about regularization)
    delta = np.linalg.lstsq(J.T @ J + 0.1 * np.eye(J.shape[1]), J.T @ residuals.flatten(), rcond=None)[0]

    # 8. Update the bfm_model
    bfm_model.update_r_t(delta[:6])
    bfm_model.update_id_exp(delta[6:])

    ############ END OF YOUR CODE #################'
    return bfm_model


def main():
    import polyscope as ps
    import gdown
    from vis import vis_mesh_ps
    import os

    if not os.path.exists("./data"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1ctaUjFMt3QpTpfRrHaUPQCF-mLowlslO",
            quiet=False,
        )

    ######## Init ########
    scan_verts, scan_faces, scan_normals, scan_kp_idx = load_scan("data/scan.ply")
    bfm_model = load_bfm()

    # Vis
    ps.init()
    vis_mesh_ps(scan_verts, scan_faces, name="Scan", kp_idx=scan_kp_idx)
    vis_mesh_ps(
        bfm_model.mean_shape, bfm_model.faces, name="BFM", kp_idx=bfm_model.kp_idx
    )
    ps.show()

    ######## Rigid shape matching ########
    R, t, scale = rigid_matching(
        scan_verts[scan_kp_idx],
        bfm_model.mean_shape[bfm_model.kp_idx],
        need_scale=True,
    )
    scan_verts = scan_verts @ R * scale + t
    scan_normals = scan_normals @ R

    vis_mesh_ps(
        bfm_model.mean_shape, bfm_model.faces, name="BFM", kp_idx=bfm_model.kp_idx
    )
    vis_mesh_ps(scan_verts, scan_faces, name="Scan", kp_idx=scan_kp_idx, enabled=False)
    ps.show()

    ######## Rigid ICP ########
    rigid_ICP(bfm_model, scan_verts, scan_normals, loss_type="Point2Point")

    vis_mesh_ps(
        bfm_model.rigid_inference(),
        bfm_model.faces,
        name="BFM rigid ICP",
        kp_idx=bfm_model.kp_idx,
        enabled=False,
    )
    ps.show()

    ######## Non-Rigid Fitting ########
    non_rigid_mathching(
        bfm_model, scan_verts, scan_normals, iters=20, loss_type="Point2Point" #Point2Plane
    )

    vis_mesh_ps(
        bfm_model.inference(),
        bfm_model.faces,
        name="BFM non_ridgid",
        kp_idx=bfm_model.kp_idx,
        enabled=False,
    )
    ps.show()


if __name__ == "__main__":
    main()
