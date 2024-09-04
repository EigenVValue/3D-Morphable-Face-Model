import os
import numpy as np
import trimesh
from scipy.spatial import KDTree
import gdown

if not os.path.exists('./data'):
    gdown.download_folder("https://drive.google.com/drive/folders/1ctaUjFMt3QpTpfRrHaUPQCF-mLowlslO", quiet=False)

import data.keypoints_idx as keypoints_idx
from BFM import BFM_layer

def load_scan(path="data/scan.ply"):
    model = trimesh.load(path, process=False)
    scan_verts, scan_faces = np.array(model.vertices), np.array(model.faces)

    return (
        scan_verts,
        scan_faces,
        np.array(model.vertex_normals),
        keypoints_idx.scan_kps,
    )


def load_bfm_from_dict(bfm_params):
    bfm_model = BFM_layer(
        bfm_params["mean_shape"],
        bfm_params["faces"],
        bfm_params["kp_idx"],
        bfm_params["id_base"],
        bfm_params["exp_base"],
    )
    return bfm_model


def load_bfm(model_path="data/BFM_params.npz"):
    bfm_params = np.load(model_path)
    return load_bfm_from_dict(bfm_params)


def hausdorff(verts1, verts2):
    # verts1 <-> verts2 distance
    verts1_tree = KDTree(verts1)
    verts2_tree = KDTree(verts2)

    dists1, _ = verts1_tree.query(verts2, k=1)
    dists2, _ = verts2_tree.query(verts1, k=1)
    return max(np.max(dists1), np.max(dists2))
