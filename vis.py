import numpy as np
import polyscope as ps


def vis_mesh_ps(verts, faces, name, kp_idx=None, enabled=True):
    # assumes ps is already inited
    ps_mesh = ps.register_surface_mesh(name, verts, faces)
    if kp_idx is not None:
        colors_vert = np.zeros((verts.shape[0], 3))
        colors_vert[kp_idx] = np.array([1, 0, 0])
        ps_mesh.add_color_quantity("keypoints", colors_vert, enabled=enabled)
    return ps_mesh
