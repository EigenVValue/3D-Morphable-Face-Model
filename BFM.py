import sys
import numpy as np


class BFM_layer:
    def __init__(self, bfm_verts, bfm_faces, bfm_kp_idx, id_base, exp_base):
        super(BFM_layer, self).__init__()
        self.N_verts = bfm_verts.shape[0]
        self.kp_idx = np.array(bfm_kp_idx)

        self.mean_shape = bfm_verts
        self.faces = bfm_faces
        self.id_base = id_base # Shape: [3N, 80]
        self.exp_base = exp_base # Shape: [3N, 64]

        self.rot_vec = np.zeros(3)
        self.t = np.zeros(3)

        self.id_coeff = np.zeros(self.id_base.shape[1])
        self.exp_coeff = np.zeros(self.exp_base.shape[1])

    def rigid_inference(self):
        return self.mean_shape @ rodriguez(self.rot_vec) + self.t

    def inference(self):
        """
        Inference the 3D face shape from the BFM model.
        """
    ############ TODO: YOUR CODE HERE ############
        # 1. Get id component of non rigid transformation
        # 2. Get exp component non rigid transformation
        # 3. Find shape: mean_shape + id + exp
        # 4. Apply rigid transform: shape @ R + t

        # 1. Get id component of non rigid transformation
        id = np.dot(self.id_base, self.id_coeff)
        # 2. Get exp component non rigid transformation
        exp = np.dot(self.exp_base, self.exp_coeff)
        # 3. Find shape: mean_shape + id + exp
        predicted_shape = self.mean_shape + id.reshape(-1, 3) + exp.reshape(-1, 3)
        # 4. Apply rigid transform: shape @ R + t
        predicted_verts = predicted_shape @ rodriguez(self.rot_vec) + self.t

    ############ END OF YOUR CODE #################'
        return predicted_verts

    def get_J_rot(self, verts, special_mode=False):
        """
        Get the Jacobian matrix of the rotation vector.

        Args:
            verts (np.ndarray): the vertices of the face shape. Shape: [N, 3]
            special_mode (bool): turns on some modifications for non-rigid case. (you don't need it)
        """
    ############ TODO: YOUR CODE HERE ############
        #1. Get the skew matrices derivative: G_0, G_1, G_2
        G_0 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        G_1 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        G_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        #2. For each 3x3 block of J, compute the derivative per vertex
        J = np.zeros((3 * self.N_verts, 3))
        for i in range(self.N_verts):
            # Compute the Jacobian block for the i-th vertex
            J_block = np.zeros((3, 3))

            # Compute the derivative of the skew symmetric matrix for x, y, and z axes
            G0_v = np.dot(G_0, verts[i])
            G1_v = np.dot(G_1, verts[i])
            G2_v = np.dot(G_2, verts[i])

            # Populate the Jacobian block
            J_block[:, 0] = G0_v
            J_block[:, 1] = G1_v
            J_block[:, 2] = G2_v

            # Update the Jacobian matrix with the computed block
            J[i * 3 : i * 3 + 3, :] = J_block.T

    ############ END OF YOUR CODE #################'

        # Leave the below part as is. It's needed modification for non-rigid case.
        # If the code above is correct, then this modification will make it work for non-rigid case without any change.
        if special_mode:
            for n in range(self.N_verts):
                J[n * 3 : n * 3 + 3, :] = J[n * 3 : n * 3 + 3, :].T
        return J

    def get_J_t(self, special_mode=False):
        """
        Get the Jacobian matrix of the translation vector.

        special_mode (bool): turns on some modifications for non-rigid case. (you don't need it)
        """
    ############ TODO: YOUR CODE HERE ############
        # Compute the Jacobian matrix of the translation vector

        J = np.eye(3,3)
        J = np.tile(J, (self.N_verts,1))
    ############ END OF YOUR CODE #################'

        # Leave the below part as is. It's needed modification for non-rigid case.
        # If the code above is correct, then this modification will make it work for non-rigid case without any change.
        if special_mode:
            J = np.zeros((3 * self.N_verts, 3))
            for n in range(self.N_verts):
                J[n * 3 : n * 3 + 3, :] = rodriguez(self.rot_vec)
        return J

    def get_J_id(self):
        """
        Get the Jacobian matrix of the identity coefficients.
        """
    ############ TODO: YOUR CODE HERE ############
        N = self.id_base.shape[0] // 3
        L = self.id_base.shape[1]
        J = np.zeros((3 * N, L))
        for i in range(L):
            J[:, i] = self.id_base[:, i]
    ############ END OF YOUR CODE #################'
        return J

    def get_J_exp(self):
        """
        Get the Jacobian matrix of the expression coefficients.
        """
    ############ TODO: YOUR CODE HERE ############
        N = self.exp_base.shape[0] // 3
        L = self.exp_base.shape[1]
        J = np.zeros((3 * N, L))
        for i in range(L):
            J[:, i] = self.exp_base[:, i]
    ############ END OF YOUR CODE #################'
        return J

    def update_id_exp(self, delta):
        self.id_coeff += delta[: len(self.id_coeff)]
        self.exp_coeff += delta[len(self.id_coeff) :]

    def update_r_t(self, delta, special_mode=False):
        self.rot_vec += delta[:3]
        self.t += delta[3:6]
        


def skew(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])


def rodriguez(v):
    theta = np.linalg.norm(v)
    if theta < sys.float_info.epsilon:
        return np.eye(3, dtype=float)
    v = v / theta
    V = skew(v)
    I = np.eye(3)
    return I + np.sin(theta) * V + (1 - np.cos(theta)) * (V @ V)
