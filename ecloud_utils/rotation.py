# rotation_utils.py
import numpy as np


def uniformRandomRotation():
    """
    替代 moleculekit.util.uniformRandomRotation()

    返回：
        M: np.ndarray, shape = (3, 3)
           均匀随机旋转矩阵，可直接用于坐标旋转
    """
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))

    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1

    M = q @ np.diag(signs)

    # 保证 det(M) = 1，避免产生镜像翻转
    if np.linalg.det(M) < 0:
        M[:, 0] *= -1

    return M

def rotate(coords, rotation_matrix, center=None):
    """
    对 N x 3 坐标进行旋转。

    coords: shape = [N, 3]
    rotation_matrix: shape = [3, 3]
    center: 旋转中心，shape = [3]
    """
    coords = np.asarray(coords, dtype=np.float32)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)

    if center is None:
        center = coords.mean(axis=0)

    center = np.asarray(center, dtype=np.float32)

    rotated_coords = (coords - center) @ rotation_matrix.T + center

    return rotated_coords