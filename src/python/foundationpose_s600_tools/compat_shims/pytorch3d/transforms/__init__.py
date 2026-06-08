import torch


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1] + (3, 3))


def so3_exp_map(log_rot, eps=1e-6):
    theta = torch.linalg.norm(log_rot, dim=-1, keepdim=True).clamp_min(eps)
    axis = log_rot / theta
    K = _skew(axis)
    eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device).expand(K.shape)
    theta_m = theta[..., None]
    return eye + torch.sin(theta_m) * K + (1 - torch.cos(theta_m)) * (K @ K)


def so3_log_map(R, eps=1e-6):
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos = cos.clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (theta / (2 * torch.sin(theta).clamp_min(eps)))[..., None]


def se3_exp_map(x):
    # Minimal approximation sufficient for imports/smoke; FoundationPose's pose
    # decode path is normally handled by upstream CUDA host logic.
    R = so3_exp_map(x[..., 3:6])
    T = torch.eye(4, dtype=x.dtype, device=x.device).repeat(x.shape[:-1] + (1, 1))
    T[..., :3, :3] = R
    T[..., :3, 3] = x[..., :3]
    return T


def se3_log_map(T):
    return torch.cat([T[..., :3, 3], so3_log_map(T[..., :3, :3])], dim=-1)


def matrix_to_axis_angle(R):
    return so3_log_map(R)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def euler_angles_to_matrix(euler, convention='XYZ'):
    raise NotImplementedError('pytorch3d shim: euler_angles_to_matrix is not implemented')


def matrix_to_euler_angles(matrix, convention='XYZ'):
    raise NotImplementedError('pytorch3d shim: matrix_to_euler_angles is not implemented')
