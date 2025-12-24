import torch
# =============================================================================
# 3DGS-style scaling / rotation parameterization (simplified utilities)
# =============================================================================


# Purpose: Convert a batch of unit quaternions (w,x,y,z) to rotation matrices.
def quat_to_rotmat(q):
    """
    Args:
        q: Tensor of shape [M, 4], expected to be normalized unit quaternions (w, x, y, z).
    Returns:
        Rotation matrices of shape [M, 3, 3].
    """
    assert q.dim() == 2 and q.size(-1) == 4
    B = q.size(0)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Standard quaternion-to-rotation-matrix formula as commonly used in 3DGS.
    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)

    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)

    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)

    return R


# Purpose: Convert a batch of rotation matrices to unit quaternions (w,x,y,z).
def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Args:
        R: Rotation matrices of shape (N, 3, 3).
    Returns:
        Unit quaternions of shape (N, 4) in (w, x, y, z) order.
    """
    assert R.ndim == 3 and R.shape[1] == 3 and R.shape[2] == 3
    N = R.shape[0]
    q = torch.empty((N, 4), dtype=R.dtype, device=R.device)

    m00 = R[:, 0, 0]
    m11 = R[:, 1, 1]
    m22 = R[:, 2, 2]

    trace = m00 + m11 + m22
    # Case 1: trace > 0
    mask_t = trace > 0
    if mask_t.any():
        t = torch.sqrt(trace[mask_t] + 1.0) * 2.0  # t = 4 * qw
        qw = 0.25 * t
        qx = (R[mask_t, 2, 1] - R[mask_t, 1, 2]) / t
        qy = (R[mask_t, 0, 2] - R[mask_t, 2, 0]) / t
        qz = (R[mask_t, 1, 0] - R[mask_t, 0, 1]) / t
        q[mask_t] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 2: trace <= 0, split by the largest diagonal element.
    mask_f = ~mask_t
    if mask_f.any():
        m00_f = m00[mask_f]
        m11_f = m11[mask_f]
        m22_f = m22[mask_f]

        # Determine which diagonal entry is largest.
        cond0 = (m00_f > m11_f) & (m00_f > m22_f)
        cond1 = (~cond0) & (m11_f > m22_f)
        cond2 = (~cond0) & (~cond1)

        # Case X: m00 is largest.
        if cond0.any():
            Rc = R[mask_f][cond0]
            t = torch.sqrt(1.0 + Rc[:, 0, 0] - Rc[:, 1, 1] - Rc[:, 2, 2]) * 2.0
            qx = 0.25 * t
            qw = (Rc[:, 2, 1] - Rc[:, 1, 2]) / t
            qy = (Rc[:, 0, 1] + Rc[:, 1, 0]) / t
            qz = (Rc[:, 0, 2] + Rc[:, 2, 0]) / t
            q[mask_f.nonzero(as_tuple=True)[0][cond0]] = torch.stack(
                [qw, qx, qy, qz], dim=-1
            )

        # Case Y: m11 is largest.
        if cond1.any():
            Rc = R[mask_f][cond1]
            t = torch.sqrt(1.0 + Rc[:, 1, 1] - Rc[:, 0, 0] - Rc[:, 2, 2]) * 2.0
            qy = 0.25 * t
            qw = (Rc[:, 0, 2] - Rc[:, 2, 0]) / t
            qx = (Rc[:, 0, 1] + Rc[:, 1, 0]) / t
            qz = (Rc[:, 1, 2] + Rc[:, 2, 1]) / t
            q[mask_f.nonzero(as_tuple=True)[0][cond1]] = torch.stack(
                [qw, qx, qy, qz], dim=-1
            )

        # Case Z: m22 is largest.
        if cond2.any():
            Rc = R[mask_f][cond2]
            t = torch.sqrt(1.0 + Rc[:, 2, 2] - Rc[:, 0, 0] - Rc[:, 1, 1]) * 2.0
            qz = 0.25 * t
            qw = (Rc[:, 1, 0] - Rc[:, 0, 1]) / t
            qx = (Rc[:, 0, 2] + Rc[:, 2, 0]) / t
            qy = (Rc[:, 1, 2] + Rc[:, 2, 1]) / t
            q[mask_f.nonzero(as_tuple=True)[0][cond2]] = torch.stack(
                [qw, qx, qy, qz], dim=-1
            )

    # Normalize to guard against numerical drift.
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    return q


# Purpose: Build the 3DGS-style lower-triangular (actually R*S) matrix from scaling and quaternion rotation.
def build_scaling_rotation_3d(scaling, quat):
    """
    This mirrors the idea in 3DGS GaussianModel.setup_functions():
      - scaling_activation = exp
      - rotation_activation = normalize (quaternion)
      - covariance_activation(L) = L @ L^T

    Args:
        scaling: Tensor of shape [M, 3], expected to be already exp-activated.
        quat: Tensor of shape [M, 4], expected to be normalized unit quaternions.
    Returns:
        L: Tensor of shape [M, 3, 3] where L = R @ S (S is diagonal scaling).
    """
    R = quat_to_rotmat(quat)  # [M,3,3]

    # Diagonal scaling matrix
    S = torch.zeros_like(R)
    S[:, 0, 0] = scaling[:, 0]
    S[:, 1, 1] = scaling[:, 1]
    S[:, 2, 2] = scaling[:, 2]

    L = torch.bmm(R, S)  # [M,3,3]
    return L
