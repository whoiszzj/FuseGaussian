import torch
import torch.nn.functional as F
import math
from utils.math import rotmat_to_quat


# Purpose: Unproject 2D Gaussian parameters into 3D Gaussians in world coordinates using depth and camera matrices.
def gaussian2d_to_3d(gaussian2d, depth_map, intrinsics, extrinsics, eps=1e-6):
    """
    Args:
        gaussian2d: Dict-like object containing 2D Gaussian parameters:
            - gaussian2d['_xyz']:      (N,2) normalized pixel coords in [-1, 1] (via tanh)
            - gaussian2d['_scaling']:  (N,2) sx, sy in pixels (via relu + clamp)
            - gaussian2d['_rotation']: (N,1) rotation logits -> sigmoid * 2π
            - gaussian2d['_features_dc']: (N,3) color logits -> sigmoid in [0,1]
        depth_map: (H,W) depth per pixel in the intrinsics' pixel coordinate system.
        intrinsics: (3,3) camera intrinsics in pixel coordinates.
        extrinsics: (4,4) camera extrinsics (world -> cam).
        eps: Small value used as the thin-axis stddev (sz) to avoid degenerate covariance.

    Returns:
        pos_world: (N',3) Gaussian centers in world space (filtered by valid depth mask).
        scale3d:   (N',3) [sx3, sy3, sz3] in world units.
        quat:      (N',4) rotation quaternions (w,x,y,z) mapping local axes to world.
        color:     (N',3) RGB in [0,1].
    """
    # -------------------------
    # 0) Decode parameterization
    # -------------------------
    u_ndc = torch.tanh(gaussian2d['_xyz'])            # (N,2) in [-1,1]
    scaling = torch.clamp(torch.relu(gaussian2d['_scaling']), 0.5)  # (N,2)
    theta = torch.sigmoid(gaussian2d['_rotation'].squeeze(-1)) * 2 * math.pi  # (N,)
    color = torch.sigmoid(gaussian2d['_features_dc'])

    sx2d = scaling[:, 0]  # pixel units
    sy2d = scaling[:, 1]

    K = intrinsics                  # (3,3)

    # Extrinsics are world->cam; invert to cam->world so all outputs are in world space.
    c2w = torch.linalg.inv(extrinsics)
    R_c2w = c2w[:3, :3]             # (3,3)
    t_c2w = c2w[:3, 3]              # (3,)

    N = u_ndc.shape[0]
    H, W = depth_map.shape[-2], depth_map.shape[-1]

    # -------------------------
    # 1) Sample depth on the depth_map at Gaussian centers (NDC grid_sample).
    # -------------------------
    depth_map_4d = depth_map.view(1, 1, H, W)        # (1,1,H,W)
    grid = u_ndc.view(1, N, 1, 2)                    # (1,N,1,2), [-1,1]
    depth_sample = F.grid_sample(
        depth_map_4d,
        grid,
        mode='nearest',
        align_corners=True,
    )                                                # (1,1,N,1)
    depth = depth_sample.view(N)                     # (N,)

    # -------------------------
    # 2) NDC -> pixel coordinates
    # -------------------------
    x_pix = (u_ndc[:, 0] + 1.0) * 0.5 * (W - 1)
    y_pix = (u_ndc[:, 1] + 1.0) * 0.5 * (H - 1)
    u_pix = torch.stack([x_pix, y_pix], dim=1)       # (N,2)

    ones = torch.ones((N, 1), dtype=u_pix.dtype, device=u_pix.device)
    uv1 = torch.cat([u_pix, ones], dim=1)            # (N,3)

    # -------------------------
    # 3) Unproject 2D mean to 3D points (camera -> world).
    # -------------------------
    Kinv = torch.linalg.inv(K)                       # (3,3)
    X_cam = depth[:, None] * (uv1 @ Kinv.T)          # (N,3)
    mask = depth > 0

    # X_world = R_c2w * X_cam + t_c2w
    pos_world = (R_c2w @ X_cam.T).T + t_c2w[None, :]  # (N,3)

    # -------------------------
    # 4) Lift 2D ellipse axes to 3D directions in camera coordinates.
    # -------------------------
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    d1 = torch.stack([cos_t,  sin_t], dim=1)         # (N,2)
    d2 = torch.stack([-sin_t, cos_t], dim=1)         # (N,2)

    u1_pix = torch.cat([u_pix + d1, ones], dim=1)    # (N,3)
    u2_pix = torch.cat([u_pix + d2, ones], dim=1)    # (N,3)

    X0 = uv1 @ Kinv.T                                # (N,3)
    X1 = u1_pix @ Kinv.T                             # (N,3)
    X2 = u2_pix @ Kinv.T                             # (N,3)

    v1 = X1 - X0
    v2 = X2 - X0

    v1 = v1 / (v1.norm(dim=1, keepdim=True) + 1e-9)
    v2 = v2 / (v2.norm(dim=1, keepdim=True) + 1e-9)

    # Thin axis: camera ray direction.
    z_cam = X0 / (X0.norm(dim=1, keepdim=True) + 1e-9)

    # Gram-Schmidt: build an orthonormal basis in camera coordinates.
    y_cam = torch.cross(z_cam, v1, dim=1)
    y_cam = y_cam / (y_cam.norm(dim=1, keepdim=True) + 1e-9)

    x_cam = torch.cross(y_cam, z_cam, dim=1)
    x_cam = x_cam / (x_cam.norm(dim=1, keepdim=True) + 1e-9)

    R3D_cam = torch.stack([x_cam, y_cam, z_cam], dim=2)  # (N,3,3)

    # Rotate the local Gaussian axes into world coordinates: R_world = R_c2w @ R_cam.
    R3D_world = R_c2w @ R3D_cam                         # (N,3,3)

    # -------------------------
    # 5) Depth scaling: pixel-scale -> world-scale (approx).
    # -------------------------
    fx = K[0, 0]
    fy = K[1, 1]

    sx3 = sx2d * depth / fx                             # (N,)
    sy3 = sy2d * depth / fy                             # (N,)
    sz3 = torch.full_like(sx3, eps)                     # thin-axis stddev

    scale3d = torch.stack([sx3, sy3, sz3], dim=1)       # (N,3)

    # -------------------------
    # 6) Rotation matrix -> quaternion
    # -------------------------
    quat = rotmat_to_quat(R3D_world)                    # (N,4), (w,x,y,z)

    return pos_world[mask], scale3d[mask], quat[mask], color[mask]
