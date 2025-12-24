import torch
import os
import numpy as np
from plyfile import PlyElement, PlyData


# Purpose: Save a 3D point cloud to an ASCII .ply file for visualization.
def save_point_cloud_as_ply(points, filename):
    """
    Save a 3D point cloud to an ASCII .ply file.

    Args:
        points (torch.Tensor): Tensor with shape (K, 3) containing 3D coordinates.
        filename (str): Output path for the .ply file.
    """
    pts = points.detach().cpu().reshape(-1, 3)
    num_points = pts.shape[0]

    with open(filename, "w") as f:
        # Write a minimal PLY header for vertex-only point cloud.
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write XYZ coordinates.
        for p in pts:
            x, y, z = p.tolist()
            f.write(f"{x} {y} {z}\n")
            
# Purpose: Convert a torch/numpy matrix to a float32 numpy array with strict shape validation.
def _to_numpy_matrix(mat, shape, name: str):
    """
    Convert a torch Tensor / numpy array to a float32 numpy matrix with strict shape checking.

    Args:
        mat: Input matrix (torch.Tensor or array-like).
        shape: Expected shape, e.g. (3, 3) or (4, 4).
        name: Matrix name for error messages.

    Returns:
        np.ndarray: float32 numpy array with the expected shape.
    """
    if torch.is_tensor(mat):
        mat_np = mat.detach().cpu().numpy()
    else:
        mat_np = np.asarray(mat)

    if mat_np.shape != tuple(shape):
        raise ValueError(f"Invalid {name} shape: expected {shape}, got {mat_np.shape}")

    return mat_np.astype(np.float32, copy=False)

# Purpose: Save MVS-style camera files ({:08d}_cam.txt) that match readMVSCamera() line layout.
def save_mvs_camera(path, cams):
    """
    Save a list of (intrinsics, extrinsics) camera matrices to MVS-style *_cam.txt files.

    The output layout is designed to be compatible with:
      - extrinsics read from lines[1:5] (4 rows)
      - intrinsics read from lines[7:10] (3 rows)

    Args:
        path (str): Output directory for camera txt files.
        cams (list): List of tuples (cam_intrinsics, cam_extrinsics), each being torch tensors or array-like.
    """
    os.makedirs(path, exist_ok=True)

    for idx, cam in enumerate(cams):
        if not isinstance(cam, (tuple, list)) or len(cam) != 2:
            raise ValueError("Each camera must be a (intrinsics, extrinsics) tuple.")

        cam_intrinsics, cam_extrinsics = cam
        K = _to_numpy_matrix(cam_intrinsics, (3, 3), "intrinsics")
        extr = _to_numpy_matrix(cam_extrinsics, (4, 4), "extrinsics")

        out_path = os.path.join(path, f"{idx:04d}_cam.txt")
        with open(out_path, "w") as f:
            # Line 0
            f.write("extrinsic\n")
            # Lines 1-4: 4x4 extrinsics
            for r in range(4):
                f.write(" ".join(f"{v:.8f}" for v in extr[r].tolist()) + "\n")
            # Line 5
            f.write("\n")
            # Line 6
            f.write("intrinsic\n")
            # Lines 7-9: 3x3 intrinsics
            for r in range(3):
                f.write(" ".join(f"{v:.8f}" for v in K[r].tolist()) + "\n")
            # Optional trailing newline
            f.write("\n")
    
# Purpose: Save fused 3D Gaussian parameters to a PLY file (layout aligned with 3DGS-style attributes).
def save_gaussian_ply(path, pos, scale, rotation, color, sh_degree: int = 0):
    """
    Save 3D Gaussian parameters to a PLY file (field layout aims to be compatible with 3DGS).

    Args:
        path: Output file path.
        pos: [M, 3] Gaussian centers.
        scale: [M, 3] Gaussian scales (sx, sy, sz).
        rotation: [M, 4] Quaternion (qw, qx, qy, qz).
        color: [M, 3] RGB values in [0, 1].
        sh_degree: Spherical harmonics degree. Degree=3 corresponds to 48 SH coefficients (3*(degree+1)^2).

    Notes:
        For missing attributes in the 3DGS-style layout, this function fills:
          - nx, ny, nz = 0
          - f_rest_* = 0
          - opacity = 1 (stored as 0 in logit space if required by downstream; here we store zeros)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert tensors to numpy arrays for PlyData writing.
    xyz = pos.detach().cpu().numpy()       # [M,3]
    M = xyz.shape[0]
    if scale is not None:
        scale_np = torch.log(scale).detach().cpu().numpy()   # [M,3]
    else:
        scale_np = np.zeros((M, 3), dtype=np.float32)
    if rotation is not None:
        rot_np = rotation.detach().cpu().numpy()  # [M,4]
    else:
        rot_np = np.zeros((M, 4), dtype=np.float32)
    if color is not None:
        color_np = color.detach().cpu().numpy()  # [M,3]

    M = xyz.shape[0]

    # 1) Define attribute dimensions (3DGS-like).
    n_sh = (sh_degree + 1) ** 2          # e.g., degree=3 -> 16
    n_sh_total = 3 * n_sh                # RGB channels, e.g., 48
    n_dc = 3                             # DC uses one SH basis => 3 channels (RGB)
    n_rest = n_sh_total - n_dc           # remaining SH coefficients, e.g., 45

    # 2) Construct per-vertex attributes.
    normals = np.zeros_like(xyz)         # [M,3] -> nx, ny, nz = 0

    # f_dc: converted from input RGB color using SH DC convention
    SH_C0 = 0.28209479177387814
    f_dc = ((color_np - 0.5) / SH_C0).astype(np.float32)

    # f_rest: fill zeros for higher-order SH terms.
    if n_rest > 0:
        f_rest = np.zeros((M, n_rest), dtype=np.float32)
    else:
        f_rest = np.zeros((M, 0), dtype=np.float32)

    # opacity: filled with zeros here as a placeholder.
    opacities = np.zeros((M, 1), dtype=np.float32)

    # 3) Define PLY attribute order (3DGS-like naming convention).
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # f_dc_*
    for i in range(n_dc):
        attr_names.append(f'f_dc_{i}')
    # f_rest_*
    for i in range(n_rest):
        attr_names.append(f'f_rest_{i}')
    # opacity
    attr_names.append('opacity')
    # scale_*
    for i in range(3):
        attr_names.append(f'scale_{i}')
    # rot_*
    for i in range(4):
        attr_names.append(f'rot_{i}')

    # Build numpy dtype for PlyElement.
    dtype_full = [(name, 'f4') for name in attr_names]

    # 4) Concatenate all attributes and write to PLY.
    attributes = np.concatenate(
        [
            xyz,            # (M,3)
            normals,        # (M,3)
            f_dc,           # (M,n_dc)
            f_rest,         # (M,n_rest)
            opacities,      # (M,1)
            scale_np,       # (M,3)
            rot_np,         # (M,4)
        ],
        axis=1
    )

    elements = np.empty(M, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    ply_el = PlyElement.describe(elements, 'vertex')
    PlyData([ply_el]).write(path)

    print(
        f"[✔] Gaussian PLY saved to {path}, "
        f"total {M} splats"
    )
