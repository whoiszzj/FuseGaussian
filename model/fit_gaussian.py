import math
import torch
from utils.math import quat_to_rotmat, rotmat_to_quat

@torch.no_grad()
# Purpose: Run K-means within each voxel in parallel on GPU and return per-point cluster labels.
def batched_kmeans_per_voxel(
    pos_all: torch.Tensor,      # [P,3]
    voxel_ids: torch.Tensor,    # [P] in [0, M-1]
    K_per_voxel: torch.Tensor,  # [M]
    max_iter: int = 15,
    chunk_k: int = 8,           # Stream over K to reduce peak memory usage.
):
    """
    Parallel K-means across all voxels, but only for voxels with K_v > 1.
    Voxels with K_v == 1 are treated as a single cluster with label=0.

    This implementation streams over the K dimension and performs an online argmin to avoid
    allocating large tensors like [P_sub, K_max, 3] for centers_per_point.

    Args:
        pos_all: [P, 3] point positions.
        voxel_ids: [P] voxel id per point in [0, M-1].
        K_per_voxel: [M] number of clusters per voxel.
        max_iter: Maximum K-means iterations.
        chunk_k: K chunk size for streaming distance computation.

    Returns:
        labels_full: [P] cluster id within each voxel, in [0, K_v-1].
    """
    device = pos_all.device
    dtype = pos_all.dtype
    P = pos_all.shape[0]
    M = K_per_voxel.shape[0]

    assert voxel_ids.shape[0] == P
    assert voxel_ids.max().item() < M

    # -----------------------------
    # 0) Select voxels that require clustering (K_v > 1)
    # -----------------------------
    mask_multi_voxel = (K_per_voxel > 1)   # [M]
    if not mask_multi_voxel.any():
        return pos_all.new_zeros(P, dtype=torch.long)

    multi_voxel_ids = mask_multi_voxel.nonzero(as_tuple=False).squeeze(-1)  # [M_multi]
    M_multi = multi_voxel_ids.numel()

    # Map original voxel ids to a compact range [0, M_multi-1].
    mapping = torch.full((M,), -1, dtype=torch.long, device=device)
    mapping[multi_voxel_ids] = torch.arange(M_multi, device=device, dtype=torch.long)

    voxel_ids_mapped = mapping[voxel_ids]               # [P], clustered voxels in [0..M_multi-1], others are -1
    mask_points_multi = (voxel_ids_mapped >= 0)         # [P]
    if not mask_points_multi.any():
        return pos_all.new_zeros(P, dtype=torch.long)

    # Build subset tensors for points that belong to multi-cluster voxels.
    pos_sub         = pos_all[mask_points_multi]        # [P_sub,3]
    voxel_ids_sub   = voxel_ids_mapped[mask_points_multi]  # [P_sub] in [0, M_multi-1]
    K_per_voxel_sub = K_per_voxel[multi_voxel_ids]      # [M_multi]

    P_sub = pos_sub.shape[0]
    K_max = int(K_per_voxel_sub.max().item())
    if K_max <= 1:
        labels_full = pos_all.new_zeros(P, dtype=torch.long)
        return labels_full

    # ----------------------------------------------------
    # 1) Count points per multi-voxel and build sorted indices + offsets
    # ----------------------------------------------------
    counts = torch.bincount(voxel_ids_sub, minlength=M_multi)  # [M_multi], N_v
    sorted_idx = torch.argsort(voxel_ids_sub)                  # [P_sub]
    offsets = torch.cumsum(counts, dim=0) - counts             # [M_multi]

    # ----------------------------------------------------
    # 2) Fully-parallel initialization of centers_sub: [M_multi, K_max, 3]
    #    For each voxel v, sample K_v points evenly over [0, N_v-1].
    # ----------------------------------------------------
    k_grid = torch.arange(K_max, device=device)[None, :]       # [1,K_max]

    Nv = counts.to(device=device, dtype=torch.float32)         # [M_multi]
    Nv_b = Nv[:, None]                                         # [M_multi,1]
    Kv = K_per_voxel_sub.to(device=device, dtype=torch.float32) # [M_multi]
    Kv_b = Kv[:, None]                                         # [M_multi,1]

    denom = torch.clamp(Kv_b - 1.0, min=1.0)                   # [M_multi,1]
    pos_rel = (k_grid.float() / denom) * (Nv_b - 1.0).clamp_min(0.0)  # [M_multi,K_max]

    max_idx_per_voxel = (Nv_b - 1.0).clamp_min(0.0)            # [M_multi,1]
    pos_rel = torch.maximum(pos_rel, torch.zeros_like(pos_rel))
    pos_rel = torch.minimum(pos_rel, max_idx_per_voxel.expand_as(pos_rel))

    pos_idx_in_voxel = pos_rel.round().long()                  # [M_multi,K_max]
    global_pos_idx = offsets[:, None] + pos_idx_in_voxel       # [M_multi,K_max]
    global_pos_idx = global_pos_idx.clamp(min=0, max=P_sub - 1)

    sample_global_idx = sorted_idx[global_pos_idx]             # [M_multi,K_max]
    centers_sub = pos_sub[sample_global_idx]                   # [M_multi,K_max,3]

    # ----------------------------------------------------
    # 3) K-means on the subset: stream over K to compute dist2 and do online argmin
    # ----------------------------------------------------
    labels_sub = torch.zeros(P_sub, dtype=torch.long, device=device)

    centers_sum   = torch.empty(M_multi * K_max, 3, device=device, dtype=dtype)
    centers_count = torch.empty(M_multi * K_max, 1, device=device, dtype=dtype)
    ones          = torch.ones(P_sub, 1, device=device, dtype=dtype)

    k_ids_all = torch.arange(K_max, device=device, dtype=torch.long)     # [K_max]
    Kv_per_point = K_per_voxel_sub[voxel_ids_sub].to(device=device)      # [P_sub] int

    # Precompute x^2 to reduce repeated work in the distance formula.
    x2 = (pos_sub * pos_sub).sum(dim=-1)                                 # [P_sub]

    for _ in range(max_iter):
        # --- Streaming argmin: avoid allocating large [P_sub, K_max, 3] tensors ---
        best_dist = torch.full((P_sub,), float("inf"), device=device, dtype=dtype)
        best_k = torch.zeros((P_sub,), device=device, dtype=torch.long)

        # Iterate over K in chunks.
        for k0 in range(0, K_max, chunk_k):
            k1 = min(k0 + chunk_k, K_max)
            kk = k_ids_all[k0:k1]  # [kc]
            kc = kk.numel()

            # Take a small chunk of centers and gather by voxel_ids_sub.
            # centers_chunk: [P_sub, kc, 3]
            centers_chunk = centers_sub[:, k0:k1, :][voxel_ids_sub]

            # Use ||x-c||^2 = x2 + c2 - 2*x·c to avoid (x-c)**2 materialization.
            c2 = (centers_chunk * centers_chunk).sum(dim=-1)                 # [P_sub,kc]
            xc = (pos_sub[:, None, :] * centers_chunk).sum(dim=-1)           # [P_sub,kc]
            dist2_chunk = x2[:, None] + c2 - 2.0 * xc                         # [P_sub,kc]

            # Mask invalid k where k >= K_v for the point's voxel.
            valid = kk[None, :] < Kv_per_point[:, None]                       # [P_sub,kc]
            dist2_chunk = dist2_chunk + (~valid) * 1e9

            # Local chunk argmin.
            chunk_min, chunk_arg = dist2_chunk.min(dim=1)                     # [P_sub]
            better = chunk_min < best_dist
            best_dist = torch.where(better, chunk_min, best_dist)
            best_k = torch.where(better, chunk_arg + k0, best_k)

        new_labels_sub = best_k

        if torch.equal(new_labels_sub, labels_sub):
            labels_sub = new_labels_sub
            break
        labels_sub = new_labels_sub

        # Update centers_sub using fully-parallel index_add_.
        cluster_index = voxel_ids_sub * K_max + labels_sub                    # [P_sub]
        centers_sum.zero_()
        centers_count.zero_()

        centers_sum.index_add_(0, cluster_index, pos_sub)
        centers_count.index_add_(0, cluster_index, ones)

        centers_new = centers_sum / centers_count.clamp_min(1.0)
        centers_sub = centers_new.view(M_multi, K_max, 3)

    # ----------------------------------------------------
    # 4) Write back labels_full; voxels with K_v==1 keep default label=0
    # ----------------------------------------------------
    labels_full = pos_all.new_zeros(P, dtype=torch.long)
    labels_full[mask_points_multi] = labels_sub
    return labels_full


# -----------------------------------------------------------------------------
# Purpose: Fuse clustered Gaussians per voxel fully in parallel on GPU (no per-voxel loops).
# -----------------------------------------------------------------------------
def fuse_voxels_from_clusters_parallel(
    voxel_coords: torch.Tensor,  # [P,3]
    pos_all:      torch.Tensor,  # [P,3]
    scale_all:    torch.Tensor,  # [P,3]
    quat_all:     torch.Tensor,  # [P,4]
    color_all:    torch.Tensor,  # [P,3]
    labels:       torch.Tensor,  # [P] cluster id within voxel in [0..K_v-1]
    K_per_voxel:  torch.Tensor,  # [M] number of clusters per voxel (K_v)
    weight_all:   torch.Tensor = None,  # [P] or None
):
    """
    Preconditions: K-means has been completed, providing labels and K_per_voxel.
    This function avoids any for-voxel/for-cluster Python loops; aggregation is done via GPU tensor ops.

    Args:
        voxel_coords: [P,3] voxel coordinate for each gaussian.
        pos_all     : [P,3]
        scale_all   : [P,3]
        quat_all    : [P,4] (w,x,y,z)
        color_all   : [P,3]
        labels      : [P]   cluster id within voxel.
        K_per_voxel : [M]   number of clusters per voxel (K_v).
        weight_all  : [P]   optional weights (e.g., opacity); defaults to 1.

    Returns:
        fused_vox   : [Q,3] voxel coordinates for fused gaussians.
        fused_pos   : [Q,3]
        fused_scale : [Q,3]
        fused_quat  : [Q,4]
        fused_color : [Q,3]
    """
    device = pos_all.device
    dtype = pos_all.dtype

    # Recompute unique voxels and inverse mapping to ensure alignment with K_per_voxel.
    unique_vox, inv = torch.unique(voxel_coords, dim=0, return_inverse=True)
    M = unique_vox.shape[0]
    P = pos_all.shape[0]

    assert K_per_voxel.shape[0] == M, "K_per_voxel must align with unique_vox."
    assert labels.shape[0] == P, "labels length must match pos_all."

    # Max clusters across voxels.
    K_max = int(K_per_voxel.max().item())
    num_clusters_total = M * K_max

    if weight_all is None:
        weight_all = pos_all.new_ones(P)  # [P]
    weight_all = weight_all.to(device=device, dtype=dtype)

    # -------------------------
    # Per-point covariance: Σ_i = R_i diag(s^2) R_i^T
    # -------------------------
    R_all = quat_to_rotmat(quat_all)     # [P,3,3]
    s2 = scale_all * scale_all           # [P,3]

    S_local = torch.zeros(P, 3, 3, device=device, dtype=dtype)
    S_local[:, 0, 0] = s2[:, 0]
    S_local[:, 1, 1] = s2[:, 1]
    S_local[:, 2, 2] = s2[:, 2]

    Sigma_i = R_all @ S_local @ R_all.transpose(1, 2)  # [P,3,3]

    # mu_i mu_i^T
    mu_i = pos_all                                     # [P,3]
    mu_i_outer = mu_i[:, :, None] * mu_i[:, None, :]   # [P,3,3]

    # Σ_i + mu_i mu_i^T
    S_i_plus = Sigma_i + mu_i_outer                    # [P,3,3]

    # Expand weights for broadcasting.
    w = weight_all[:, None]                            # [P,1]

    # -------------------------
    # Map (voxel, cluster k) to a global cluster index
    # -------------------------
    voxel_ids = inv                                    # [P] in [0,M-1]
    # Requirement: labels[p] < K_per_voxel[voxel_ids[p]]
    global_cluster_id = voxel_ids * K_max + labels     # [P] in [0, M*K_max-1]

    # -------------------------
    # Allocate accumulation buffers for all clusters
    # -------------------------
    cluster_weight_sum = torch.zeros(num_clusters_total, 1, device=device, dtype=dtype)
    cluster_pos_sum    = torch.zeros(num_clusters_total, 3, device=device, dtype=dtype)
    cluster_color_sum  = torch.zeros(num_clusters_total, 3, device=device, dtype=dtype)
    cluster_S_sum_flat = torch.zeros(num_clusters_total, 9, device=device, dtype=dtype)

    # Flatten 3x3 matrices for index_add_ accumulation.
    S_i_plus_flat = S_i_plus.reshape(P, 9)             # [P,9]

    # -------------------------
    # Parallel accumulation (index_add_)
    # -------------------------
    cluster_weight_sum.index_add_(0, global_cluster_id, w)                    # [P,1]
    cluster_pos_sum.index_add_(0,    global_cluster_id, w * pos_all)          # [P,3]
    cluster_color_sum.index_add_(0,  global_cluster_id, w * color_all)        # [P,3]
    cluster_S_sum_flat.index_add_(0, global_cluster_id, w * S_i_plus_flat)    # [P,9]

    # -------------------------
    # Compute μ, Σ and color for each cluster
    # -------------------------
    W = cluster_weight_sum.clamp_min(1e-8)          # [C,1]
    mu_flat = cluster_pos_sum / W                   # [C,3]
    E_xxT_flat = cluster_S_sum_flat / W             # [C,9]
    E_xxT = E_xxT_flat.view(num_clusters_total, 3, 3)  # [C,3,3]

    mu_outer_flat = (mu_flat[:, :, None] * mu_flat[:, None, :]).view(num_clusters_total, 9)
    mu_outer = mu_outer_flat.view(num_clusters_total, 3, 3)

    Sigma_flat = E_xxT - mu_outer                   # [C,3,3]

    color_flat = cluster_color_sum / W              # [C,3]

    # -------------------------
    # Build a valid cluster mask:
    #   1) k < K_v
    #   2) W > 0 (non-empty)
    # -------------------------
    # Build (v, k) indices for all clusters.
    v_ids = torch.arange(M, device=device).unsqueeze(1).repeat(1, K_max)  # [M,K_max]
    k_ids = torch.arange(K_max, device=device).unsqueeze(0).repeat(M, 1)  # [M,K_max]
    v_ids_flat = v_ids.reshape(-1)     # [C]
    k_ids_flat = k_ids.reshape(-1)     # [C]

    Kv_flat = K_per_voxel[v_ids_flat]  # [C]
    valid_k_mask = k_ids_flat < Kv_flat

    non_empty_mask = (cluster_weight_sum.squeeze(-1) > 0)  # [C]
    valid_mask = valid_k_mask & non_empty_mask             # [C]

    if not valid_mask.any():
        return None

    # Keep only valid clusters.
    mu_valid    = mu_flat[valid_mask]        # [Q,3]
    Sigma_valid = Sigma_flat[valid_mask]     # [Q,3,3]
    color_valid = color_flat[valid_mask]     # [Q,3]
    vox_idx_valid = v_ids_flat[valid_mask]   # [Q]

    # -------------------------
    # Eigen-decompose Σ for all valid clusters to recover scale + rotation
    # -------------------------
    evals, evecs = torch.linalg.eigh(Sigma_valid)       # evals: [Q,3]  evecs: [Q,3,3]
    evals_clamped = evals.clamp_min(1e-6)
    scale_valid = torch.sqrt(evals_clamped)             # [Q,3]
    R_valid = evecs                                     # [Q,3,3]

    quat_valid = rotmat_to_quat(R_valid)                # [Q,4]

    fused_pos   = mu_valid
    fused_scale = scale_valid
    fused_quat  = quat_valid
    fused_color = color_valid
    fused_vox   = unique_vox[vox_idx_valid]             # [Q,3]

    return fused_vox, fused_pos, fused_scale, fused_quat, fused_color


# -----------------------------------------------------------------------------
# Purpose: Top-level API to compute per-voxel K, run voxel-wise K-means, and fuse Gaussians on GPU.
# -----------------------------------------------------------------------------
def cluster_and_fuse_gaussians(
    voxel_coords: torch.Tensor,  # [P,3]
    pos_all:      torch.Tensor,  # [P,3]
    scale_all:    torch.Tensor,  # [P,3]
    quat_all:     torch.Tensor,  # [P,4]
    color_all:    torch.Tensor,  # [P,3]
    max_points_per_gaussian: int = 32,
    kmeans_max_iter: int = 15,
    weight_all: torch.Tensor = None,  # [P], optional weights (e.g., opacity)
):
    """
    One-stop API:
      1) Count points per voxel N_v, compute K_v = ceil(N_v / max_points_per_gaussian)
      2) Run batched K-means across voxels to get per-point cluster ids
      3) Fuse gaussians in parallel on GPU based on clustering results

    Args:
        voxel_coords           : [P,3] voxel coordinate for each gaussian.
        pos_all                : [P,3]
        scale_all              : [P,3]
        quat_all               : [P,4] (w,x,y,z)
        color_all              : [P,3]
        max_points_per_gaussian: Controls how many original points map to a fused gaussian.
        kmeans_max_iter        : K-means iterations.
        weight_all             : [P] optional weights (defaults to 1).

    Returns:
        fused_vox   : [Q,3]
        fused_pos   : [Q,3]
        fused_scale : [Q,3]
        fused_quat  : [Q,4]
        fused_color : [Q,3]
    """
    assert max_points_per_gaussian > 0, "max_points_per_gaussian must be > 0"

    device = pos_all.device
    dtype = pos_all.dtype

    # 1) Compute unique voxels, counts, and K_v per voxel.
    unique_vox, inv, counts = torch.unique(
        voxel_coords.to(device=device), dim=0, return_inverse=True, return_counts=True
    )
    M = unique_vox.shape[0]

    N_per_voxel = counts                                     # [M]
    K_per_voxel = (N_per_voxel + max_points_per_gaussian - 1) // max_points_per_gaussian  # [M]

    # 2) Batched K-means across all voxels.
    labels = batched_kmeans_per_voxel(
        pos_all=pos_all,
        voxel_ids=inv,
        K_per_voxel=K_per_voxel,
        max_iter=kmeans_max_iter,
    )  # [P]

    # 3) Fully-parallel Gaussian fusion using clustering results.
    fused = fuse_voxels_from_clusters_parallel(
        voxel_coords=voxel_coords,
        pos_all=pos_all,
        scale_all=scale_all,
        quat_all=quat_all,
        color_all=color_all,
        labels=labels,
        K_per_voxel=K_per_voxel,
        weight_all=weight_all,
    )

    return fused
