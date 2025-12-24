import torch
from utils.io import save_gaussian_ply

@torch.no_grad()
# Purpose: Pack multi-view 3D Gaussians into voxel-indexed tensors on GPU for later clustering/fusion.
def build_gaussian_voxels_gpu(
    gaussian3d_list,
    voxel_indices_list,
    device="cuda"
):
    """
    Aggregate multi-view Gaussians into voxels and return concatenated tensors on GPU.

    Args:
        gaussian3d_list: list of per-view tuples (pos, scale, quat, color).
        voxel_indices_list: list[Tensor], each of shape [N_i, 3] (int voxel coordinates).
        device: Torch device string or torch.device.

    Returns:
        voxel_coords: [P, 3] concatenated voxel coordinates for all gaussians.
        pos_all:      [P, 3] concatenated positions.
        scale_all:    [P, 3] concatenated scales.
        quat_all:     [P, 4] concatenated quaternions.
        color_all:    [P, 3] concatenated colors.
    """
    device = torch.device(device)

    all_voxel_coords = []
    all_pos = []
    all_scale = []
    all_quat = []
    all_color = []
    
    for gaussian3d, vidx in zip(gaussian3d_list, voxel_indices_list):
        pos = gaussian3d[0]
        scale = gaussian3d[1]
        quat = gaussian3d[2]
        color = gaussian3d[3]
        pos = pos.to(device)          # [N,3]
        scale = scale.to(device)    # [N,3]
        quat = quat.to(device)    # [N,4]
        color = color.to(device)    # [N,3]
        vidx = vidx.to(device)      # [N,3] int

        # If you have a validity mask, filter here; currently all gaussians are treated as valid.
        if pos.numel() == 0:
            continue

        all_voxel_coords.append(vidx)
        all_pos.append(pos)
        all_scale.append(scale)
        all_quat.append(quat)
        all_color.append(color)
        
    if len(all_voxel_coords) == 0:
        raise RuntimeError("No Gaussians provided; check your inputs.")

    # Concatenate all views into a single batch.
    voxel_coords = torch.cat(all_voxel_coords, dim=0)  # [P,3]
    pos_all       = torch.cat(all_pos, dim=0)            # [P,3]
    scale_all    = torch.cat(all_scale, dim=0)         # [P,3]
    quat_all    = torch.cat(all_quat, dim=0)         # [P,4]
    color_all   = torch.cat(all_color, dim=0)         # [P,3]
    
    # Debug: write a PLY of all gaussians before fusion.
    save_gaussian_ply(f"fuse_out/gaussians_all.ply",pos_all, scale_all, quat_all, color_all)
    print("Save gaussians done")

    return voxel_coords, pos_all, scale_all, quat_all, color_all

    # # Group by voxel_coords (kept for reference; not used in current pipeline)
    # unique_coords, inverse = torch.unique(
    #     voxel_coords, dim=0, return_inverse=True
    # )  # unique_coords: [M,3], inverse: [P] in [0,M-1]

    # (
    #     voxel_coords_packed,
    #     gaussians,
    #     mask
    # ) = _pack_gaussians_by_group_with_cap(
    #     pos_all,
    #     scale_all,
    #     quat_all,
    #     color_all,
    #     inverse,
    #     unique_coords,
    #     max_gaussians_per_voxel=max_gaussians_per_voxel,
    # )

    # return voxel_coords_packed, gaussians, mask


# Purpose: Group gaussians by voxel id with an upper bound per voxel, splitting overflow into sub-voxels.
def _pack_gaussians_by_group_with_cap(
    pos,
    scale,
    quat,
    color,
    group_ids,
    unique_coords,
    max_gaussians_per_voxel=128,
):
    """
    Group P gaussians by group_ids, capping each voxel to max_gaussians_per_voxel.
    When a voxel has more than the cap, it is split into multiple "sub-voxels".

    Args:
        pos:         [P, 3]
        scale:      [P, 3]
        quat:      [P, 4]
        color:      [P, 3]
        group_ids:  [P] in [0, M-1], indices into unique_coords.
        unique_coords: [M, 3], original voxel coordinates.
        max_gaussians_per_voxel: Maximum gaussians stored per voxel slot.

    Returns:
        voxel_coords_packed: [M', 3]
        pos_padded:          [M', K, 3]
        scale_padded:       [M', K, 3]
        quat_padded:       [M', K, 4]
        mask_padded:        [M', K] bool
        packed2orig:        [M'], mapping from sub-voxel index to original voxel index (0..M-1).
    """
    device = pos.device
    group_ids = group_ids.to(torch.long)
    P = pos.shape[0]
    M = int(unique_coords.shape[0])

    # 1) Gaussian count per original voxel.
    counts = torch.bincount(group_ids, minlength=M)  # [M]

    max_pts = int(max_gaussians_per_voxel)
    # 2) Number of sub-voxel slots per voxel (ceil division).
    num_slots_per_group = (counts + max_pts - 1) // max_pts  # ceil

    # 3) Total number of sub-voxels.
    total_slots = int(num_slots_per_group.sum().item())
    if total_slots == 0:
        raise RuntimeError("No Gaussians in any group; check your inputs.")

    # 4) Expanded voxel coordinates and mapping back to the original voxel index.
    packed2orig = torch.arange(M, device=device).repeat_interleave(num_slots_per_group)
    voxel_coords_packed = unique_coords.to(device).repeat_interleave(
        num_slots_per_group, dim=0
    )  # [M',3]

    # 5) Sort by group_ids so items in the same voxel are contiguous.
    order = torch.argsort(group_ids)
    group_sorted = group_ids[order]  # [P]
    pos_sorted    = pos[order]         # [P,3]
    scale_sorted = scale[order]      # [P,3]
    quat_sorted = quat[order]      # [P,4]
    color_sorted = color[order]      # [P,3]
    # 6) Offsets into the sorted arrays for each voxel.
    offsets = torch.zeros(M, dtype=torch.long, device=device)
    if M > 0:
        offsets[1:] = counts.cumsum(0)[:-1]  # [M]

    idx_sorted = torch.arange(P, device=device)
    index_in_group = idx_sorted - offsets[group_sorted]  # [P], 0..counts[i]-1

    # 7) Within a voxel, the j-th gaussian maps to:
    #    slot_id     = j // max_pts
    #    local_index = j %  max_pts
    slot_id = index_in_group // max_pts         # [P]
    local_index = index_in_group % max_pts      # [P]

    # 8) Starting sub-voxel index for each original voxel.
    slots_offsets = torch.zeros(M, dtype=torch.long, device=device)
    if M > 0:
        slots_offsets[1:] = num_slots_per_group.cumsum(0)[:-1]  # [M]

    # 9) Sub-voxel id for each gaussian.
    new_group_id = slots_offsets[group_sorted] + slot_id        # [P] in [0, M'-1]

    # 10) Allocate packed tensors.
    K = max_pts
    pos_padded = torch.zeros(
        (total_slots, K, 3),
        device=device,
        dtype=pos.dtype,
    )
    scale_padded = torch.zeros(
        (total_slots, K, 3),
        device=device,
        dtype=scale.dtype,
    )
    quat_padded = torch.zeros(
        (total_slots, K, 4),
        device=device,
        dtype=quat.dtype,
    )
    color_padded = torch.zeros(
        (total_slots, K, 3),
        device=device,
        dtype=color.dtype,
    )
    mask_padded = torch.zeros(
        (total_slots, K),
        device=device,
        dtype=torch.bool,
    )

    # 11) Scatter into packed tensors via advanced indexing.
    pos_padded[new_group_id, local_index] = pos_sorted
    scale_padded[new_group_id, local_index] = scale_sorted
    quat_padded[new_group_id, local_index] = quat_sorted
    color_padded[new_group_id, local_index] = color_sorted
    mask_padded[new_group_id, local_index] = True

    return voxel_coords_packed, (pos_padded, scale_padded, quat_padded, color_padded), mask_padded
