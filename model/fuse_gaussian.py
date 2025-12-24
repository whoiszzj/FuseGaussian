import torch
import os
from model.gaussian2d_to_3d import gaussian2d_to_3d
from model.voxel_grid import build_gaussian_voxels_gpu
from utils.io import save_gaussian_ply, save_point_cloud_as_ply
from model.fit_gaussian import cluster_and_fuse_gaussians

class FuseGaussian():
    # Initialize FuseGaussian with multi-view images, cameras, depths and 2D Gaussians.
    def __init__(self, images, cams, depths, gaussians, device="cuda"):
        # Move all input data to the specified device
        self.device = device
        self.images = [img.to(self.device) for img in images]
        self.cams = [[cam.to(self.device) for cam in cam_group] for cam_group in cams]
        self.depths = [depth.to(self.device) for depth in depths]

        self.gaussians = []
        for g in gaussians:
            gaussian_on_device = {}
            for k, v in g.items():
                gaussian_on_device[k] = v.to(self.device)
            self.gaussians.append(gaussian_on_device)
        self.image_num = len(images)

    # Fuse multi-view 2D Gaussians into a sparse 3D voxel grid.
    def fuse(self, voxel_size=2, max_points_per_gaussian=8, kmeans_max_iter=15):
        """
        Args:
            voxel_size: Voxel edge length used to quantize 3D positions into voxel indices.
                Smaller values create finer voxels (more clusters/less fusion); larger values fuse more aggressively.
            max_points_per_gaussian: Upper bound of points assigned to a fused gaussian within a voxel.
                This controls per-voxel K as K_v = ceil(N_v / max_points_per_gaussian).
            kmeans_max_iter: Maximum number of K-means iterations for voxel-wise clustering.
        """
        # 1) Convert all per-view 2D Gaussians into 3D Gaussians in world space.
        gaussian3d = []
        for i in range(self.image_num):
            gaussian3d_i = gaussian2d_to_3d(self.gaussians[i], self.depths[i], self.cams[i][0], self.cams[i][1])
            gaussian3d.append(gaussian3d_i)
            # Debug Save
            # os.makedirs("fuse_out/points_3d", exist_ok=True)
            # save_point_cloud_as_ply(gaussian3d_i[0], f"fuse_out/points_3d/gaussian3d_{i}.ply")
            # print(f"Save point cloud of gaussian3d_{i} done")
        # gaussian3d: list([N, 3], [N, 3, 3]), list length == M
        print("Convert 2D Gaussians to 3D Gaussians done")
        
        # 2) Compute voxel indices by quantizing 3D positions.
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        idxs = []
        for i in range(self.image_num):
            points = gaussian3d[i][0]
            idx = (points / voxel_size).floor().long()
            idxs.append(idx)
        # idxs: list([N, 3]), list length == M
        
        # 3) Group gaussians by voxel on GPU and pack into contiguous tensors.
        voxel_coords, pos_all, scale_all, quat_all, color_all = build_gaussian_voxels_gpu(
            gaussian3d,
            idxs,
            device=self.device
        )
        print("Build voxel grid done")
        
        # 4) Cluster and fuse gaussians within each voxel to reduce redundancy.
        fused_vox, fused_pos, fused_scale, fused_quat, fused_color = cluster_and_fuse_gaussians(
            voxel_coords,
            pos_all,
            scale_all,
            quat_all,
            color_all,
            max_points_per_gaussian=max_points_per_gaussian,
            kmeans_max_iter=kmeans_max_iter,
        )
        
        save_gaussian_ply(f"fuse_out/fuse_gaussian.ply", fused_pos, fused_scale, fused_quat, fused_color)
        print("Save fuse gaussian done")
