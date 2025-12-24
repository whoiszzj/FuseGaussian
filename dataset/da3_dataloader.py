import os
from PIL import Image
import numpy as np
import re
import torch
from dataset.base_dataloader import BaseDataLoader


class DA3Dataset(BaseDataLoader):
    # Purpose: Initialize DA3 dataset paths and indexable file lists.
    def __init__(self, data_path, device="cuda"):
        """
        Args:
            data_path: Root directory of a DA3 scene.
            device: Target torch device (e.g., "cuda", "cpu").
        """
        super().__init__(data_path=data_path, device=device)
        
        self.images_path = os.path.join(self.data_path, 'images')
        self.cams_path = os.path.join(self.data_path, 'cams')
        self.depth_path = os.path.join(self.data_path, 'depths')
        self.gaussians_path = os.path.join(self.data_path, 'gaussians')
        # Validate required sub-directories before building file lists.
        assert os.path.exists(self.images_path) and os.path.exists(self.cams_path) and os.path.exists(self.depth_path) and os.path.exists(self.gaussians_path), "Images, cams, depth maps and gaussians paths do not exist"
        
        images_paths = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path) if f.endswith('.png') or f.endswith('.jpg')]
        cams_paths = [os.path.join(self.cams_path, f) for f in os.listdir(self.cams_path) if f.endswith('.npz')]
        depth_paths = [os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path) if f.endswith('.npy')]
        gaussians_paths = [os.path.join(self.gaussians_path, f, "net", "gaussian_model.pth.tar") for f in os.listdir(self.gaussians_path)]
        # Ensure view counts match across modalities so idx aligns between image/cam/depth/gaussian.
        assert len(images_paths) == len(cams_paths) == len(depth_paths) == len(gaussians_paths), "Number of images, cams, depth maps and gaussians do not match"
        self.images_paths = sorted(images_paths)
        self.cams_paths = sorted(cams_paths)
        self.depth_paths = sorted(depth_paths)
        self.gaussians_paths = sorted(gaussians_paths)
        self.interval_scale = 1.06
        
    # Purpose: Return the number of views/samples in the dataset.
    def __len__(self):
        """Returns:
            Number of views/samples.
        """
        return len(self.images_paths)
    
    # Purpose: Load and return one RGB image tensor at the given index.
    def get_image(self, idx):
        """Args:
            idx: View index.
        Returns:
            Image tensor on self.device.
        """
        file_name = f"{idx:04d}.jpg" if self.images_paths[idx].endswith('.jpg') else f"{idx:04d}.png"
        assert file_name in self.images_paths[idx].split('/')[-1], "File name does not match"
        
        image = Image.open(self.images_paths[idx])
        image = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image).to(self.device)
    
    # Purpose: Load and return camera intrinsics and extrinsics tensors at the given index.
    def get_cam(self, idx):
        """Args:
            idx: View index.
        Returns:
            (intrinsics, extrinsics) tensors on self.device.
        """
        file_name = f"{idx:04d}.npz"
        assert file_name in self.cams_paths[idx].split('/')[-1], "File name does not match"
        cam = np.load(self.cams_paths[idx])
        cam_intrinsics = cam['intrinsics']
        cam_extrinsics = cam['extrinsics']
        if cam_extrinsics.shape[0] == 3 and cam_extrinsics.shape[1] == 4:
            # Some DA3 exports store extrinsics as 3x4; convert to a standard 4x4 matrix.
            cam_extrinsics = np.concatenate([cam_extrinsics, np.array([[0, 0, 0, 1]])], axis=0).reshape(4, 4)
        return (torch.from_numpy(cam_intrinsics).to(self.device).float(), torch.from_numpy(cam_extrinsics).to(self.device).float())
        
    # Purpose: Load and return a depth map tensor at the given index.
    def get_depth(self, idx):
        """Args:
            idx: View index.
        Returns:
            Depth tensor on self.device.
        """
        file_name = f"{idx:04d}.npy"
        assert file_name in self.depth_paths[idx].split('/')[-1], "File name does not match"
        
        depth = np.load(self.depth_paths[idx])
        return torch.from_numpy(depth).to(self.device).float()
    
    
    # Purpose: Load and return a gaussian model checkpoint/object at the given index.
    def get_gaussian(self, idx):
        """Args:
            idx: View index.
        Returns:
            Gaussian checkpoint/object loaded onto self.device (map_location).
        """
        file_name = f"{idx:04d}/net/gaussian_model.pth.tar"
        assert file_name in self.gaussians_paths[idx], "File name does not match"
        
        gaussians = torch.load(self.gaussians_paths[idx], map_location=self.device)
        return gaussians
    
    
    
if __name__ == "__main__":
    dataset = DA3Dataset(data_path="/home/zzj/Work/Data/Temp/scan10/output")
    images, cams, depths, gaussians = dataset.get_data()
    print(len(images))
    print(len(cams))
    print(len(depths))
    print(len(gaussians))
    print(images[0].shape)
    print(cams[0])
    print(depths[0].shape)
    print(gaussians[0]['_xyz'].shape)