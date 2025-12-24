import os
from PIL import Image
import numpy as np
import re
import torch
from dataset.base_dataloader import BaseDataLoader


class DTUDataset(BaseDataLoader):
    # Purpose: Initialize DTU dataset paths and indexable file lists.
    def __init__(self, data_path, device="cuda"):
        """
        Args:
            data_path: Root directory of a DTU scene.
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
        cams_paths = [os.path.join(self.cams_path, f) for f in os.listdir(self.cams_path) if f.endswith('_cam.txt')]
        depth_paths = [os.path.join(self.depth_path, f) for f in os.listdir(self.depth_path) if f.endswith('.pfm')]
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
        file_name = f"{idx:08d}.jpg" if self.images_paths[idx].endswith('.jpg') else f"{idx:08d}.png"
        assert file_name in self.images_paths[idx].split('/')[-1], "File name does not match"
        
        image = Image.open(self.images_paths[idx])
        image = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image).to(self.device)
        
    # Purpose: Parse a DTU camera file and return intrinsics/extrinsics and depth sampling metadata.
    def read_cam_file(self, filename):
        """Args:
            filename: Path to a DTU *_cam.txt file.
        Returns:
            intrinsics (3x3), extrinsics (4x4), depth_min, depth_interval.
        """
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval
    
    # Purpose: Load and return camera intrinsics and extrinsics tensors at the given index.
    def get_cam(self, idx):
        """Args:
            idx: View index.
        Returns:
            (intrinsics, extrinsics) tensors on self.device.
        """
        file_name = f"{idx:08d}_cam.txt"
        assert file_name in self.cams_paths[idx].split('/')[-1], "File name does not match"
        
        cam_intrinsics, cam_extrinsics, depth_min, depth_interval = self.read_cam_file(self.cams_paths[idx])
        return (torch.from_numpy(cam_intrinsics).to(self.device), torch.from_numpy(cam_extrinsics).to(self.device))
        
    # Purpose: Read a PFM file and return (data, scale) for depth loading.
    def read_pfm(self, filename):
        """Args:
            filename: Path to a .pfm file.
        Returns:
            (data, scale) where data is a numpy array and scale is a float.
        """
        file = open(filename, 'rb')
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        # Negative scale indicates little-endian storage (common for PFM).
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        file.close()
        return data, scale
    
    # Purpose: Load and return a depth map tensor at the given index.
    def get_depth(self, idx):
        """Args:
            idx: View index.
        Returns:
            Depth tensor on self.device.
        """
        file_name = f"depth_map_{idx:04d}.pfm"
        assert file_name in self.depth_paths[idx].split('/')[-1], "File name does not match"
        
        depth = np.array(self.read_pfm(self.depth_paths[idx])[0], dtype=np.float32)
        return torch.from_numpy(depth).to(self.device)
    
    
    # Purpose: Load and return a gaussian model checkpoint/object at the given index.
    def get_gaussian(self, idx):
        """Args:
            idx: View index.
        Returns:
            Gaussian checkpoint/object loaded onto self.device (map_location).
        """
        file_name = f"{idx:08d}/net/gaussian_model.pth.tar"
        assert file_name in self.gaussians_paths[idx], "File name does not match"
        
        gaussians = torch.load(self.gaussians_paths[idx], map_location=self.device)
        return gaussians
    
    
    
if __name__ == "__main__":
    dataset = DTUDataset(data_path="/home/zzj/Work/Data/DTU/test/scan10")
    images, cams, depths, gaussians = dataset.get_data()
    print(len(images))
    print(len(cams))
    print(len(depths))
    print(len(gaussians))
    print(images[0].shape)
    print(cams[0])
    print(depths[0].shape)
    print(gaussians[0]['_xyz'].shape)