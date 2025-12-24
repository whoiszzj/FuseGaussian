from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import torch


class BaseDataLoader(ABC):
    # Purpose: Initialize the base dataloader with shared configuration (e.g., device and scene name).
    def __init__(self, data_path: str, device: Union[str, torch.device] = "cuda") -> None:
        """
        Args:
            data_path: Root directory of a scene.
            device: Target torch device (e.g., "cuda", "cpu", torch.device("cuda:0")).
        """
        self.data_path = data_path
        self.scene_name = self.data_path.rstrip("/").split("/")[-1]
        self.device = torch.device(device)

    # Purpose: Return the number of available views/samples in the dataset.
    @abstractmethod
    def __len__(self) -> int:
        """Returns:
            Number of views/samples in this dataset.
        """
        raise NotImplementedError

    # Purpose: Load and return a single RGB image tensor for the given index.
    @abstractmethod
    def get_image(self, idx: int) -> torch.Tensor:
        """Args:
            idx: View index.
        Returns:
            Image tensor (typically HxWx3, float32 in [0, 1]).
        """
        raise NotImplementedError

    # Purpose: Load and return camera intrinsics/extrinsics tensors for the given index.
    @abstractmethod
    def get_cam(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            idx: View index.
        Returns:
            (intrinsics, extrinsics) as torch tensors.
        """
        raise NotImplementedError

    # Purpose: Load and return a single depth map tensor for the given index.
    @abstractmethod
    def get_depth(self, idx: int) -> torch.Tensor:
        """Args:
            idx: View index.
        Returns:
            Depth map tensor (typically HxW).
        """
        raise NotImplementedError

    # Purpose: Load and return the gaussian model checkpoint/object for the given index.
    @abstractmethod
    def get_gaussian(self, idx: int) -> Any:
        """Args:
            idx: View index.
        Returns:
            Gaussian checkpoint/object as loaded by torch.load().
        """
        raise NotImplementedError

    # Purpose: Load and return all images in the dataset as a list.
    def get_images(self) -> List[torch.Tensor]:
        """Returns:
            A list of image tensors in index order [0..len(self)-1].
        """
        return [self.get_image(i) for i in range(len(self))]

    # Purpose: Load and return all cameras in the dataset as a list.
    def get_cams(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns:
            A list of (intrinsics, extrinsics) tuples in index order.
        """
        return [self.get_cam(i) for i in range(len(self))]

    # Purpose: Load and return all depth maps in the dataset as a list.
    def get_depths(self) -> List[torch.Tensor]:
        """Returns:
            A list of depth map tensors in index order.
        """
        return [self.get_depth(i) for i in range(len(self))]

    # Purpose: Load and return all gaussian checkpoints/objects in the dataset as a list.
    def get_gaussians(self) -> List[Any]:
        """Returns:
            A list of gaussian checkpoints/objects in index order.
        """
        return [self.get_gaussian(i) for i in range(len(self))]

    # Purpose: Load and return (images, cams, depths, gaussians) with consistent ordering.
    def get_data(self) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[Any]]:
        """Returns:
            A 4-tuple: (images, cams, depths, gaussians), each aligned by index.
        """
        return self.get_images(), self.get_cams(), self.get_depths(), self.get_gaussians()


