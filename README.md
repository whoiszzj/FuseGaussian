# FuseGaussian

This project fuses multi-view **2D Gaussians** (one checkpoint per view) with camera and depth information into sparse **3D Gaussians** (exported as PLY), for visualization and downstream processing.

## Overview

- **Dataset**: DA3 / DTU / MVS dataloaders (`dataset/`) with a unified base interface:
  - `get_images / get_cams / get_depths / get_gaussians / get_data`
- **Model**:
  - `gaussian2d_to_3d`: unprojects single-view 2D Gaussians + depth + camera matrices into 3D (world coordinates)
  - `voxel_grid`: aggregates multi-view 3D Gaussians into voxels
  - `fit_gaussian`: voxel-wise parallel K-means + parallel fusion (GPU tensor ops)
  - `fuse_gaussian`: end-to-end fusion pipeline wrapper
- **IO**:
  - exports camera files and 3D Gaussian PLY (`utils/io.py`)

## Project Structure

```
FuseGaussian/
  dataset/                 # DA3/DTU/MVS dataloaders + base class
  model/                   # core logic: 2D->3D, voxel grouping, clustering & fusion
  utils/                   # IO and math utilities
  main.py                  # example entry
```

## Dependencies

- Python: recommended `>= 3.8`
- Main dependencies:
  - `torch`
  - `numpy`
  - `Pillow`
  - `plyfile`

Install what you need (example):

```bash
pip install numpy pillow plyfile
```

Install `torch` following the official guide for your CUDA/CPU environment.

## Data Layout (Conventions)

The three dataloaders have slightly different layouts, but they all need to align the following modalities:

- `images/`: RGB images (`.png` / `.jpg`)
- `cams/`: camera intrinsics/extrinsics
- `depths/` or `APD/`: depth maps (format depends on dataset)
- `gaussians/`: per-view 2D Gaussian checkpoints (default path: `*/net/gaussian_model.pth.tar`)

See:

- `dataset/da3_dataloader.py`
- `dataset/dtu_dataloader.py`
- `dataset/mvs_dataloader.py`

## Quick Start

Run via CLI arguments (recommended):

```bash
python3 main.py --dataset dtu --data_path /path/to/scene --device cuda
```

### Key Arguments

- `--dataset`: dataset type, one of `dtu`, `da3`, `mvs`
- `--data_path`: path to the scene root directory (**required**)
- `--device`: torch device string, e.g. `cuda` or `cpu`
- `--voxel_size`: voxel edge length used to quantize 3D positions (default: `2.0`)
- `--max_points_per_gaussian`: max points per fused gaussian within a voxel (default: `8`)
- `--kmeans_max_iter`: max K-means iterations per voxel (default: `15`)
- `--save_cams`: if set, save MVS-style camera files to `fuse_out/cams`

Example with all knobs:

```bash
python3 main.py \
  --dataset dtu \
  --data_path /path/to/scene \
  --device cuda \
  --voxel_size 2 \
  --max_points_per_gaussian 8 \
  --kmeans_max_iter 15 \
  --save_cams
```

### Outputs (default)

- `fuse_out/fuse_gaussian.ply`: fused 3D Gaussians (PLY)
- `fuse_out/gaussians_all.ply`: all 3D Gaussians before fusion (for debugging/inspection)
If `--save_cams` is set:
- `fuse_out/cams/`: exported camera files in MVS-style `*_cam.txt` format

## Notes

- The pipeline targets GPU by default (`device="cuda"`). To run on CPU, pass `device="cpu"` explicitly (much slower).
- This repository does not include datasets; please prepare your own and follow the dataloader conventions.


