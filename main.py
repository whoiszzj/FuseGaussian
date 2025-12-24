import argparse

from dataset.da3_dataloader import DA3Dataset
from dataset.dtu_dataloader import DTUDataset
from dataset.mvs_dataloader import MVSDataset
from model.fuse_gaussian import FuseGaussian
from utils.io import save_mvs_camera


# Purpose: Parse CLI arguments to configure dataset selection and fusion hyperparameters.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse multi-view 2D Gaussians into 3D Gaussians.")
    parser.add_argument("--dataset", type=str, default="dtu", choices=["dtu", "da3", "mvs"], help="Dataset type.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the scene root directory.")
    parser.add_argument("--device", type=str, default="cuda", help='Torch device string, e.g. "cuda" or "cpu".')

    # Expose FuseGaussian.fuse() knobs
    parser.add_argument("--voxel_size", type=float, default=2.0, help="Voxel edge length used for quantization.")
    parser.add_argument(
        "--max_points_per_gaussian",
        type=int,
        default=8,
        help="Max points per fused gaussian within a voxel (controls K_v).",
    )
    parser.add_argument("--kmeans_max_iter", type=int, default=15, help="Max K-means iterations per voxel.")

    parser.add_argument("--save_cams", action="store_true", help="Save MVS-style camera files to fuse_out/cams.")
    return parser


# Purpose: Construct the appropriate dataset instance according to CLI args.
def build_dataset(dataset_type: str, data_path: str, device: str):
    if dataset_type == "dtu":
        return DTUDataset(data_path=data_path, device=device)
    if dataset_type == "da3":
        return DA3Dataset(data_path=data_path, device=device)
    if dataset_type == "mvs":
        return MVSDataset(data_path=data_path, device=device)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


# Purpose: Entry point to run the end-to-end fusion pipeline with CLI-configurable parameters.
def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = build_dataset(args.dataset, args.data_path, args.device)
    images, cams, depths, gaussians = dataset.get_data()

    fuse_gaussian = FuseGaussian(images, cams, depths, gaussians, device=args.device)
    fuse_gaussian.fuse(
        voxel_size=args.voxel_size,
        max_points_per_gaussian=args.max_points_per_gaussian,
        kmeans_max_iter=args.kmeans_max_iter,
    )

    if args.save_cams:
        save_mvs_camera("fuse_out/cams", cams)

    print("Fuse Gaussian done")


if __name__ == "__main__":
    main()