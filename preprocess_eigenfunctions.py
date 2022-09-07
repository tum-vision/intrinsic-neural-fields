import os
from os.path import abspath
import argparse
import numpy as np

from mesh import EigenfuncsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the eigenfunctions of a mesh")
    parser.add_argument("out_dir", type=str, help="Path to the directory where the preprocessed eigenfunctions should be stored")
    parser.add_argument("path_to_mesh", type=str, help="Path to the mesh file")
    parser.add_argument("k_max", type=int, help="Maximum k for the eigenfunctions")
    parser.add_argument("--laplacian_type", default="cotan", type=str, help="Laplacian type: cotan, robust, pc_vert_robust")
    parser.add_argument('--keep_first_efunc', default=False, action="store_true")
    args = parser.parse_args()
    return args


def get_geometry_type(args):
    return "pointcloud" if args.laplacian_type == "pc_vert_robust" else "mesh"


def main():
    args = parse_args()
    print(f"Computing the first {args.k_max} eigenfunctions of a {get_geometry_type(args)}...")

    # Compute the eigenfunctions
    eigenfuncs_proc = EigenfuncsProcessor(args.path_to_mesh, args.k_max, args.laplacian_type,
                                          skip_first_efunc=(not args.keep_first_efunc))

    os.makedirs(args.out_dir, exist_ok=True)

    skip_suffix = "all_efuncs" if args.keep_first_efunc else "skip_first_efuncs"
    # Save eigenfunctions on disk
    np.save(os.path.join(args.out_dir, f"eigenfunctions_{args.laplacian_type}_kmax{args.k_max}_{skip_suffix}.npy"),
            eigenfuncs_proc.get_eigenfunctions())

    # Save eigenvalues on disk
    np.save(os.path.join(args.out_dir, f"eigenvalues_{args.laplacian_type}_kmax{args.k_max}_{skip_suffix}.npy"),
            eigenfuncs_proc.get_eigenvalues())

    # Symlink of mesh
    mesh_dst_path = os.path.join(args.out_dir, os.path.basename(args.path_to_mesh))
    if not os.path.exists(mesh_dst_path):
        # TODO: Could make relative for portability
        os.symlink(src=abspath(args.path_to_mesh), dst=mesh_dst_path)

    print("Done.")


if __name__ == "__main__":
    main()
