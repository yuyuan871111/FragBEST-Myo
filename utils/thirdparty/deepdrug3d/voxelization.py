# This file is derived from DeepDrug3D 
# Original copyright (c) 2019 Pu et al.

# Modifications:
# - Adapted for Python 3.10 compatibility 
# - Integrated into FragBEST-Myo workflow 

# Modified by Yu-Yuan (Stuart) Yang, 2024 
# Licensed under GPL-3.0 (see LICENSE)

import argparse

import h5py
import numpy as np

if __name__ == "__main__":
    from build_grid import Grid3DBuilder, cal_n_points
else:
    from .build_grid import Grid3DBuilder, cal_n_points


def map_xyz_to_voxel(
    xyz_coords: np.array,
    voxel: np.array,
    voxel_list: list | range,
    voxel_start: int,
    spacing: float,
    potentials: np.array,
):
    """
    Map the coordinates of the atoms to the voxel representation
    """
    cnt = 0
    for x in voxel_list:
        for y in voxel_list:
            for z in voxel_list:
                temp_voxloc = np.array([x, y, z]).astype(float) * spacing
                distances = np.linalg.norm(xyz_coords - temp_voxloc, axis=1)
                min_dist = np.min(distances)
                index = np.where(distances == min_dist)
                if min_dist < spacing / 2:
                    voxel[
                        :, x - voxel_start, y - voxel_start, z - voxel_start
                    ] = potentials[index, :]
                    cnt += 1
                else:
                    voxel[
                        :, x - voxel_start, y - voxel_start, z - voxel_start
                    ] = np.zeros(
                        (potentials.shape[1],)
                    )  # original codes with potentials (channels) are shown using "np.ones" instead of "np.zeros".
    print(f"Number of existing points in the voxel: {cnt}")

    return voxel


def site_voxelization(site, r, spacing, shape):
    """
    Convert the binding site information to numpy array
    """
    site = np.array(site, dtype=np.float64)

    N = cal_n_points(r, spacing)
    if N % 2 == 0:
        # The voxel length is even
        voxel_length = N
        voxel_start = -voxel_length // 2
        voxel_end = voxel_length // 2
        coords = site[:, 0:3] - spacing / 2
    else:
        # The voxel length is odd
        voxel_length = N + 1
        voxel_start = -voxel_length // 2
        voxel_end = voxel_length // 2
        coords = site[:, 0:3]
    voxel_list = range(voxel_start, voxel_end, 1)
    assert len(voxel_list) == voxel_length, "The voxel length is not correct."

    if not shape:
        print("DFIRE potential included in the voxel representation")
        potentials = site[:, 3:]
        voxel = np.zeros(
            shape=(potentials.shape[1], voxel_length, voxel_length, voxel_length),
            dtype=np.float64,
        )
        map_xyz_to_voxel(
            xyz_coords=coords,
            voxel=voxel,
            voxel_list=voxel_list,
            voxel_start=voxel_start,
            spacing=spacing,
            potentials=potentials,
        )

    else:
        print("Binary occupation only for voxel representation")
        potentials = np.ones((site.shape[0], 1))
        voxel = np.zeros(
            shape=(1, voxel_length, voxel_length, voxel_length), dtype=np.float64
        )
        map_xyz_to_voxel(
            xyz_coords=coords,
            voxel=voxel,
            voxel_list=voxel_list,
            voxel_start=voxel_start,
            spacing=spacing,
            potentials=potentials,
        )
    return voxel


class Vox3DBuilder(object):
    """
    This class convert the pdb file to the voxel representation for the input
    of deep learning architecture. The conversion is around 30 mins.
    """

    @staticmethod
    def voxelization(
        pdb_path: str,
        aux_input_path: str,
        output_folder: str,
        r: int = 15,
        spacing: float = 1,
        shape: bool = False,
    ):
        """
        Input:
            pdb_path: protein coordinates
            aux_input_path: path to the pdb file of the protein
            output_folder: path of the output folder
            r: radius (Å)
            spacing: spacing between points along the radius (Å)
            shape: whether to return the shape of the binding grid only
        Output: dataframe of the binding grid, including coordinates and potentials for different atom types.
        """

        # build grid
        print("Generating pocket grid representation")
        pocket_grid = Grid3DBuilder.build(
            pdb_path=pdb_path,
            aux_input_path=aux_input_path,
            r=r,
            spacing=spacing,
            output_folder=output_folder,
            shape=shape,
        )

        # convert to numpy array
        print("Converting to numpy array")
        pocket_voxel = site_voxelization(pocket_grid, r, spacing, shape)
        pocket_voxel = np.expand_dims(pocket_voxel, axis=0)

        # save to h5 file
        pdb_filename = pdb_path.split("/")[-1].replace(".pdb", "")
        with h5py.File(f"{output_folder}/{pdb_filename}.h5", "w") as f:
            f.create_dataset(
                "X", data=pocket_voxel, compression="gzip", compression_opts=9
            )
        return pocket_voxel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f", type=str, required=True, help="Input pdb file name"
    )
    parser.add_argument(
        "--aux", "-a", type=str, required=True, help="Input auxilary file name"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output folder name"
    )
    parser.add_argument("--radius", "-r", type=int, required=True, help="Grid radius")
    parser.add_argument(
        "--spacing",
        "-n",
        type=float,
        required=True,
        help="The spacing between points along diameter",
    )
    parser.add_argument(
        "--shape",
        "-s",
        dest="shape",
        action="store_true",
        help="Return shape of grid only",
    )
    parser.add_argument("--potential", "-p", dest="shape", action="store_false")
    parser.set_defaults(shape=False)

    opt = parser.parse_args()
    n_points = int(2 * opt.radius / opt.spacing) + 1
    Vox3DBuilder().voxelization(
        pdb_path=opt.file,
        aux_input_path=opt.aux,
        r=opt.radius,
        spacing=opt.spacing,
        output_folder=opt.output,
        shape=opt.shape,
    )
