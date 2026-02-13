# This file is derived from DeepDrug3D 
# Original copyright (c) 2019 Pu et al.

# Modifications:
# - Adapted for Python 3.10 compatibility 
# - Integrated into FragBEST-Myo workflow 
# - Modern code style and formatting

# Modified by Yu-Yuan (Stuart) Yang, 2024 
# Licensed under GPL-3.0 (see LICENSE)

import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy.spatial as sp
from biopandas.pdb import PandasPdb
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
    from potential import cal_DFIRE_potential, preparation_mol2_file
else:
    from .potential import cal_DFIRE_potential, preparation_mol2_file

"""
The following functions process the input pdb files by moving the center of pocket to [0,0,0]
and align the protein to the principle axes of the pocket
"""

near_atom_threshold = 2  # default: 2 Å
threshold_eps = 1.414


def cal_n_points(r: int, spacing: float):
    """
    Calculate the number of points along the diameter
    """
    N = int(2 * r / spacing) + 1
    return N


def normalize(v):
    """
    vector normalization
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vrrotvec(a, b):
    """
    Function to rotate one vector to another, inspired by
    vrrotvec.m in MATLAB
    """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a, b))
    angle = np.arccos(np.minimum(np.dot(a, b), [1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1, 3))
        c[mind] = 0
        ax = normalize(np.cross(a, c))
    r = np.concatenate((ax, angle))
    return r


def vrrotvec2mat(r):
    """
    Convert the axis-angle representation to the matrix representation of the
    rotation
    """
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = normalize(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]

    m = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    return m


def coords_transform(protein_coords, pocket_center, pocket_coords):
    """
    Transform the protein coordinates so that the pocket is centered at [0,0,0]
    and align the protein coordinates according to the principle axes of the pocket
    """
    pocket_coords = pocket_coords - pocket_center  # center the pocket to 0,0,0
    protein_coords = (
        protein_coords - pocket_center
    )  # center the protein according to the pocket center

    inertia = np.cov(pocket_coords.T)
    e_values, e_vectors = np.linalg.eig(inertia)
    sorted_index = np.argsort(e_values)[::-1]
    sorted_vectors = e_vectors[:, sorted_index]

    # Align the first principal axes to the X-axes
    rx = vrrotvec(np.array([1, 0, 0]), sorted_vectors[:, 0])
    mx = vrrotvec2mat(rx)
    pa1 = np.matmul(mx.T, sorted_vectors)

    # Align the second principal axes to the Y-axes
    ry = vrrotvec(np.array([0, 1, 0]), pa1[:, 1])
    my = vrrotvec2mat(ry)
    transformation_matrix = np.matmul(my.T, mx.T)
    # transform the protein coordinates to the center of the pocket and align with the principal
    # axes with the pocket
    transformed_coords = (np.matmul(transformation_matrix, protein_coords.T)).T
    return transformed_coords


"""
The following functions generate and refine the binding pocket grid
"""


def sGrid(center, r, N):
    """
    Generate spherical grid points at the center provided
    """
    center = np.array(center)
    x = np.linspace(center[0] - r, center[0] + r, N)
    y = np.linspace(center[1] - r, center[1] + r, N)
    z = np.linspace(center[2] - r, center[2] + r, N)

    # Generate grid of points
    X, Y, Z = np.meshgrid(x, y, z)
    data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    # indexing the interior points
    tree = sp.cKDTree(data)
    mask = tree.query_ball_point(center, 1.01 * r)
    points_in_sphere = data[mask]
    return points_in_sphere


def in_hull(p, hull):
    """
    Test if a point is inside a convex hull
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def site_refine(site, protein_coords):
    """
    Binding site refinement
    """
    # distance matrix for the removal of the grid points that are too close (<= 2 A) to any protein atoms
    dist = cdist(site[:, 0:3], protein_coords, "euclidean")
    inside_site = []
    on_site = []
    for i in range(len(dist)):
        if np.any(dist[i, :] <= near_atom_threshold):
            on_site.append(site[i, :])
        else:
            inside_site.append(site[i, :])
    on_site = np.array(on_site)
    inside_site = np.array(inside_site)

    # remove any grid points outside the convex hull
    in_bool = in_hull(inside_site[:, 0:3], protein_coords)
    hull_site = inside_site[in_bool]

    # remove isolated grid points
    iso_dist = cdist(hull_site[:, 0:3], hull_site[:, 0:3])
    labels = DBSCAN(eps=threshold_eps, min_samples=3, metric="precomputed").fit_predict(
        iso_dist
    )
    unique, count = np.unique(labels, return_counts=True)
    sorted_label = [x for _, x in sorted(zip(count, unique))]
    sorted_label = np.array(sorted_label)
    assert (len(sorted_label) != 1) | (
        sorted_label[0] != -1
    ), "No valid grid points by clustering methods. Please use smaller spacing."
    null_index = np.argwhere(sorted_label == -1)
    cluster_labels = np.delete(sorted_label, null_index)
    save_labels = np.flip(cluster_labels, axis=0)[0]
    final_label = np.zeros(labels.shape)
    for k in range(len(labels)):
        if labels[k] == save_labels:
            final_label[k] = 1
        else:
            continue
    final_label = np.array(final_label, dtype=bool)

    # potential energy normalization
    iso_site = hull_site[final_label]
    return iso_site, on_site


# replace the residue type with new residue type
def replace_res(original_string, new_res):
    temp = "{:6}".format(new_res)
    new_string = original_string.replace("VAL1  ", temp)
    return new_string


def read_aux_file(aux_input_path):
    """
    Read the auxilary input file
    return: resi, content
    """
    content = []
    with open(aux_input_path) as in_strm:
        for line in in_strm.readlines():
            line = line.replace("\n", "")
            # modified to ignore empty lines (Yu-Yuan Yang 2025)
            if line == "":
                continue
            idx = line.index(":")
            content.append(line[idx + 1 : None])
    resi = (
        # modified to ignore empty item due to the spaces (Yu-Yuan Yang 2025)
        # make it a list of strings (to accept chain IDs as well)
        [str(x) for x in content[0].split(" ") if x != ""]
        if content[0] != ""
        else []
    )
    return resi, content


def get_pocket_coords(protein_df: pd.DataFrame, resi: list):
    """
    Get the coordinates of the pocket
    """
    pocket_df = protein_df[protein_df["residue_number"].isin(resi)]
    pocket_coords = np.array(
        [pocket_df["x_coord"], pocket_df["y_coord"], pocket_df["z_coord"]]
    ).T
    return pocket_coords


def get_pocket_center(protein_df: pd.DataFrame, aux_input_path: str):
    """
    Get the center of the pocket
    """
    resi, content = read_aux_file(aux_input_path)
    if len(content[1]) != 0:
        # using provided center
        pocket_center = np.array([float(x) for x in content[1].split(" ")])
        print(
            "Center provided as {:.2f} {:.2f} {:.2f}".format(
                pocket_center[0], pocket_center[1], pocket_center[2]
            )
        )
    else:
        # calculate center by the residues
        print("No center is provided")
        pocket_coords = get_pocket_coords(protein_df, [int(i) for i in resi])
        pocket_center = np.mean(pocket_coords, axis=0)
        print(
            "Center calculated as {:.2f} {:.2f} {:.2f}".format(
                pocket_center[0], pocket_center[1], pocket_center[2]
            )
        )
    return pocket_center


def parase_pdb_aux(
    pdb_path, aux_input_path, pass_transform_matrix=False, transform_coords=True
):
    """
    Parse the pdb file and the auxilary input file
    Input:
        pdb_path: path to the pdb file
        aux_input_path: path to the auxilary input file
        pass_transform_matrix: whether to return the transformation matrix
        transform_coords: whether to transform the coordinates
    Output:
        ppdb: PandasPdb object
        transformed_coords: transformed protein coordinates
        pocket_center: center of the pocket (if pass_transform_matrix=True)
        pocket_coords: coordinates of the pocket (if pass_transform_matrix=True)
    """
    # read pdb
    ppdb = PandasPdb().read_pdb(pdb_path)
    protein_df = ppdb.df["ATOM"]

    # check whether the center of the pocket is provided
    resi, _ = read_aux_file(aux_input_path)
    pocket_center = get_pocket_center(protein_df, aux_input_path)
    pocket_coords = get_pocket_coords(protein_df, [int(i) for i in resi])

    # transform the coordinates or not
    protein_coords = np.array(
        [protein_df["x_coord"], protein_df["y_coord"], protein_df["z_coord"]]
    ).T
    if transform_coords:
        transformed_coords = coords_transform(
            protein_coords, pocket_center, pocket_coords
        )
    else:
        transformed_coords = protein_coords

    if pass_transform_matrix:
        return ppdb, transformed_coords, pocket_center, pocket_coords
    else:
        return ppdb, transformed_coords


# main function
class Grid3DBuilder(object):
    """Given an align protein, generate the binding grid
    and calculate the DFIRE potentials"""

    @staticmethod
    def build(
        pdb_path: str,
        aux_input_path: str,
        output_folder: str,
        ligand_path: str = None,
        r: int = 15,
        spacing: float = 1,
        shape: bool = False,
    ):
        """
        Input:
            pdb_path: protein coordinates
            aux_input_path: path to the pdb file of the protein
            output_folder: path of the output folder
            ligand_path: path to the pdb file of the ligand (transformed together)
            r: radius (Å)
            spacing: spacing between points along the radius (Å)
            shape: whether to return the shape of the binding grid only
        Output: dataframe of the binding grid, including coordinates and potentials for different atom types.
        """
        assert (
            spacing < threshold_eps
        ), f"Spacing {spacing} is too large, must be less than the DBSCAN eps threshold ({threshold_eps})."

        # Parse the pdb file and the auxilary input file
        pdb_filename = osp.basename(pdb_path).replace(".pdb", "")

        # The number for the grid points along the diameter
        n_points = cal_n_points(r, spacing)

        if ligand_path is not None:
            # Parse the ligand pdb file if provided
            ligand_filename = osp.basename(ligand_path).replace(".pdb", "")
            ppdb, transformed_coords, pocket_center, pocket_coords = parase_pdb_aux(
                pdb_path, aux_input_path, pass_transform_matrix=True
            )
            lpdb = PandasPdb().read_pdb(ligand_path)
            ligand_df = lpdb.df["HETATM"]
            ligand_coords = np.array(
                [ligand_df["x_coord"], ligand_df["y_coord"], ligand_df["z_coord"]]
            ).T
            ligand_transformed_coords = coords_transform(
                ligand_coords, pocket_center, pocket_coords
            )

        else:
            # parse the protein pdb file only
            ppdb, transformed_coords = parase_pdb_aux(pdb_path, aux_input_path)
            pdb_filename = osp.basename(pdb_path).replace(".pdb", "")

        # Generate a new pdb file with transformed coordinates
        ppdb.df["ATOM"]["x_coord"] = transformed_coords[:, 0]
        ppdb.df["ATOM"]["y_coord"] = transformed_coords[:, 1]
        ppdb.df["ATOM"]["z_coord"] = transformed_coords[:, 2]

        os.makedirs(output_folder, exist_ok=True)  # make output folder
        output_trans_pdb_path = osp.join(
            output_folder, pdb_filename + "_transformed.pdb"
        )
        print(f"Saving the binding pocket aligned pdb file to: {pdb_filename}.pdb")
        ppdb.to_pdb(output_trans_pdb_path)  # save the transformed pdb file

        if ligand_path is not None:
            # save the transformed ligand pdb file
            lpdb.df["HETATM"]["x_coord"] = ligand_transformed_coords[:, 0]
            lpdb.df["HETATM"]["y_coord"] = ligand_transformed_coords[:, 1]
            lpdb.df["HETATM"]["z_coord"] = ligand_transformed_coords[:, 2]
            lpdb.to_pdb(osp.join(output_folder, ligand_filename + "_transformed.pdb"))

        # Grid generation and DFIRE potential calculation
        print("The radius of the binding grid is: {}".format(r))
        print(f"The number of points along the diameter is: {n_points}")
        binding_site = sGrid(np.array([0, 0, 0]), r, n_points)
        new_site, protein_grid = site_refine(binding_site, transformed_coords)
        print(f"The number of points in the refined binding set is {len(new_site)}")
        print(f"The number of points on the protein is {len(protein_grid)}")

        if shape:
            # only return the shape of the binding grid
            print("Output only shape of the binidng grid")
            for total_potE, file_suffix in zip(
                [protein_grid, new_site], ["protein", "pocket"]
            ):
                df = pd.DataFrame(total_potE, columns=["x", "y", "z"])
                df.to_csv(
                    osp.join(output_folder, f"{pdb_filename}_{file_suffix}.grid"),
                    index=False,
                )
        else:
            # both return the shape and the DFIRE potentials

            # using opebabel's pybel to write mol2 file
            input_pdb_path = osp.join(output_folder, pdb_filename + "_transformed.pdb")
            preparation_mol2_file(
                input_pdb_path=input_pdb_path, output_folder=output_folder
            )

            # calculate the DFIRE potentials
            input_trans_mol2_path = osp.join(
                output_folder, pdb_filename + "_transformed.mol2"
            )
            df = cal_DFIRE_potential(
                new_site=new_site, input_trans_mol2_path=input_trans_mol2_path
            )
            df.to_csv(osp.join(output_folder, pdb_filename + ".grid"), index=False)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--pdb", type=str, required=True, help="Input pdb file name/path"
    )
    parser.add_argument(
        "-a", "--aux", type=str, required=True, help="Input auxilary file name/path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output folder name/path"
    )
    parser.add_argument("-l", "--ligand", type=str, help="Input ligand file name/path")
    parser.add_argument(
        "-r", "--radius", type=int, required=True, help="Grid radius (Å)"
    )
    parser.add_argument(
        "-n",
        "--spacing",
        type=float,
        required=True,
        help="The spacing between points along diameter",
    )
    parser.add_argument(
        "-s",
        "--shape",
        dest="shape",
        action="store_true",
        help="Return shape of grid only",
    )
    # parser.add_argument("-p", "--potential", dest="shape", action="store_false") # ligand-linux is not working for now.
    parser.set_defaults(shape=True, ligand=None)

    opt = parser.parse_args()
    Grid3DBuilder().build(
        pdb_path=opt.pdb,
        aux_input_path=opt.aux,
        r=opt.radius,
        spacing=opt.spacing,
        ligand_path=opt.ligand,
        output_folder=opt.output,
        shape=opt.shape,
    )
