from time import time

import numpy as np
import pymesh

from ..thirdparty.deepdrug3d.build_grid import parase_pdb_aux
from ..thirdparty.deepdrug3d.write_aux_file import write_aux_file
from ..thirdparty.masif.source.default_config.masif_opts import masif_opts
from ..thirdparty.masif.source.input_output.protonate import protonate
from ..thirdparty.masif.source.triangulation import computeMSMS
from ..thirdparty.masif.source.triangulation.fixmesh import fix_mesh
from .visualization.visualization import save_grid_as_pdb
from .voxelization import site_voxelization


def voxelization_workflow(
    pdb_file_path,
    BS_string: str = None,
    prot_method: str = "propka",
    radius_for_grid: float = 15,
    spacing: float = 1.0,
    mesh_file_path: str = None,
    aux_file_path: str = None,
):
    """pdb_file_path: str or Path
    BS_string: str
    prot_method: str [propka, reduce]
    radius_for_grid: float (Å)
    spacing: float (Å)
    """
    time_start = time()

    # get the parent path
    pdbfilename = pdb_file_path.split("/")[-1].replace(".pdb", "")
    data_parent_path = pdb_file_path.split(f"/{pdbfilename}")[0]
    out_file_path = f"{data_parent_path}/{pdbfilename}_proton.pdb"

    # write the aux file
    assert (BS_string is not None) or (aux_file_path is not None), (
        "The binding site string is not provided!"
    )
    if aux_file_path is None:
        aux_file_path = f"{data_parent_path}/{pdbfilename}_proton_aux.txt"
        write_aux_file(aux_filepath=aux_file_path, binding_residue_ids=BS_string)
    else:
        aux_file_path = aux_file_path

    # Protonate the pdb file
    if mesh_file_path is None:
        print("Protonating the pdb file...")
        protonate(
            in_pdb_file=pdb_file_path,
            out_pdb_file=out_file_path,
            method=prot_method,
        )

    # read the new pdb file
    ppdb, protein_coords, pocket_center, pocket_coords = parase_pdb_aux(
        pdb_path=out_file_path,
        aux_input_path=aux_file_path,
        pass_transform_matrix=True,
        transform_coords=False,
    )

    if mesh_file_path is None:
        # compute surface by MSMS and fix mesh
        vertices1, faces1, normals1, names1, areas1 = computeMSMS.computeMSMS(
            out_file_path, protonate=True
        )
        mesh = pymesh.form_mesh(vertices=vertices1, faces=faces1)
        regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

        # set the interest region
        # compute the distance between each vertex and the pocket center
        distances = np.linalg.norm(regular_mesh.vertices - pocket_center, axis=1)
        interest_vertices_mask = distances <= radius_for_grid
        interest_vertices_bool = np.array(
            [1 if each else 0 for each in interest_vertices_mask]
        )
        regular_mesh.add_attribute("interest")
        regular_mesh.set_attribute("interest", interest_vertices_bool)
        print(f"Interest surface points: {sum(interest_vertices_bool)}")

        # save mesh
        mesh_file_path = f"{data_parent_path}/{pdbfilename}_proton_mesh.ply"
        pymesh.save_mesh(
            mesh_file_path,
            regular_mesh,
            *regular_mesh.get_attribute_names(),
            use_float=True,
            ascii=True,
        )

    else:
        regular_mesh = pymesh.load_mesh(mesh_file_path)
        # breakpoint()
        interest_vertices_mask = (
            regular_mesh.get_attribute("vertex_interest").flatten().astype(bool)
        )

    # voxelise the mesh
    site_mesh_vertices = regular_mesh.vertices[interest_vertices_mask] - pocket_center
    site_grid_voxel, grid_xyz_coords, grid2voxel_dist = site_voxelization(
        site_mesh_vertices,
        r=radius_for_grid,
        spacing=spacing,
        shape=True,
        pass_voxel_coord=True,
        log_info=False,
    )

    voxels = np.squeeze(site_grid_voxel, axis=0)
    print(site_grid_voxel.shape)

    time_end = time()
    print(f"Time cost: {time_end - time_start:.2f} seconds")
    print(f"{ppdb.df['ATOM'].__len__()} atoms in the original pdb file.")

    # save the voxel
    voxel_file_path = (
        f"{data_parent_path}/"
        f"{pdbfilename}_proton_voxel_{str(spacing).replace('.', '')}.npy"
    )
    np.save(voxel_file_path, voxels)

    # save the grid coordinates
    bool_filter = grid2voxel_dist < grid2voxel_dist.max()
    bool_filter = np.squeeze(bool_filter, axis=1)
    grid_w_values_xyz = grid_xyz_coords[bool_filter] + pocket_center

    grid_file_path = (
        f"{data_parent_path}/"
        f"{pdbfilename}_proton_grids_{str(spacing).replace('.', '')}.pdb"
    )
    save_grid_as_pdb(grid_file_path, grid_w_values_xyz)

    return None
