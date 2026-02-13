import os
from time import time

import pymesh

# import masif source code
from ..thirdparty.masif.source.default_config.masif_opts import masif_opts
from ..thirdparty.masif.source.input_output.protonate import protonate
from ..thirdparty.masif.source.input_output.save_ply import save_ply
from ..thirdparty.masif.source.triangulation import computeMSMS
from ..thirdparty.masif.source.triangulation.compute_normal import compute_normal
from ..thirdparty.masif.source.triangulation.computeAPBS import computeAPBS
from ..thirdparty.masif.source.triangulation.computeCharges import (
    assignChargesToNewMesh,
    computeCharges,
)
from ..thirdparty.masif.source.triangulation.computeHydrophobicity import (
    computeHydrophobicity,
)
from ..thirdparty.masif.source.triangulation.fixmesh import fix_mesh


def test_masif():
    time_start = time()
    # display current workdir
    current_dir = os.getcwd()

    # set the protonation method
    prot_method = "propka"  # or set to "reduce"
    dataset_dir = f"{current_dir}/tests/test_data"

    # extract the first row of index_df as an example
    query = "8pnw"
    pdbfilename = "8pnwA"
    pdb_file_path = f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}.pdb"
    out_file_path = f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton.pdb"

    # Protonate the pdb file
    tmp_file_base = protonate(
        pdb_file_path, out_file_path, method=prot_method, keep_tempfiles=True
    )

    # Compute surface by MSMS, which is a Solvent Excluded Surfaces (SES)
    vertices1, faces1, normals1, names1, areas1 = computeMSMS.computeMSMS(
        out_file_path, protonate=True
    )

    # Compute "charged" vertices
    vertex_hbond = computeCharges(
        pdb_filename=out_file_path.replace(".pdb", ""), vertices=vertices1, names=names1
    )

    # For each surface residue, assign the hydrophobicity of its amino acid.
    vertex_hphobicity = computeHydrophobicity(names=names1)

    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices=vertices1, faces=faces1)
    regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

    # Compute the normals
    vertex_normal = compute_normal(
        vertex=regular_mesh.vertices, face=regular_mesh.faces
    )

    # Assign charges on new vertices based on charges of old vertices (nearest
    # neighbor)
    vertex_hbond = assignChargesToNewMesh(
        new_vertices=regular_mesh.vertices,
        old_vertices=vertices1,
        old_charges=vertex_hbond,
        seeder_opts=masif_opts,
    )
    vertex_hphobicity = assignChargesToNewMesh(
        new_vertices=regular_mesh.vertices,
        old_vertices=vertices1,
        old_charges=vertex_hphobicity,
        seeder_opts=masif_opts,
    )

    # Compute APBS charges
    pdb2pqr_skip = True if prot_method == "propka" else False
    vertex_charges = computeAPBS(
        vertices=regular_mesh.vertices,
        pdb_file=out_file_path,
        tmp_file_base=tmp_file_base,
        pdb2pqr_skip=pdb2pqr_skip,
        # pdb2pqr
        # set to "True" to skip pdb2pqr if using "propka" to protonate previously.
        # set to "False" if using "reduce" to protonate previously
    )
    os.system(f"rm -r {tmp_file_base}") if prot_method == "propka" else None

    # Save the mesh.
    save_ply(
        out_file_path.replace(".pdb", ".ply"),
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
    )

    time_end = time()
    print(f"Time elapsed: {time_end - time_start:.1f} seconds")

    # check if the output files are created
    mesh_with_feature_file_path = (
        f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton.ply"
    )
    assert os.path.exists(mesh_with_feature_file_path)

    # remove the output files
    test_files_path = f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton*"
    os.system(f"rm {test_files_path}")
