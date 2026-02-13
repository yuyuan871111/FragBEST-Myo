from pathlib import Path
from time import time

import pymesh

# import masif source code
from ..thirdparty.masif.source.default_config.masif_opts import masif_opts
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


def generate_masif_features(
    pdb_filepath: str | Path,
    ply_filepath: str | Path,
    logging: bool = False,
):
    """Generate MASIF features for a given PDB file.
    Input:
        - pdb_filepath: Path to the PDB file.
        - ply_filepath: Path to the output PLY file.
    Output:
        - Save the PLY file with the MASIF features.

    Reference running time for 12,524 atoms: 62.5 seconds
        [msms] Time elapsed: 2.3 seconds
        [compute features] Time elapsed: 2.7 seconds
        [fix mesh] Time elapsed: 10.7 seconds
        [assign charges] Time elapsed: 0.6 seconds
        [apbs] Time elapsed: 46.1 seconds
        [overall] Time elapsed: 62.5 seconds
    """
    # Check the file extension
    assert str(pdb_filepath).lower().endswith(".pdb"), (
        f"Invalid file extension: {pdb_filepath}. Must to be a PDB file."
    )

    if logging:
        print("Surface feature generation...")

    time_start = time()

    # Compute surface by MSMS, which is a Solvent Excluded Surfaces (SES)
    vertices1, faces1, normals1, names1, areas1 = computeMSMS.computeMSMS(
        pdb_filepath, protonate=True
    )

    # Compute "charged" vertices (hydrogen bonds)
    vertex_hbond = computeCharges(
        pdb_filename=pdb_filepath.replace(".pdb", ""), vertices=vertices1, names=names1
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
    vertex_charges = computeAPBS(
        vertices=regular_mesh.vertices,
        pdb_file=pdb_filepath,
        pdb2pqr_skip=False,
    )

    # Save the mesh.
    save_ply(
        ply_filepath,
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        # set to "True" to normalize charges: charges = charges / 10
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
    )

    time_end = time()
    if logging:
        print(f"[overall] Time elapsed: {time_end - time_start:.1f} seconds")
