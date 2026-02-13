import json
import warnings
from pathlib import Path
from typing import Literal

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pymesh
from MDAnalysis.analysis import align
from natsort import natsorted

from ..ppseg.dataset import get_mask_col_idx, load_h5, preprocess_h5, write_h5
from ..ppseg.fragment import fragment_idx_label_dict
from ..ppseg.inference import inference
from ..ppseg.myo.default_config import LIGAND_FRAG_INFO_PATH
from ..ppseg.visualization.visualization import fragmentation_from_universe
from ..ppseg.voxelization import map_voxel_to_xyz, site_voxelization
from ..thirdparty.deepdrug3d.build_grid import read_aux_file
from ..thirdparty.deepdrug3d.write_aux_file import write_aux_file
from .feature_handler import generate_masif_features

_TYPES = Literal["complex", "protein", "ligand"]


def get_ligand_around_resids(
    u: mda.Universe,
    ligand_name: str,
    ligand_aa_dist: int,
    aa_existence_time: float = 0.50,
    with_segid: bool = False,
) -> list:
    """Get the residues around the ligand over a trajectory.

    Arguments:
        u (required): The input MDAnalysis Universe, representing the molecular system.
        ligand_name (required): The name of the ligand to find residues around.
        ligand_aa_dist: The distance (Å) from the ligand to consider residues as nearby.
        aa_existence_time: The fraction of the trajectory during which a residue must
            be present near the ligand to be included (default = 50%).
        with_segid: If ``True``, includes the segment ID (segid) in the returned
            residues.

    Returns:
        list: A list of the residues (include only resids, if `with_segid`
        is ``False``; include segids and resids, if `with_segid` is ``True``)
        around the ligand.

    """
    u.trajectory[0]  # set to frame 0

    frame_traj = []
    around_segresids_traj = []
    around_u = u.select_atoms(
        f"protein and (around {ligand_aa_dist} resname {ligand_name})", updating=True
    )

    # get the residues around the ligand
    for ts in u.trajectory:
        frame_traj.append(ts.frame)

        # update the around_u
        around_segresids = np.unique(
            around_u.segids + "_" + around_u.resids.astype(str)
        ).tolist()
        around_segresids_traj.append(around_segresids)

    # create a list to know all residues appearing in the trajectory
    all_avail_segresids = []
    for each in around_segresids_traj:
        all_avail_segresids.extend(each)
    all_avail_segresids = np.unique(np.array(all_avail_segresids))

    # intialise the one-hot encoding dictionary of the residues around the ligand
    one_hot_segresids_traj = {"template": {}}
    for each_segresid in all_avail_segresids:
        one_hot_segresids_traj["template"][each_segresid] = 0

    # create the one-hot encoding of the residues around the ligand
    for frame_num, each_frame in zip(frame_traj, around_segresids_traj):
        one_hot_segresids_traj[frame_num] = one_hot_segresids_traj["template"].copy()

        for each_segresid in each_frame:
            one_hot_segresids_traj[frame_num][each_segresid] = 1
    one_hot_segresids_traj_df = pd.DataFrame(one_hot_segresids_traj).T
    one_hot_segresids_traj_df.index.name = "Frame"
    one_hot_segresids_traj_df.drop("template", axis=0, inplace=True)

    # filter the residues that are present in at least 50% of the trajectory
    filter = (
        one_hot_segresids_traj_df.sum(axis=0) / len(one_hot_segresids_traj_df.index)
    ) >= aa_existence_time
    around_u_segresids = filter.index[filter.values].tolist()
    if with_segid:
        return around_u_segresids
    else:
        around_u_resids = [
            str(each.split("_")[1]) for each in around_u_segresids if "_" in each
        ]
        return around_u_resids


def get_resname_with_resid(u: mda.Universe, resids: list) -> list:
    """Get the residue name with the residue ID.

    Arguments:
        u: The input MDAnalysis Universe.
        resids: The residue IDs to get the residue names.

    Returns:
        list: A list of the residue names with the residue IDs.

    Example:

    .. code-block:: python

        from MDAnalysis import Universe
        u = Universe("example.pdb")
        resids = [1, 2, 3]
        resname_with_resid = get_resname_with_resid(u, resids)
        print(resname_with_resid)
        # Output:
        >>> ['ALA1', 'ARG2', 'GLU3']

    """
    resname_with_resid = []
    for each_resid in resids:
        sele_string = f"resid {each_resid}"
        sele_resid = u.select_atoms(sele_string)
        sele_aa = mda.lib.util.convert_aa_code(sele_resid.residues[0].resname)
        resname_with_resid.append(f"{sele_aa}{each_resid}")

    return resname_with_resid


# handling the complex trajectory
class TrajectoryHandler:
    """Trajectory handler for protein-ligand complex / protein-only trajectory.

    This class is used to handle the trajectory of a protein-ligand complex or
    protein-only trajectory. It provides methods to read the trajectory, get the
    residues at the pocket, get the pocket center, write the structure, features,
    labels, interest region, and voxelised data. It also provides methods to
    preprocess the data and write the auxiliary files.


    Arguments:
        top_path (required): the path to the topology file
            (.pdb, .gro, ... [MDAnalysis compatible])

        trajectory_path (required): the path to the trajectory file
            (.trr, .xtc, ... [MDAnalysis compatible])

        ligand_name (optional, recommended to provide): the name of the ligand

        radius_of_interest (optional): the radius (Å)
            to consider the interest region (default: ``16.0``)

        spacing (optional): the spacing (Å) between the grid points
            (default = ``0.5`` due to the sampling theorem
            from the mesh spacing 1Å)

        distance_cutoff (optional): the surface points will be labelled only
            if the distance of the point to the ligand's heavy atoms within
            this distance cutoff. (default: ``5.0`` Å)

        warning_check (optional): if ``True``, the warnings will be shown
            (default = ``True``)

    Returns:
        `self.top_path` was set to the top_path
        `self.trajectory_path` was set to the trajectory_path
        `self.ligand_name` was set to the ligand_name
        `self.universe` was set to the MDAnalysis Universe object
        `self.warning_check` was set to the warning_check


    .. note::
        Functions:
            high-level functions (can use self.variables and self.functions):
            low-level functions (can only use self.functions):

    """

    def __init__(
        self,
        top_path: str | Path,
        trajectory_path: str | Path = None,
        ligand_name: str = None,
        radius_of_interest: float = 16.0,
        spacing: float = 0.5,
        distance_cutoff: float = 5.0,
        warning_check: bool = True,
    ):
        """Initialize the TrajectoryHandler with the given parameters.

        Args:
            top_path (str | Path): Path to the topology file.
            trajectory_path (str | Path, optional): Path to the trajectory file.
                Defaults to None.
            ligand_name (str, optional): Name of the ligand. Defaults to None.
            radius_of_interest (float, optional): Radius to consider the interest
                region. Defaults to ``16.0``.
            spacing (float, optional): Spacing for the grid points. Defaults to ``0.5``.
            distance_cutoff (float, optional): Distance cutoff for labeling surface
                points. Defaults to ``5.0``.
            warning_check (bool, optional): Whether to show warnings. Defaults to
                ``True``.

        """
        self.top_path = top_path
        self.trajectory_path = trajectory_path
        self.ligand_name = ligand_name
        self.universe = (
            mda.Universe(top_path, trajectory_path)
            if trajectory_path is not None
            else mda.Universe(top_path)
        )
        self.warning_check = warning_check

        # optional
        self.set_config(
            radius_of_interest=radius_of_interest,
            spacing=spacing,
            distance_cutoff=distance_cutoff,
        )

        # check the ligand
        if self.warning_check:
            self._check_ligand()
        else:
            warnings.filterwarnings("ignore")

    def set_config(
        self,
        radius_of_interest: float = None,
        spacing: float = None,
        distance_cutoff: float = None,
    ):
        """Set the configuration (radius of interest, spacing, distance_cutoff), and
        detect whether the trajectory has multiple segids.

        Arguments:
            radius_of_interest: float (recommend = 16), the radius (Å) to
                consider the interest region
            spacing: float (recommend = 0.5), the spacing (Å) to
                consider the interest region
            distance_cutoff: float (Å) (recommend = 5), the surface points will be
                labelled only if the distance of the point to the ligand's heavy atoms
                within this distance cutoff.

        Returns:
            `self.radius_of_interest` was set to the radius_of_interest
            `self.spacing` was set to the spacing
            `self.distance_cutoff` was set to the distance_cutoff

        """
        if radius_of_interest is not None:
            self.radius_of_interest = radius_of_interest

        if spacing is not None:
            self.spacing = spacing

        if distance_cutoff is not None:
            self.distance_cutoff = distance_cutoff

        if len(np.unique(self.universe.residues.segids)) > 1:
            self.multi_segids = True
        else:
            self.multi_segids = False

    def get_frame(self, frame_number: int):
        """Get the frame of the trajectory.

        Arguments:
            frame_number: int, the frame number to get

        Returns:
            `self.universe.trajectory` was set to the frame_number

        """
        self.universe.trajectory[frame_number]

    def get_residues_at_pocket_by_center(self, pocket_center: list = None):
        """Get the residues at the pocket by the pocket center and the radius
        (`self.radius_of_interest`).

        Arguments:
            pocket_center: list, the pocket center. The default is `None`,
            which will attempt to use the pocket center stored in the trajectory
            handler.

        Returns:
            `self.residues_at_pocket` was set to the residues at the pocket
            `self.residues_at_pocket_str` was set to the residues at the pocket
            in string format

        """
        if pocket_center is None:
            pocket_center = self.pocket_center

        pocket_residues = self.universe.select_atoms(
            f"protein and byres point {self.__list2str(pocket_center)} "
            f"{self.distance_cutoff}",
        )

        if self.multi_segids:
            self.residues_at_pocket = [
                f"{segid}_{resid}"
                for segid, resid in zip(
                    pocket_residues.residues.segids,
                    pocket_residues.residues.resids,
                )
            ]
        else:
            self.residues_at_pocket = [
                str(each) for each in pocket_residues.residues.resids
            ]

        self.residues_at_pocket_str = self.__list2str(self.residues_at_pocket)

    def get_residues_at_pocket(
        self, ligand_aa_dist: int = 5, aa_existence_time: float = 0.5
    ):
        """[Require ligand name] Get the resnames of the anchored residues
        at the pocket over a trajectory.

        Arguments:
            ligand_aa_dist: int, the distance (Å) from the ligand to
                consider the residues.
            aa_existence_time: float, the fraction of the trajectory that
                the residue should be present to be considered.

        Returns:
            `self.residues_at_pocket` was set to the residues at the pocket
            `self.residues_at_pocket_str` was set to the residues at the pocket
            in string format

        """
        self.ligand_aa_dist = ligand_aa_dist
        self.aa_existence_time = aa_existence_time

        # sanity check
        if self.ligand_name is None:
            raise ValueError("ligand_name is not provided.")

        # get the residues at the pocket
        self.residues_at_pocket = get_ligand_around_resids(
            u=self.universe,
            ligand_name=self.ligand_name,
            ligand_aa_dist=ligand_aa_dist,
            aa_existence_time=aa_existence_time,
            with_segid=True if self.multi_segids else False,
        )
        self.residues_at_pocket_str = self.__list2str(self.residues_at_pocket)

    def get_pocket_center(self, frame: int = 0):
        """[Require ligand name] Get the pocket center at a specific frame
        (default = 0).

        Arguments:
            frame: int, the frame number to get the pocket center

        Returns:
            self.pocket_center was set to the pocket center
            self.pocket_center_str was set to the pocket center in string format

        .. note::
            Deprecated the deepdrug3d version to calculate the pocket center.
            Instead, use mdanalysis to calculate the center of geometry.

        """
        # requirement: get_pocket_residues
        if not hasattr(self, "pocket_residues"):
            self.get_pocket_residues()

        self.get_frame(frame)
        self.pocket_center = self.pocket_residues.center_of_geometry().tolist()
        self.pocket_center_str = self.__list2str(self.pocket_center)

        # reset the universe
        self.get_frame(0)

    def get_protein(self):
        """Get the protein from the trajectory by MDAnalysis selection.

        Returns:
            `self.protein` was set to the protein

        """
        self.protein = self.universe.select_atoms("protein", updating=True)

    def get_ligand(self):
        """[Require ligand name] Get the ligand from the trajectory
        by MDAnalysis selection.

        Returns:
            `self.ligand` was set to the ligand

        """
        self.ligand = self.universe.select_atoms(
            f"resname {self.ligand_name}", updating=True
        )

    def get_complex(self):
        """[Require ligand name] Get the complex from the trajectory
        (inlcuding, protein, ligand, protein + ligand) by MDAnalysis selection.

        Returns:
            `self.ligand` was set to the ligand
            `self.protein` was set to the protein
            `self.complex` was set to the complex (protein + ligand)

        """
        self.get_protein()
        self.get_ligand()
        self.complex = self.universe.select_atoms(
            f"protein or resname {self.ligand_name}", updating=True
        )

    def get_pocket_residues(self):
        """[Require `self.residues_at_pocket`] Get the residues at the pocket from
        the trajectory by `MDAnalysis` selection.

        Returns:
            `self.pocket_residues` (`MDAnlysiis` AtomGroup) was set to the residues
                at the pocket

        """
        # requirement: get_residues_at_pocket
        if not hasattr(self, "residues_at_pocket"):
            self.get_residues_at_pocket()

        assert self.residues_at_pocket != [], "residues_at_pocket is empty"

        pocket_residue_str = self.__resid_for_selection(
            self.residues_at_pocket, self.multi_segids
        )
        self.pocket_residues = self.universe.select_atoms(
            f"protein and ({pocket_residue_str})", updating=True
        )

    def read_pocket_from_string(
        self,
        residues_at_pocket_str: str = None,
        pocket_center_str: str = None,
    ):
        """Read the residues at the pocket and the pocket center from strings.

        Arguments:
            residues_at_pocket_str: str, the residues at the pocket in string format
                (default: None)
            pocket_center_str: str, the pocket center in string format
                (default: None)

        Returns:
            `self.residues_at_pocket` was set to the residues at the pocket
            `self.residues_at_pocket_str` was set to the residues at the pocket
                in string format
            `self.pocket_center` was set to the pocket center
            `self.pocket_center_str` was set to the pocket center in string format
        """
        if residues_at_pocket_str:
            self.residues_at_pocket_str = residues_at_pocket_str.strip()
            self.residues_at_pocket = [
                each for each in residues_at_pocket_str.split(" ") if each != ""
            ]

        if pocket_center_str:
            self.pocket_center_str = pocket_center_str.strip()
            pocket_center = [
                float(each) for each in self.pocket_center_str.split(" ") if each != ""
            ]
            assert len(pocket_center) == 3, (
                "The pocket center is not providely correctly. "
                "Please provide the pocket center in the format 'x y z'."
            )
            self.pocket_center = pocket_center

        self.get_pocket_residues()

    def read_pocket_aux_file(self, aux_file_path: str | Path):
        """Read the residues at the pocket and the pocket center from an auxiliary file.

        Arguments:
            aux_file_path: str, the path to the auxiliary file

        Returns:
            `self.residues_at_pocket` was set to the residues at the pocket
            `self.residues_at_pocket_str` was set to the residues at the pocket
                in string format
            `self.pocket_center` was set to the pocket center
            `self.pocket_center_str` was set to the pocket center in string format

        """
        self.residues_at_pocket, content = read_aux_file(aux_file_path)

        assert content[0].replace(" ", "") != "" or content[1].replace(" ", "") != "", (
            "Both residues IDs and pocket center are not provided."
        )

        # residues in the pocket
        if self.residues_at_pocket == []:
            # if the residues at the pocket are not provided
            self.get_residues_at_pocket_by_center(
                pocket_center=[
                    float(each) for each in content[1].split(" ") if each != ""
                ]
            )
        else:
            self.residues_at_pocket_str = content[0]

        if content[1].replace(" ", "") == "":
            # if pocket center is not provided, calculate it
            self.get_pocket_center()
        else:
            self.pocket_center_str = content[1]
            self.pocket_center = [
                float(each) for each in self.pocket_center_str.split(" ") if each != ""
            ]
            assert len(self.pocket_center) == 3, (
                "The pocket center is not providely correctly. "
                "Please provide the pocket center in the format 'x y z'."
            )
        self.get_pocket_residues()

    def read_fragment_aux_file(self, aux_file_path: str | Path = None):
        """Read the fragments from an auxiliary file.

        Arguments:
            aux_file_path: str, the path to the auxiliary file
                (format: json), if not provided, use the default example file

        Returns:
            `self.labels_info` was set to the fragments

        Example:

        .. code-block:: python

            from ProBiSEnSe.utils.datasets.traj_handler import TrajectoryHandler
            traj_handler = TrajectoryHandler(
                top_path="example.pdb",
                trajectory_path="example.xtc",
                ligand_name="LIG",
            )
            aux_file_path = "example.json"
            traj_handler.read_fragment_aux_file(aux_file_path)
            print(traj_handler.label_fragment_info)
            # Output:
            >>> {
                    "0": {
                    "name": "out of the threshold"
                    },
                    "1": {
                        "name": "fragment 1",
                        "fragments_idx": [0, 1, 2, 3, 30, 45, 52]
                    },
                    "2": {
                        "name": "fragment 2",
                        "fragments_idx":
                        [4, 5, 6, 7, 26, 27, 29, 31, 32, 43, 44, 46, 47, 50]
                }

        """
        if aux_file_path is None:
            aux_file_path = LIGAND_FRAG_INFO_PATH
        with open(aux_file_path) as f:
            self.label_fragment_info = json.load(f)

        self.fragidx_label_dict = fragment_idx_label_dict(
            labels_info=self.label_fragment_info
        )

    def align_traj_to_pocket(
        self,
        reference: mda.Universe | mda.AtomGroup | int = None,
        select_Hs: bool = False,
        update_pocket_center: bool = True,
    ):
        """Use the pocket resids to align the trajectory to the pocket.

        Requirement: `residues_at_pocket`.

        Arguments:
            reference: MDAnalysis Universe object, AtomGroup object, or int,
                the reference to align the trajectory
            select_Hs: bool, if ``True``, the H atoms will be selected
            update_pocket_center: bool, if ``True``, the pocket center will be
                updated after the alignment (default: ``True``)


        Returns:
            Align the trajectory to the pocket. See `self.universe`,
                it will be updated.

        """
        around_u_resids = self.__resid_for_selection(
            resid_list=self.residues_at_pocket,
            with_segid=self.multi_segids,
        )

        # ignore the H atoms when aligning the structure
        if not select_Hs:
            # exclude hydrogens (some H name like 1HD1)
            around_u_resids = (
                f"{around_u_resids} and "
                "(not ((name *H* and not name N* and not name O*) or (type H)))"
            )

        # align the trajectory to the pocket
        self.get_frame(0)  # set to frame 0

        if reference is None:
            alignment = align.AlignTraj(
                mobile=self.universe,
                reference=self.universe,
                select=around_u_resids,
                ref_frame=0,
                in_memory=True,
            )
        elif isinstance(reference, int):
            alignment = align.AlignTraj(
                mobile=self.universe,
                reference=self.universe,
                select=around_u_resids,
                ref_frame=reference,
                in_memory=True,
            )
        else:
            alignment = align.AlignTraj(
                mobile=self.universe,
                reference=reference,
                select=around_u_resids,
                in_memory=True,
            )
        alignment.run()

        # update the pocket center
        if update_pocket_center:
            self.get_pocket_center()
            if self.warning_check:
                print(
                    "The pocket center has been updated to "
                    f"{self.pocket_center}"
                    " after the alignment."
                )

    def preprocess_workflow(
        self,
        pdb_path: str | Path,
        ply_path: str | Path,
        h5_path: str | Path,
        frame: int = 0,
        with_label: bool = True,
    ):
        """Preprocessing workflow for a frame, including writing the structure,
        features, labels, interest region, and voxelised data.

        Arguments:
            pdb_path: str, the path to the PDB file
            ply_path: str, the path to the PLY file
            h5_path: str, the path to the H5 file
            frame: int, the frame number to get the features
            with_label: bool, if ``True``, the labels will be included in the H5 file

        Returns:
            Save the PDB file in `pdb_path`
            Save the PLY file with the MASIF features in `ply_path`
            Save the H5 file in `h5_path` with ['raw'] or ['raw' and 'label']
            (if `with_label` is ``True``)

            .. note::
                - `raw`: the voxelised data of the features
                - `label: the voxelised data of the labels (if `with_label` is ``True``)

        """
        # check
        assert self.pocket_center is not None, "pocket_center is not provided"

        # write the structure
        self.write_structure(pdb_path=pdb_path, frame=frame)
        if self.warning_check:
            print(f"Writing the PDB file: {frame} completed")

        # write the features
        self.write_features_to_ply(pdb_path=pdb_path, ply_path=ply_path, frame=frame)
        self.add_interest_region_to_ply(ply_path=ply_path)
        if with_label:
            self.add_labels_to_ply(
                ply_path=ply_path,
                ref_ligand_frame=frame,
            )
        if self.warning_check:
            print(
                "Writing the PLY file with the features"
                f"{' and labels' if with_label else ''}: {frame} completed"
            )

        # write the h5 file
        self.write_voxelised_data_to_h5(
            ply_path=ply_path,
            h5_path=h5_path,
            with_label=with_label,
        )
        if self.warning_check:
            print(f"Writing the h5 file: {frame} completed")

    def write_pocket_aux_file(self, aux_file_path: str | Path):
        """Write the residues at the pocket and the pocket center to an auxiliary file.

        Arguments:
            aux_file_path: str, the path to the auxiliary file

        Returns:
            Save the auxiliary file in `aux_file_path`

        """
        # requirement: get_residues_at_pocket
        if not hasattr(self, "residues_at_pocket"):
            self.get_residues_at_pocket()

        if not hasattr(self, "pocket_center"):
            self.get_pocket_center()

        write_aux_file(
            aux_filepath=aux_file_path,
            binding_residue_ids=self.residues_at_pocket_str,
            binding_site_center=self.pocket_center_str,
        )

    def write_trajectory(
        self,
        traj_path: str | Path,
        start_frame: int = 0,
        end_frame: int = None,
        structure_type: _TYPES = None,
        step: int = 1,
    ):
        """Write the trajectory as a traj file.

        Arguments:
            traj_path: str, the path to the trajectory file
            start_frame: int, the frame number to start
            end_frame: int, the frame number to end. If ``None``,
                it will be the total number of frames.
            structure_type: str, the type of structure to write
                (complex, protein, ligand). If ``None``, it will be all atoms.
            step: int, the step to write the frames

        Returns:
            Save the trajectory file in `traj_path`.

        """
        # initialisation
        if end_frame is None:
            end_frame = len(self.universe.trajectory)

        # check
        assert end_frame <= len(self.universe.trajectory), (
            "end_frame should be less than the total number of frames."
        )
        if structure_type is not None:
            assert hasattr(self, structure_type), (
                f"{structure_type} is not provided. Please run `get_complex()` first."
            )

        # write the trajectory
        atoms_to_save = (
            self.universe.select_atoms("all", updating=True)
            if structure_type is None
            else getattr(self, structure_type)
        )
        with mda.Writer(traj_path, atoms_to_save.n_atoms) as w:
            for ts in self.universe.trajectory[start_frame:end_frame:step]:
                w.write(atoms_to_save)

        # reset the universe
        self.get_frame(0)

    def write_structure(
        self,
        pdb_path: str | Path,
        frame: int,
        structure_type: _TYPES = "protein",
        fragmentation: bool = False,
    ):
        """Write the structure as a PDB file for a specific frame.

        Arguments:
            pdb_path: str, the path to the PDB file
            frame: int, the frame number to get the structure
            structure_type: str, the type of structure to write
                (complex, protein, ligand)
            fragmentation: bool, if ``True``, the structure will be fragmented

        Returns:
            Save the PDB file in `pdb_path`

        """
        # requirement: get_complex
        if not hasattr(self, structure_type):
            self.get_complex()

        self.get_frame(frame)
        if fragmentation and structure_type != "protein":
            self._fragment_universe()

        with mda.Writer(pdb_path, getattr(self, structure_type).n_atoms) as W:
            W.write(getattr(self, structure_type))

        # reset the universe
        self.get_frame(0)

    def write_features_to_ply(
        self,
        pdb_path: str | Path,
        ply_path: str | Path,
        frame: int = None,
    ):
        """Write the MASIF features to a PLY file. If the PDB file does
        not exist, it will be created from the trajectory.

        Arguments:
            pdb_path: str, the path to the PDB file
            ply_path: str, the path to the PLY file
            frame: int, the frame number to get the features

        Returns:
            Save the PLY file with the MASIF features in `ply_path`

        """
        # check
        if not Path(pdb_path).is_file() and frame is not None:
            if self.warning_check:
                print(
                    f"The PDB file does not exist. Creating the PDB file from "
                    f"the trajectory (frame {frame})..."
                )
            self.write_structure(pdb_path=pdb_path, frame=frame)

        elif not Path(pdb_path).is_file() and frame is None:
            raise ValueError(
                "The PDB file does not exist. If providing the frame number, "
                "an PDB file will be created from the trajectory."
            )
        else:
            if self.warning_check:
                print(f"Using the PDB file: {pdb_path}")

        # workflow of generating masif features
        generate_masif_features(pdb_path, ply_path)

    def write_voxelised_data_to_h5(
        self,
        ply_path: str | Path,
        h5_path: str | Path,
        with_label: bool = True,
    ):
        """Write the surface vertices into voxelised data and save in an H5 file.

        Arguments:
            ply_path: str, the path to the input PLY file
            h5_path: str, the path to the outpu H5 file
            with_label: bool, if ``True``, the labels will be included in the H5 file

        Returns:
            Save the H5 file in `h5_path` with ['raw'] or ['raw' and 'label']
            (if `with_label` is ``True``)

            .. note::
                - `raw`: the voxelised data of the features
                - `label`: the voxelised data of the labels \
                    (if `with_label` is ``True``)

        """
        # check
        assert Path(ply_path).is_file(), "The PLY file does not exist."
        assert str(h5_path).endswith(".h5"), (
            "The H5 file should have the extension '.h5'."
        )
        self._check_config()

        # read mesh file
        regular_mesh = pymesh.load_mesh(ply_path)

        # voxelisation
        voxel_features, voxel_labels = self._voxelisation(
            mesh=regular_mesh, with_label=with_label
        )

        # write h5 file
        if with_label:
            write_h5(data=voxel_features, h5_filename=h5_path, label=voxel_labels)
        else:
            write_h5(data=voxel_features, h5_filename=h5_path, label=None)

    def add_labels_to_ply(
        self,
        ply_path: str | Path,
        ref_ligand_frame: int,
        ply_path_output: str | Path = None,
    ):
        """Add the labels to a PLY file.

        Arguments:
            ply_path: str, the path to the PLY file
            ref_ligand_frame: int, the frame number to get the reference
                ligand for the surface.
            ply_path_output: str, the path to the output PLY file

        Returns:
            Save the PLY file with the labels in `ply_path_output`.
            If the `ply_path_output` is not provided, otherwise in `ply_path`.

        """
        # check
        if not hasattr(self, "ligand"):
            self.get_complex()
        assert self.distance_cutoff is not None, "distance_cutoff is not provided"
        assert hasattr(self, "label_fragment_info"), (
            "label_fragment_info is not provided, please read the "
            "fragment auxiliary file first (using `read_fragment_aux_file`)."
        )

        # set output path
        if ply_path_output is None:
            ply_path_output = ply_path

        # load the mesh
        regular_mesh = pymesh.load_mesh(ply_path)
        mesh_vertices_coords = regular_mesh.vertices

        # load ligand atoms
        ligand_heavy_atoms = self.ligand.select_atoms("(not name H*)", updating=True)
        self.get_frame(ref_ligand_frame)
        ligand_coords = self.ligand.atoms.positions
        ligand_atoms_idxs = self.ligand.atoms.indices  # get the ligand atoms indices
        ligand_heavy_atoms_idxs = [
            int(np.where(ligand_atoms_idxs == each)[0])
            for each in ligand_heavy_atoms.atoms.indices
        ]

        # generate the labels
        labels = self.__generate_label_for_each_vertex_with_fragment(
            mesh_coords=mesh_vertices_coords,
            ligand_coords=ligand_coords,
            ligand_heavy_atoms_idxs=ligand_heavy_atoms_idxs,
            fragidx_label_dict=self.fragidx_label_dict,
            distance_cutoff=self.distance_cutoff,
        )

        # add attribute and save mesh
        regular_mesh.add_attribute("vertex_label")
        regular_mesh.set_attribute("vertex_label", labels)
        self.__save_mesh(ply_path_output, regular_mesh)

        # reset the universe
        self.get_frame(0)

    def add_interest_region_to_ply(
        self,
        ply_path: str | Path,
        ply_path_output: str | Path = None,
    ):
        """Add the interest region to a PLY file.

        Arguments:
            ply_path: str, the path to the input PLY file.
            ply_path_output: str, the path to the output PLY file. If not provided,
                the input PLY file will be overwritten.

        Returns:
            Save the PLY file with the interest region in `ply_path_output`.
            If the `ply_path_output` is not provided, otherwise in `ply_path`.

        """
        # set output path
        if ply_path_output is None:
            ply_path_output = ply_path

        # load the mesh
        regular_mesh = pymesh.load_mesh(ply_path)
        mesh_vertices_coords = regular_mesh.vertices

        # compute the distance between each vertex and the pocket center
        distances = np.linalg.norm(mesh_vertices_coords - self.pocket_center, axis=1)
        interest_vertices_bool = np.array(
            [1 if each else 0 for each in (distances <= self.radius_of_interest)]
        )

        # add attribute and save mesh
        regular_mesh.add_attribute("vertex_interest")
        regular_mesh.set_attribute("vertex_interest", interest_vertices_bool)
        self.__save_mesh(ply_path, regular_mesh)

    def add_prediction_to_ply(  # noqa: D102
        self,
        ply_path: str | Path,
        h5_path: str | Path,
        model,
        ply_path_output: str | Path = None,
        device: str = "cpu",
    ):
        # set output path
        if ply_path_output is None:
            ply_path_output = ply_path

        # load the mesh
        regular_mesh = pymesh.load_mesh(ply_path)

        # prediction
        _, _, _, pred, probs, mask = self.__predict(h5_path, model, device)

        # map the voxel to the xyz
        preds_on_vertex, probs_on_vertex = self._map_voxel_to_vertices(
            voxel_pred=pred,
            voxel_probs=probs,
            voxel_mask=mask,
            regular_mesh=regular_mesh,
        )

        # add attribute and save mesh
        regular_mesh.add_attribute("vertex_pred")
        regular_mesh.set_attribute("vertex_pred", preds_on_vertex)

        regular_mesh.add_attribute("vertex_predprobs")
        regular_mesh.set_attribute("vertex_predprobs", probs_on_vertex)
        self.__save_mesh(ply_path_output, regular_mesh)

    def _check_config(self):
        if self.radius_of_interest is None:
            raise ValueError("radius_of_interest is not provided")
        elif self.spacing is None:
            raise ValueError("spacing is not provided")
        else:
            assert self.radius_of_interest / self.spacing < 100, (
                "The r/spacing should be less than 100 to avoid computer freezing."
            )

    def _check_ligand(self):
        if self.ligand_name is None:
            print("ligand_name is not provided")
        else:
            if (
                self.ligand_name
                not in self.universe.select_atoms("not protein").residues.resnames
            ):
                raise ValueError(f"{self.ligand_name} is not in the trajectory")
            else:
                print(f"{self.ligand_name} is in the trajectory")

    def _get_attribute(self):
        return {
            each_object: type(getattr(self, each_object))
            for each_object in self.__dict__.keys()
        }

    def _voxelisation(self, mesh: pymesh.Mesh, with_label: bool = True):
        # check
        if not hasattr(self, "pocket_center"):
            self.get_pocket_center()

        # normlaise with the pocket center
        mesh_vertices = mesh.vertices - self.pocket_center

        # get the features from the mesh file
        apbs_charge = mesh.get_attribute("vertex_charge")
        hbond = mesh.get_attribute("vertex_hbond")
        hphob = mesh.get_attribute("vertex_hphob")
        occupancy = mesh.get_attribute("vertex_interest")
        features = np.column_stack((apbs_charge, hbond, hphob, occupancy))
        channel = features.shape[1]

        # get the label from the mesh file
        label = mesh.get_attribute("vertex_label") if with_label else None

        # merge the vertices, features, and labels (for voxelisation mapping)
        mesh_data = (
            np.column_stack((mesh_vertices, features, label))
            if with_label
            else np.column_stack((mesh_vertices, features))
        )

        interest_vetices_mask = (occupancy == 1).astype(bool)
        mesh_data = mesh_data[interest_vetices_mask]

        # voxelisation
        log_info = True if self.warning_check else False
        grid_voxel = site_voxelization(
            site=mesh_data,
            r=self.radius_of_interest,
            spacing=self.spacing,
            shape=False,
            pass_voxel_coord=False,
            log_info=log_info,
        )
        return grid_voxel[0:channel, :, :, :], (
            grid_voxel[channel, :, :, :] if with_label else None
        )

    def _fragment_universe(self):
        mol_f_idx_keys = natsorted(self.label_fragment_info.keys())
        mol_f_idx_keys.remove("0")  # ignore the background label
        mol_f_idx = [
            self.label_fragment_info[each]["fragments_idx"] for each in mol_f_idx_keys
        ]
        self.universe = fragmentation_from_universe(
            universe=self.universe,
            ligand_name=self.ligand_name,
            mol_f_idx=mol_f_idx,
            using_type="fragment_idx",
        )
        return None

    def _map_voxel_to_vertices(
        self,
        voxel_pred: np.array,
        voxel_probs: np.array,
        voxel_mask: np.array,
        regular_mesh: pymesh.Mesh,
    ):
        # map voxel to xyz
        voxel_pred_probs = np.vstack([voxel_pred, voxel_probs])
        voxel_xyz_pred_probs = map_voxel_to_xyz(
            data=voxel_pred_probs,
            r=self.radius_of_interest,
            spacing=self.spacing,
            pocket_center=self.pocket_center,
            mask=voxel_mask,
        )
        voxel_xyz = voxel_xyz_pred_probs[:, 0:3]
        voxel_pred = voxel_xyz_pred_probs[:, 3]
        voxel_probs = voxel_xyz_pred_probs[:, 4]

        # map to vertices
        mesh_vertices_coords = regular_mesh.vertices
        vertex_interest = regular_mesh.get_attribute("vertex_interest")

        preds_on_vertex = []
        probs_on_vertex = []
        for idx, each_interest in enumerate(vertex_interest):
            if each_interest == 1:
                vertex_xyz = mesh_vertices_coords[idx]

                # find the cloest point in test_xyz
                dist = np.linalg.norm(voxel_xyz - vertex_xyz, axis=1)
                min_idx = np.argmin(dist)
                preds_on_vertex.append(voxel_pred[min_idx])
                probs_on_vertex.append(voxel_probs[min_idx])
            else:
                preds_on_vertex.append(0)
                probs_on_vertex.append(0)

        preds_on_vertex = np.array(preds_on_vertex)
        probs_on_vertex = np.array(probs_on_vertex, dtype=float)

        return preds_on_vertex, probs_on_vertex

    # basic functions (without using self.variables)
    @staticmethod
    def __list2str(list_input: list, sep: str = " "):
        return sep.join([str(each) for each in list_input])

    @staticmethod
    def __resid_for_selection(resid_list: list, with_segid: bool = False):
        if with_segid:
            return " or ".join(
                [
                    f"(segid {each.split('_')[0]} and resid {each.split('_')[1]})"
                    for each in resid_list
                ]
            )
        else:
            return " or ".join([f"resid {each}" for each in resid_list])

    @staticmethod
    def __save_mesh(ply_path: str | Path, mesh: pymesh.Mesh):
        pymesh.save_mesh(
            ply_path,
            mesh,
            *[
                each
                for each in mesh.get_attribute_names()
                if each
                not in ["vertex_x", "vertex_y", "vertex_z", "face_vertex_indices"]
            ],
            use_float=True,
            ascii=True,
        )

    @staticmethod
    def __generate_label_for_each_vertex_with_fragment(
        mesh_coords: np.array,
        ligand_coords: np.array,
        ligand_heavy_atoms_idxs: list[int],
        fragidx_label_dict: dict,
        distance_cutoff: float = 5.0,
    ):
        labels = []
        for vertex in mesh_coords:
            # compute the distance between each vertex and the heavy atoms to
            # make sure it is in the threshold
            distances = np.linalg.norm(ligand_coords - vertex, axis=1)
            distances_heavy_atoms = distances[ligand_heavy_atoms_idxs]
            if np.min(distances_heavy_atoms) <= distance_cutoff:
                min_distance_idx = ligand_heavy_atoms_idxs[
                    np.argmin(distances_heavy_atoms)
                ]
                # In case of multiple occurrences of the minimum values,
                # the indices corresponding to the first occurrence are returned.
                labels.append(fragidx_label_dict[min_distance_idx])
            else:
                labels.append(0)

        return np.array(labels, dtype=float)

    @staticmethod
    def __predict(
        h5_path: str | Path,
        model,
        device: str = "cpu",
    ):
        """Predict the labels from the H5 file.

        Arguments:
            h5_path: str, the path to the H5 file
            model: the model to predict
            device: str, the device to run the model

        Returns:
            tuple:
                - `data`: the input data
                - `_label_if_exist`: the labels if exist else return None
                - `outputs`: the outputs from the model
                - `pred`: the predictions
                - `probs`: the probabilities
                - `mask`: the mask

        """
        # load data
        data, _label_if_exist = load_h5(h5_path)
        mask_col_idx = get_mask_col_idx(data)
        data, _label_if_exist = preprocess_h5(
            data, _label_if_exist, except_channels=[mask_col_idx]
        )
        # data: [C, H, W, D]

        # inference (prediction)
        outputs, pred, probs, mask = inference(model, data, device, return_mask=True)
        # outputs: [C, H, W, D]; predict, probs, mask: [H, W, D]
        pred = np.expand_dims(pred, axis=0)
        probs = np.expand_dims(probs, axis=0)
        # pred, probs: [1, H, W, D]

        return data, _label_if_exist, outputs, pred, probs, mask
