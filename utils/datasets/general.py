import logging
import os

import torch
from ignite.utils import setup_logger
from natsort import natsorted

# import self defined functions
from ..parallel.framework import TrajHandlerPrediction, TrajHandlerPreprocess
from ..ppseg.holo_descriptor.holo_descriptor import HoloDescriptor
from ..ppseg.ignite.utils import save_config
from ..thirdparty.unet3d_model.unet3d import UnetModel
from .traj_handler import TrajectoryHandler


################################################
# Data Preparation
def preprocess_workflow(
    traj_handler: TrajectoryHandler,
    pdb_path,
    ply_path,
    h5_path,
    frame,
    index_path,
    logger=None,
    with_label=False,
):
    """Preprocess workflow for each frame

    Args:
        traj_handler: TrajectoryHandler object
        pdb_path: path to save the pdb file
        ply_path: path to save the ply file
        h5_path: path to save the h5 file
        frame: frame index
        index_path: path to the index file
        logger: logger object
            If ``None``, no logging will be printed (default: ``None``).
        with_label: bool
            If ``True``, the label will be added to the h5 file (default: ``False``).
            Only used for holo conformation.
    """
    traj_handler.preprocess_workflow(
        pdb_path=pdb_path,
        ply_path=ply_path,
        h5_path=h5_path,
        frame=frame,
        with_label=with_label,
    )

    # write the index file
    with open(index_path, "a") as f:
        f.write(f"{os.path.basename(pdb_path)}\n")

    # print out finished!
    if logger:
        logger.info(f"Preprocess: {frame} done")


# apo conformation
def apo_data_preparation_recipe(config):
    """Data preparation recipe for apo conformation"""
    # logger setting
    logger = setup_logger(
        name=f"\033[32m{config.mode}\033[0m",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=f"{config.output_path}/{config.mode}-info.log",
    )

    # main function for the data preparation
    ref_complex_handler = TrajectoryHandler(
        top_path=config.input_ref_filepath,
        trajectory_path=None,
        ligand_name=config.ref_ligname,
        distance_cutoff=config.dist_threshold_to_heavy_atom,
        radius_of_interest=config.radius_for_grid,
        spacing=config.spacing,
        warning_check=config.debug,
    )

    # get the pocket information ####
    # [update 2025.02.25] we use the same function as the holo conformation to
    # calculate the pocket center to simplify the code
    # however, there will be a slight difference in the pocket center
    # (up to 0.2Å in each direction) because the original version exclude H for
    # the calculation
    ref_complex_handler.get_residues_at_pocket(ligand_aa_dist=config.ligand_aa_dist)
    ref_complex_handler.get_pocket_center()
    ref_complex_handler.write_pocket_aux_file(config.output_aux_filepath)

    # aligned to the pocket ####
    logger.info("Aligning to the pocket...")

    # read the apo trajectory
    traj_handler = TrajectoryHandler(
        top_path=config.input_top_filepath,
        trajectory_path=config.input_traj_filepath,
        ligand_name=None,
        radius_of_interest=config.radius_for_grid,
        distance_cutoff=config.dist_threshold_to_heavy_atom,
        spacing=config.spacing,
        warning_check=config.debug,
    )

    # align the trajectory
    traj_handler.read_pocket_aux_file(config.output_aux_filepath)
    traj_handler.align_traj_to_pocket(
        reference=ref_complex_handler.universe,
        select_Hs=(not config.aligned_sele_wo_hydrogen),
    )

    # save the aligned trajectory
    traj_handler.write_trajectory(config.output_aligned_traj_filepath)

    # make dataset ####
    logger.info("Making dataset...")

    # make directory
    os.makedirs(config.output_p_data_folderpath, exist_ok=True)

    # setup the parallel processing
    p_jobs = TrajHandlerPreprocess(max_workers=config.max_workers)
    p_jobs.prepare(traj_handler, config)

    # make a parallel pool
    # p_jobs.inputs should be in the same order as the arguments of preprocess_workflow
    p_jobs.set_function(func=preprocess_workflow, logger=logger, with_label=False)
    p_jobs.run()


# holo conformation
def holo_data_preparation_recipe(config):
    """Data preparation recipe for holo conformation"""
    # logger setting
    logger = setup_logger(
        name=f"\033[32m{config.mode}\033[0m",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=f"{config.output_path}/{config.mode}-info.log",
    )

    # main function for the data preparation

    # aligned to the pocket ####
    logger.info("Aligning to the pocket...")

    # get the residues around the ligand
    traj_handler = TrajectoryHandler(
        top_path=config.input_ref_filepath,
        trajectory_path=config.input_traj_filepath,
        ligand_name=config.ligand_name,
        distance_cutoff=config.dist_threshold_to_heavy_atom,
        radius_of_interest=config.radius_for_grid,
        spacing=config.spacing,
        warning_check=config.debug,
    )
    traj_handler.get_residues_at_pocket(ligand_aa_dist=config.ligand_aa_dist)

    # save the pocket information in an auxiliary file
    traj_handler.write_pocket_aux_file(config.output_aux_filepath)

    # align the pocket
    traj_handler.align_traj_to_pocket(select_Hs=(not config.aligned_sele_wo_hydrogen))

    # save the aligned trajectory
    traj_handler.write_trajectory(config.output_aligned_traj_filepath)

    # make dataset ####
    logger.info("Making dataset...")

    # read aux file
    traj_handler.read_fragment_aux_file(config.input_labels_filepath)
    traj_handler.get_complex()

    # make directory
    os.makedirs(config.output_p_data_folderpath, exist_ok=True)

    # setup the parallel processing
    p_jobs = TrajHandlerPreprocess(max_workers=config.max_workers)
    p_jobs.prepare(traj_handler, config)

    # make a parallel pool
    # p_jobs.inputs should be in the same order as the arguments of preprocess_workflow
    p_jobs.set_function(func=preprocess_workflow, logger=logger, with_label=True)
    p_jobs.run()


# check files
def checking_workflow(config):
    """Check the files"""
    # logger setting
    logger = setup_logger(
        name=f"\033[32m{config.mode}\033[0m",
        level=logging.DEBUG,
        filepath=f"{config.output_report_folderpath}/{config.mode}-info.log",
    )

    # read the index file
    with open(config.check_index_path) as f:
        index_files = f.read().splitlines()
    indexs = [
        i.replace(f"{config.protein_filename}_", "").replace(".pdb", "")
        for i in natsorted(index_files)
    ]

    # check the files
    logger.info("Checking the files in the index file...")
    pdb_missing, ply_missing, h5_missing = [], [], []
    for index in indexs:
        pdb_filename = f"{config.protein_filename}_{index}.pdb"
        ply_filename = f"{config.protein_filename}_{index}.ply"
        h5_filename = f"{config.protein_filename}_{index}.h5"
        if not os.path.exists(f"{config.input_check_path}/{pdb_filename}"):
            pdb_missing.append(pdb_filename)
            logger.error(f"Missing: {pdb_filename}")
        if not os.path.exists(f"{config.input_check_path}/{ply_filename}"):
            ply_missing.append(ply_filename)
            logger.error(f"Missing: {ply_filename}")
        if not os.path.exists(f"{config.input_check_path}/{h5_filename}"):
            h5_missing.append(h5_filename)
            logger.error(f"Missing: {h5_filename}")

    if len(pdb_missing) == 0 and len(ply_missing) == 0 and len(h5_missing) == 0:
        logger.info("All files listed in the index file are found!")
    else:
        logger.info(f"Total missing pdb files: {pdb_missing}")
        logger.info(f"Total missing ply files: {ply_missing}")
        logger.info(f"Total missing h5 files: {h5_missing}")

    # check the trajectory and non-transformed frames
    traj_handler = TrajectoryHandler(
        config.input_ref_filepath,
        config.input_traj_filepath,
        warning_check=False,
    )
    md_length = len(traj_handler.universe.trajectory)
    logger.info(f"Total frames: {md_length}")
    logger.info(f"Total frames in the index file: {len(indexs)}")
    frames_missing = []
    for idx in range(md_length):
        if str(idx) not in indexs:
            frames_missing.append(idx)
            logger.error(f"Missing: frame {idx}")
    if len(frames_missing) == 0:
        logger.info("All frames are found!")
    else:
        logger.info(f"Total missing frames: {frames_missing}")


############################################
# Prediction
def read_model(model_ckpt_path, in_channels=4, out_channels=7, device="cpu"):
    """Read the model"""
    model = UnetModel(
        in_channels=in_channels, out_channels=out_channels, final_activation="softmax"
    )
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    return model


def add_prediction_to_ply(
    traj_handler,
    ply_path,
    h5_path,
    json_path,
    frame,
    logger,
    model_path,
    descriptor_only=False,
    **kwargs,
):
    """Add the prediction to the ply file and generate the descriptor to
    the json file"""
    try:
        if not descriptor_only:
            # read the model
            model = read_model(model_ckpt_path=model_path, **kwargs)

            # save the prediction
            traj_handler.add_prediction_to_ply(
                ply_path=ply_path,
                h5_path=h5_path,
                model=model,
            )

        # generate descriptor
        holo_descriptor = HoloDescriptor(ply_path)
        holo_descriptor.run()
        holo_descriptor.save(json_path)

        # print out finished!
        logger.info(f"Prediction: {frame} done")

    except Exception as e:
        logger.error(f"Prediction: {frame} failed, return empty holo_descriptor ({e})")
        holo_descriptor = HoloDescriptor(ply_path)
        holo_descriptor.save(json_path)


def run_prediction(config):
    """Run the prediction with parallel processing"""
    # configuration manager (output settings)
    os.makedirs(config.output_path, exist_ok=True)
    save_config(config, config.output_path)

    # logger setting
    logger = setup_logger(
        name=f"\033[32m{config.mode}\033[0m",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=f"{config.output_path}/{config.mode}-info.log",
    )

    # main function
    # load the trajectory
    traj_handler = TrajectoryHandler(
        top_path=config.input_top_filepath,
        trajectory_path=config.input_traj_filepath,
        ligand_name=None,
        warning_check=config.debug,
    )
    traj_handler.read_pocket_aux_file(config.input_aux_filepath)

    # setup the parallel processing
    p_jobs = TrajHandlerPrediction(max_workers=config.max_workers)
    p_jobs.prepare(traj_handler, config)

    # make a parallel pool
    # p_jobs.inputs should be in the same order as the arguments
    p_jobs.set_function(
        add_prediction_to_ply,
        logger=logger,
        model_path=config.model_path,
        descriptor_only=config.descriptor_only,
    )
    p_jobs.run()
