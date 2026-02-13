#!/usr/bin/env bash

# config
CURRENT_PATH=$(pwd)
PREFIX_PATH=$CURRENT_PATH/utils/thirdparty

# init config.cfg
cd $CURRENT_PATH/utils && echo "[General]" > config.cfg && cd ..


## three-party software ##
# PyMesh: [2026/02/06] Due to the lack of maintenance of PyMesh, 
# we have to use the open-source wheel file from https://github.com/yuyuan871111/PyMesh
# instead of the official distribution from PyPI. 
# Link: https://github.com/yuyuan871111/PyMesh/releases/download/v0.3.1/pymesh2-0.3-cp310-cp310-linux_x86_64.whl
# The wheel file will be downloaded directly by pixi.

# If you want to compile by yourself, you can follow the instructions below.
# conda activate py310                                                   # activate conda environment
# cd $PREFIX_PATH && git clone https://github.com/yuyuan871111/PyMesh    # clone PyMesh
# cd ./PyMesh && git submodule update --init                             # download all dependencies
# cd ./third_party && ./build.py all && cd ..                            # build third-party dependencies
# ./setup.py build && conda deactivate                                   # build PyMesh
# ./setup.py install && cd ..                                            # install PyMesh to environment
# ./setup.py bdist_wheel                                                 # (optional) build wheel file

# install python dependencies
pixi install

# Download third-party repositories
cd $PREFIX_PATH && git submodule update --init -- masif FragBEST_pymol_plugin

# Reduce: https://github.com/rlabduke/reduce/tree/master
# Use Bioconda (pixi) to install reduce (managed by pixi in this project)

# MSMS: https://ccsb.scripps.edu/msms/
# Sanner, M. F., Olson A.J. & Spehner, J.-C. (1996). Reduced Surface: An Efficient Way to Compute Molecular Surfaces. Biopolymers 38:305-320.
cd $PREFIX_PATH/msms
if [ ! -f "msms" ]; then
    wget -O msms_i86_64Linux2_2.6.1.tar.gz 'https://ccsb.scripps.edu/msms/download/933/'
    tar -xf msms_i86_64Linux2_2.6.1.tar.gz  # Linux 64-bit
    mv msms.x86_64Linux2.2.6.1 msms         # Linux 64-bit
fi
cd ..

# APBS 3.4.1: https://github.com/Electrostatics/apbs/releases/tag/v3.4.1
# if not found in current GitHub repository, please download from the below link using wget
if [ ! -f "$PREFIX_PATH/apbs/APBS-3.4.1.Linux/bin/apbs" ]; then
    wget https://github.com/Electrostatics/apbs/releases/download/v3.4.1/APBS-3.4.1.Linux.zip
    mv APBS-3.4.1.Linux.zip $PREFIX_PATH/apbs/.
    cd $PREFIX_PATH/apbs && unzip APBS-3.4.1.Linux.zip -d . && cd ..
fi
## end ##

# add line to config.cfg
cd $CURRENT_PATH/utils
echo '[ThirdParty]' >> config.cfg
echo APBS_BIN=$PREFIX_PATH/apbs/APBS-3.4.1.Linux/bin/apbs >> config.cfg
echo MULTIVALUE_BIN=$PREFIX_PATH/apbs/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue >> config.cfg
echo PDB2PQR_BIN=pdb2pqr >> config.cfg
echo PROPKA3_BIN=propka3 >> config.cfg
echo REDUCE_BIN=reduce >> config.cfg
echo REDUCE_HET_DICT=$PREFIX_PATH/reduce/reduce_wwPDB_het_dict.txt >> config.cfg
echo MSMS_BIN=$PREFIX_PATH/msms/msms >> config.cfg
echo PDB2XYZRN=$PREFIX_PATH/msms/pdb_to_xyzrn >> config.cfg
echo DLIGAND_BIN=$PREFIX_PATH/deepdrug3d >> config.cfg
cd ..