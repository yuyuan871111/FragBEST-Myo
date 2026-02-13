Developer notes
---------------

**Source codes structure:** 

.. code-block:: bash

    FragBEST-Myo
    |--- config                 # .yaml file for train, test, inference (command line usage)
    |--- dataset                # example and reference files 
    |--- docs                   # source of documentation
    |--- imgs                   # source of images
    |--- tutorial               # tutorial (user guide) to help you to understand how to run our tool
    |--- utils                  # main python codebase
    |--- datasets             # high-level to deal with dataset
    |--- default_config       # default configuration to define third-party path
    |--- parallel             # parallel framework for multiple cores
    |--- ppseg                # Protein Pocket SEGmentation codebase
        |--- analysis           # post-process analysis for DL results
        |--- holo_descriptor    # holo-descriptor related functions
        |--- ignite             # PyTorch Ignite (DL model train/val/test)
        |--- myo                # FragBEST-Myo model and auxiliary files
        |--- visualization      # visualization/plotting of DL results
    |--- pymol_scripts        # generate PyMOL visualization state (.pse)
    |--- tests                # testing scripts
    |--- thirdparty           # script modified from third-party open sources
    |--- pytest.ini           # pytest configuration
    |--- seed.py              # fix seed to make the work reproducible
    |--- main.py                # command line usage script

    # environment
    |--- build.sh               # script for building the environment
    |--- pyproject.toml         # required packages and functions in pixi environment

    # others
    |--- .gitignore             # git ignore manager
    |--- .gitmodules            # git submodule manager
    |--- .gitattributes         # git attribute manager



The below codes should be run in the `pixi` environment.  

.. code-block:: bash

   # Activate the virtual environment (pixi)   
   pixi run shell   
   # in the virtual environment, it looks like:   
   # (FragBEST-Myo) username@servername:~/path/to/FragBEST-Myo$   


Check code style
================

.. code-block:: bash

    poe style-check  


Fix code style
==============

.. code-block:: bash

    poe style-fix   


Test packages
=============

.. code-block:: bash

    poe test  
    

Build docs (locally, interactively)
===================================

.. code-block:: bash

    poe build-doc-dev  


Build docs (HTML)
=================

.. code-block:: bash
    
    poe build-doc  


Deep learning - Semantic Segmentation (pytorch-ignite)
======================================================

.. image:: https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700
   :target: https://github.com/pytorch-ignite/code-generator
   :alt: Code-Generator

The deep learning model framework is based on PyTorch Ignite. 
We used the codes from the code generator as the starting point 
and then optimize the workflow.   

Cheatsheet for developers
=========================

.. toctree::
   :maxdepth: 1

   cheatsheet/Sphinx_cheatsheet
   cheatsheet/Conda_cheatsheet
   cheatsheet/Poetry_cheatsheet
    

   