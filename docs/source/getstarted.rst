Installation
------------
FragBEST-Myo uses `pixi` to manage the packages and virtual environments.

.. note::  
   A virtual environment is created to ensure that all dependencies 
   are effectively managed and kept separate from your system's 
   Python installation.


**Prerequisite package:** `pixi`_.

**System requirement:** Linux.


FragBEST-Myo basically follows the same preparation protocol as `MaSIF`_ 
to generate chemical features and surface files, so some third-party libraries 
and programs are required, including `msms`_, `PyMesh`_, `APBS`_, `reduce`_. 

However, in order to make the packages compatible with our workflow and 
python version (3.10), we modified some code in `PyMesh`_ and `MaSIF`_ 
(see `pymesh change logs`_ and `masif change logs`_).

Additionally, FragBEST-Myo directly adapts code from 
`DeepDrug3D`_ and `3D U-Net pytorch implementation`_  
based on GPL-3.0 and MIT licenses, respectively. 
Other dependencies are shown in `pyproject.toml`_.

Please check more details about the third-party dependencies used in this project:

.. toctree::
   :maxdepth: 1

   thirdparty


**To use FragBEST-Myo, start by installing the necessary dependencies. Please follow the steps below:**

.. important::

   The installation code will automatically download `msms`_ (check with 
   `the author <https://ccsb.scripps.edu/msms/license/>`_ if you are a commercial user) 
   and `APBS`_ from the source links, install the `adapted masif`_, `reduce`_ and 
   `FragBEST_pymol_plugin` (adapted from `MaSIF`_) from `Github repo` (as a submodule), 
   use our `adapted DeepDrug3D <https://github.com/fornililab/FragBEST-Myo/tree/main/utils/thirdparty/deepdrug3d>`_ 
   and `adapted 3D U-Net pytorch implementation <https://github.com/fornililab/FragBEST-Myo/tree/main/utils/thirdparty/unet3d_model>`_ 
   (located in ``./utils/thirdparty``). Since `PyMesh`_ is officially unsupported for python 3.10, we installed 
   `PyMesh`_ by a pre-built wheel `here <https://github.com/fornililab/FragBEST-Myo/tree/main/utils/thirdparty/pymesh/pymesh2-0.3-cp310-cp310-linux_x86_64.whl>`_ 
   from our `adapted PyMesh repo <https://github.com/yuyuan871111/PyMesh>`_.


.. code-block:: bash

   # Clone the repository
   git clone https://github.com/fornililab/FragBEST-Myo.git

   # Build the virtual environment
   cd FragBEST-Myo
   pixi run build




**To verify the success of the installation, execute the following code:**

.. code-block:: bash

   # Test the installation
   pixi run poe test -n 4   # "-n 4" means using 4 threads to run the tests parallely
   # Some warnings are expected, but all tests should pass.

   # Test the installation of pymol environment
   pixi run -e pymol pymol --help
   # You should be able to view the help documentation for PyMOL.




Activate the environment
------------------------

To activate FragBEST-Myo's environment, you have to be in `~/path/to/FragBEST-Myo` or its 
subfolder to activate the virtual environment: 

.. code-block:: bash

   # Activate the virtual environment (pixi)
   pixi run shell
   # in the virtual environment, it looks like:
   # (FragBEST-Myo) username@servername:~/path/to/FragBEST-Myo$


Activate jupyter notebook
-------------------------
WHen you are in the virtual environment, you can start jupyter notebooks.
Here we demonstrate how to open the tutorial notebooks.

.. code-block:: bash

   # (in the virtual environment) Go to the tutorial directory
   cd tutorial

   # (in the virtual environment) Activate jupyter notebook
   jupyter notebook
   # This will open the jupyter notebook interface in your default web browser.
   # Then, you can open any notebook file (.ipynb) to start exploring FragBEST-Myo.


Leave the environment
---------------------
To leave the virtual environment, you can simly run:

.. code-block:: bash

   # Leave the virtual environment (pixi)
   exit
   # return to:
   # username@servername:~/path/to/FragBEST-Myo$


.. _pixi: https://pixi.sh/latest/

.. _MaSIF: https://github.com/LPDI-EPFL/masif

.. _msms: https://ccsb.scripps.edu/msms/

.. _PyMesh: https://github.com/PyMesh/PyMesh

.. _APBS: https://github.com/Electrostatics/apbs

.. _DeepDrug3D: https://github.com/pulimeng/DeepDrug3D

.. _reduce: https://github.com/rlabduke/reduce

.. _3D U-Net pytorch implementation: https://github.com/zyl200846/3D-UNet-PyTorch-Implementation

.. _pymesh change logs: https://github.com/PyMesh/PyMesh/compare/main...yuyuan871111:PyMesh:main

.. _masif change logs: https://github.com/LPDI-EPFL/masif/compare/master...yuyuan871111:masif:master

.. _pyproject.toml: https://github.com/fornililab/FragBEST-Myo/blob/main/pyproject.toml

.. _adapted masif: https://github.com/yuyuan871111/masif/tree/5e9cb94a0c80944990c7a86dcce80555f2c3eda5

.. _FragBEST_pymol_plugin: https://github.com/yuyuan871111/FragBEST_pymol_plugin