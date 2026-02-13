## Important Notice
This folder contains code modified from this source: [link](https://github.com/pulimeng/DeepDrug3D)   
* DeepDrug3D reference: 
```
Pu L, Govindaraj RG, Lemoine JM, Wu HC, Brylinski M (2019) DeepDrug3D: Classification of ligand-binding pockets in proteins with a convolutional neural network. PLOS Computational Biology 15(2): e1006718. https://doi.org/10.1371/journal.pcbi.1006718
```
The original source code is under GPL-3.0 license.

The package was modified by Yu-Yuan (Stuart) Yang (2024) in order to be compatible with our workflow under GPL-3.0 license. The modified files include:
```bash
# We only used code in DeepDrug3D/DataGeneration,
# and the modified files are listed:
build_grid.py       # MODIFIED main function for building grid
visualization.py    # MODIFIED proccessing script for visualization on VMD or PyMol
voxelization.py     # MODIFIED proccessing script for voxelization
potential.py        # MODIFIED script for calculating the potential
dligand-linux       # REMOVED
dummy_mol2.mol2     # REMOVED
fort.21_drug        # REMOVED
environmental.yml   # REMOVED
example.pdb         # MOVED to ./test
example_aux.txt     # MOVED to ./test
global_vars.py      # ADDED for generating global variable used in our workflow
write_aux_file.py    # ADDED for writing auxiliary info to a text file
```

---

# Indepedently run some DeepDrug3D functions
## How to use it?
```bash
# build grid
python build_grid.py -f ./test/example.pdb -a ./test/example_aux.txt -o ./test/output -r 5 -n 1 -s
```

  - `-f` input pdb file path.
  - `-a` input auxilary file path, with binding residue numbers and center of ligand (optional). An example of the auxilary file is provided in `./test/example_aux.txt`.
  - `-r` the radius of the spherical grid.
  - `-n` the spacing between points along the dimension of the spherical grid.
  - `-o` output folder path.
  - `-p` or `-s` whether to calculate the potential nor not. If not, only the binary occupied grid will be returne, i.e., the shape of the grid only. Default, yes (`-p`).

Several files will be saved, including `./test/output/example_transformed.pdb` (coordinate-transformed pdb file), `./test/output/example_transformed.mol2` (coordinate-transformed mol2 file for the calculation of DFIRE potential), `./test/output/example_site.grid` (grid representation of the binding pocket grid for visualization), `./test/output/example_protein.grid` (grid representation of the protein within 2 Å) and `./test/output/example.h5` (numpy array of the voxel representation).


To visualize the output binidng pocket grid, run 

```bash
# visualization
python visualization.py -i ./test/output/example_pocket.grid -c 0
```
  - `-i` input binding pocket grid file path.
  - `-c` channel to visualize. Note that if you pass `-s` in the previous step, the channel number `-c` has to be 0.
  
An output `example_grid.pdb` will be generated for visualization. Note this pocket grid matches the transformed protein `example_transformed.pdb`.
