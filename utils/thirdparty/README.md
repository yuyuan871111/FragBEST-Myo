# Third-party notice

This project (“FragBEST-Myo”) uses third‑party software and, in some cases, includes adapted source code. The third‑party components listed below are licensed by their respective authors under their own license terms. License texts and additional notices apply as provided by the upstream projects and/or as included in this repository.

- **Bundled / vendored code**: located under `utils/thirdparty/` (when applicable).
- **Externally obtained tools**: some tools may be downloaded and installed from their official sources during setup; those are not necessarily redistributed as part of this repository.
- **Python dependencies** (installed from package managers) are listed in `pyproject.toml` and/or the pixi environment configuration.

If you believe an attribution is missing or incorrect, please open an issue or pull request.

---

## Summary inventory

| Component | Upstream | License | How used here | Where to find it |
|---|---|---|---|---|
| 1\)`MaSIF` | https://github.com/LPDI-EPFL/masif | Apache‑2.0 | Adapted parts in the fork and used it as a submodule of FragBEST-Myo | Patched fork: https://github.com/yuyuan871111/masif/tree/master |
| 2\)`PyMesh` | https://github.com/PyMesh/PyMesh | MPL | Patched fork used for Python 3.10 and obtained it from our fork (downloaded and installed by configured environment setup) | Patched fork: https://github.com/yuyuan871111/PyMesh |
| 3\)`msms` | https://ccsb.scripps.edu/msms/ | See upstream license/terms | External tool (downloaded and installed by configured environment setup) | Official source |
| 4\)`APBS` | https://github.com/Electrostatics/apbs | See upstream LICENSE | External tool (downloaded and installed by configured environment setup) | Official source |
| 5\)`DeepDrug3D` | https://github.com/pulimeng/DeepDrug3D | GPL‑3.0 | Adapted source code included | `utils/thirdparty/deepdrug3d` |
| 6\)`3D U‑Net (PyTorch)` | https://github.com/zyl200846/3D-UNet-PyTorch-Implementation | MIT | Adapted source code included | `utils/thirdparty/unet3d_model` |
| 7\)`FragBEST_pymol_plugin` | https://github.com/LPDI-EPFL/masif/tree/master/source/masif_pymol_plugin | Derived from MaSIF (Apache‑2.0) | Adapted plugin for visualization | New repository adapted from `masif_pymol_plugin`: https://github.com/yuyuan871111/FragBEST_pymol_plugin |

> Note: “License” links below point to upstream license files when available. Where code is vendored into this repository, license texts are included alongside the vendored code.

## Details

### 1. Patched forks / adapted repositories (not necessarily bundled here)

#### 1\)MaSIF (adapted; Apache‑2.0)
- **Upstream:** https://github.com/LPDI-EPFL/masif  
- **License:** Apache‑2.0 (upstream: https://github.com/LPDI-EPFL/masif/blob/master/LICENSE)  
- **Our adapted repository:** https://github.com/yuyuan871111/masif/tree/5e9cb94a0c80944990c7a86dcce80555f2c3eda5  
- **Role:** Prepare ply file (include only chemical features per vertex) from pdb file.
- **Notes:** We adapted portions of MaSIF to ensure compatibility with Python 3.10 and our workflow.  

#### 2\)PyMesh (patched; MPL)
- **Upstream:** https://github.com/PyMesh/PyMesh  
- **License:** MPL (see upstream repository)  
- **Our patched fork:** https://github.com/yuyuan871111/PyMesh  
- **Role:** Handle ply surface files, attributes, and to regularize meshes.
- **Notes:** PyMesh is officially unsupported on Python 3.10; we maintain a patched fork for compatibility.  

#### 7\)FragBEST_pymol_plugin (adapted from MaSIF; Apache‑2.0)
- **Upstream basis:** MaSIF PyMOL plugin: https://github.com/LPDI-EPFL/masif/tree/master/source/masif_pymol_plugin  
- **Our adapted plugin repo:** https://github.com/yuyuan871111/FragBEST_pymol_plugin/tree/master  
- **Role:** Visualize protein surface vertices, chemical features, and labels per vertex.  
- **License:** Derived from MaSIF (Apache‑2.0); see MaSIF license and any applicable NOTICE requirements.

---

### 2. External tools (obtained from official sources)

#### 3\)msms
- **Upstream:** https://ccsb.scripps.edu/msms/  
- **License/terms:** https://ccsb.scripps.edu/msms/license/  
- **Role:** Compute reduced protein surfaces.  
- **Important:** Commercial users should review MSMS license terms with the author.  
- **Distribution note:** Typically obtained from the official source during setup.

#### 4\)APBS
- **Upstream:** https://github.com/Electrostatics/apbs/tree/main  
- **License:** https://github.com/Electrostatics/apbs/blob/main/LICENSE.md  
- **Role:** Adaptive Poisson–Boltzmann solver used by the pipeline.  
- **Distribution note:** Typically installed/downloaded from official sources during setup.

---

### 3. Included (vendored) third‑party code under `utils/thirdparty`

#### 5\)DeepDrug3D (adapted; GPL‑3.0)
- **Upstream:** https://github.com/pulimeng/DeepDrug3D/tree/master  
- **License:** GPL‑3.0 (upstream: https://github.com/pulimeng/DeepDrug3D/blob/master/LICENSE)  
- **In this repository:** `utils/thirdparty/deepdrug3d`  
- **Role:** Find residues at the binding site by residue ids or pocket center from a text file.
- **Notes:** We adapted portions of DeepDrug3D for compatibility with our workflow/environment. The adapted source is distributed under GPL‑3.0 as required.

#### 6\)3D U‑Net PyTorch implementation (adapted; MIT)
- **Upstream:** https://github.com/zyl200846/3D-UNet-PyTorch-Implementation/tree/master  
- **License:** MIT (upstream: https://github.com/zyl200846/3D-UNet-PyTorch-Implementation/blob/master/LICENSE)  
- **In this repository:** `utils/thirdparty/unet3d_model`  
- **Role:** 3D U-Net model.
- **Notes:** We adapted portions for integration. MIT license notice requirements are preserved with the included sources.

---

### 4. Notes on licensing & “license conflicts”

This repository may combine components under different licenses (e.g., Apache‑2.0, GPL‑3.0, MIT etc.). We handle license compatibility using an approach similar to the one described by MDAnalysis (rationale and policy):  
https://www.mdanalysis.org/2023/09/22/licensing-update/#rationale-for-our-change-in-target-license

> This section is informational and does not override the licenses of third‑party components. Each component remains under its own license.

---

### 5. Where to find the complete dependency list

- Python package dependencies: `pyproject.toml` (and pixi environment configuration, if present).
- Bundled third‑party sources and their internal license files: `utils/thirdparty/` and `LICENSES`.

---

### 6. Corrections

If any item above is inaccurate (e.g., a component is bundled vs downloaded, or license metadata changed), please open an issue/PR with:
- component name + upstream URL
- license identifier + license link
- where it is used in this repository
- whether it is bundled, submoduled, or installed externally
