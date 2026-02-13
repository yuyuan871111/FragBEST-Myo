# This file is derived from DeepDrug3D 
# Original copyright (c) 2019 Pu et al.

# Modifications:
# - Adapted for Python 3.10 compatibility 
# - Integrated into FragBEST-Myo workflow 

# Modified by Yu-Yuan (Stuart) Yang, 2024 
# Licensed under GPL-3.0 (see LICENSE)

import argparse
import string

import numpy as np
import pandas as pd


def makeLine(coord, potE, k, num_chr_dict, occu=1):
    atom = "ATOM"
    atom_sn = str(k + 1)
    atom_name = "D1"
    assert k < 26**4, "Too many points"
    thousand, one = np.divmod(k, 26**3)
    number_sub = k - thousand * 26**3
    hundred, one = np.divmod(number_sub, 26**2)
    number_sub = number_sub - hundred * 26**2
    ten, one = np.divmod(number_sub, 26)
    res_name = (
        num_chr_dict[hundred] + num_chr_dict[ten] + num_chr_dict[one]
    )  # 3-letter residue name (the code will reuse again when all combinations are used)
    x = "{:.3f}".format(round(coord[0], 3))
    y = "{:.3f}".format(round(coord[1], 3))
    z = "{:.3f}".format(round(coord[2], 3))
    OC = "{:.2f}".format(round(occu, 2))
    EE = "{:.2f}".format(round(potE, 2))
    string = (
        atom
        + " " * 2
        + "{:>5}".format(atom_sn)
        + " "
        + "{:4}".format(atom_name)
        + " "
        + "{:>3}".format(res_name)
        + " " * 2
        + "   1"
        + " " * 4
        + "{:>8}".format(x)
        + "{:>8}".format(y)
        + "{:>8}".format(z)
        + "{:>6}".format(OC)
        + "{:>6}".format(EE)
        + " " * 8
        + "\n"
    )
    return string


def main(input, channel):
    num_range = np.linspace(0, 25, 26, dtype=int)
    chr_range = list(string.ascii_uppercase[:27])
    num_chr_dict = dict(zip(num_range, chr_range))

    site = pd.read_csv(input)
    site = site.to_numpy()
    with open("{}_grid_channel_{}.pdb".format(input[:-5], channel), "w") as in_strm:
        for k in range(len(site)):
            temp_coords = site[k, :]
            cl = temp_coords[channel]
            temp_string = makeLine(temp_coords, cl, k, num_chr_dict)
            in_strm.write(temp_string)
    print("Number of points is " + str(len(site)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input grid file"
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        required=True,
        help="Channel to visualize (shape only: -c 0)",
    )
    opt = parser.parse_args()
    main(opt.input, opt.channel)
