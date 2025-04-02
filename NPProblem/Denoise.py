from os import path
from PIL import Image
import numpy as np

from NPProblem.Coordinate import Coordinate


def convert(image_path, output_txt):
    """
    Convert a monochrome image to a Spin/JH-coupling system.
    """
    output_txt = path.abspath(output_txt)

    f_dirname = path.dirname(output_txt)
    f_basename = path.basename(output_txt)

    img = Image.open(image_path)
    img = img.convert("1")

    h_system = (np.array(img) * 2) - 1

    h_output_txt = f"{f_dirname}/h_{f_basename}"
    np.savetxt(f"{h_output_txt}", h_system, fmt="%+d")

    cd = Coordinate(*img.size)

    j_system = np.full(cd.full_size, +1, dtype=np.int8)
    j_system[::2, ::2] = h_system

    j_output_txt = f"{f_dirname}/j_{f_basename}"
    np.savetxt(j_output_txt, j_system, fmt="%+d")
