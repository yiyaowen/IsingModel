from os import path
from PIL import Image
import numpy as np

from NPProblem.Coordinate import Coordinate


def convert(image_path):
    """
    Convert a monochrome image to a Spin/JH-coupling system.
    """
    output_path = path.abspath(image_path)

    output_dir = path.dirname(output_path)
    output_name = path.basename(output_path)
    output_name = path.splitext(output_name)[0]

    img = Image.open(image_path)
    img = img.convert("1")

    h_system = (np.array(img) * 2) - 1

    h_output_txt = f"{output_dir}/h_{output_name}.txt"
    np.savetxt(f"{h_output_txt}", h_system, fmt="%+d")

    cd = Coordinate(*img.size)

    j_system = np.full(cd.full_size, +1, dtype=np.int8)
    j_system[::2, ::2] = h_system

    j_output_txt = f"{output_dir}/j_{output_name}.txt"
    np.savetxt(j_output_txt, j_system, fmt="%+d")
