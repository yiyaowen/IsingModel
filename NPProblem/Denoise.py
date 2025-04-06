from os import path
from PIL import Image
import numpy as np

from NPProblem.Coordinate import Coordinate


def convert(image_path, color_type):
    """
    Convert a grayscale image to a Spin/JH-coupling system.
    """
    output_path = path.abspath(image_path)

    output_dir = path.dirname(output_path)
    output_name = path.basename(output_path)
    output_name = path.splitext(output_name)[0]

    img = Image.open(image_path)
    match color_type:
        case "monochrome":
            img = img.convert("1")
        case "grayscale":
            img = img.convert("L")
        case _:
            raise ValueError("Invalid Color Type")

    h_system = np.array(img, dtype=int)

    h_output_txt = f"{output_dir}/h_{output_name}.txt"
    np.savetxt(f"{h_output_txt}", h_system, fmt="%+4d")

    cd = Coordinate(*img.size)

    j_system = np.full(cd.full_size, +1, dtype=int)
    j_system[::2, ::2] = h_system

    j_output_txt = f"{output_dir}/j_{output_name}.txt"
    np.savetxt(j_output_txt, j_system, fmt="%+4d")
