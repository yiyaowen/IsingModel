from os import path
from PIL import Image
import numpy as np

from NPProblem.Coordinate import Coordinate


def convert(image_path, graph_type):
    """
    Convert a monochrome image to a Spin/J-coupling system.
    """
    output_path = path.abspath(image_path)
    output_name = path.splitext(output_path)[0]

    img = Image.open(image_path)
    img = img.convert("1")

    cd = Coordinate(*img.size)

    system = np.full(cd.full_size, +7, dtype=np.int8)
    system[::2, ::2] = +1 * np.ones(cd.size)

    # TODO: Optimize the set of the J-couplings
    for r in range(cd.rows):
        for c in range(cd.cols):
            if img.getpixel((c, r)) == 255:
                match graph_type:
                    case "nearest":
                        """
                        Nearest Graph
                        -------------
                        |   | 1 |   |
                        ------↑------
                        | 4 ← 0 → 2 |
                        ------↓------
                        |   | 3 |   |
                        -------------
                        """
                        sc0 = cd.sc(r + 0, c + 0)
                        sc1 = cd.sc(r - 1, c + 0)
                        sc2 = cd.sc(r + 0, c + 1)
                        system[cd.jc(sc0, sc1)] = -7
                        system[cd.jc(sc0, sc2)] = -7
                    case "king":
                        """
                        King's Graph
                        -------------
                        | 1 | 2 | 3 |
                        ----↖-↑-↗----
                        | 8 ← 0 → 4 |
                        ----↙-↓-↘----
                        | 7 | 6 | 5 |
                        -------------
                        """
                        sc0 = cd.sc(r + 0, c + 0)
                        sc2 = cd.sc(r - 1, c + 0)
                        sc3 = cd.sc(r - 1, c + 1)
                        sc4 = cd.sc(r + 0, c + 1)
                        system[cd.jc(sc0, sc2)] = -7
                        system[cd.jc(sc0, sc3)] = -7
                        system[cd.jc(sc0, sc4)] = -7
                    case _:
                        raise ValueError("Invalid Graph Type")

    np.savetxt(f"{output_name}.txt", system, fmt="%+d")
