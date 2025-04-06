from cv2 import bilateralFilter
from os import path
from PIL import Image, ImageFilter
import click
import numpy as np


def apply_filter(image, filter_type, filter_param):
    """
    Apply a filter to the image based on the filter type.
    """
    match filter_type:
        case "median":
            size = 2 * filter_param + 1
            return image.filter(ImageFilter.MedianFilter(size=size))
        case "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius=filter_param))
        case "bilateral":
            d = 2 * filter_param + 1
            # Apply bilateral filter using OpenCV
            return Image.fromarray(bilateralFilter(
                np.array(image), d=d, sigmaColor=75, sigmaSpace=75))


@click.command()
@click.argument(
    "source",
    type=click.Path(exists=True)
    )
@click.option(
    "--filter-type", "-t",
    default="median",
    type=click.Choice(["median", "gaussian", "bilateral"]),
    )
@click.option(
    "--filter-param", "-p",
    default=1,
    help="Half-size of the filter kernel."
    )
def main(
    source, filter_type, filter_param
    ):
    image_path = path.abspath(source)

    # Load grayscale image
    input_image = Image.open(image_path).convert("L")
    output_image = apply_filter(input_image, filter_type, filter_param)

    image_dirname = path.dirname(image_path)
    image_basename = path.basename(image_path)

    # Save processed image
    output_image.save(f"{image_dirname}/filter_{image_basename}")


if __name__ == "__main__":
    main()
