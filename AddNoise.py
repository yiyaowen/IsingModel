from os import path
from PIL import Image
import click
import numpy as np


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Convert image to numpy array
    img_array = np.array(image)

    # Generate random noise
    random_matrix = np.random.rand(*img_array.shape)

    # Add salt noise
    img_array[random_matrix < salt_prob] = 255

    # Add pepper noise
    img_array[random_matrix > 1 - pepper_prob] = 0

    return Image.fromarray(img_array)


def add_poisson_noise(image):
    # Convert image to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Generate Poisson noise
    noisy_array = np.random.poisson(img_array)

    # Clip values to valid range for image data
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_array)


@click.command()
@click.argument(
    "source",
    type=click.Path(exists=True)
    )
@click.option(
    "--noise-type", "-t",
    default="salt_pepper",
    type=click.Choice(["salt_pepper", "poisson"])
    )
@click.option(
    "--noise-param", "-p",
    default=0.1,
    help="Probability for salt and pepper noise."
    )
def main(
    source, noise_type, noise_param
    ):
    image_path = path.abspath(source)

    # Load grayscale image
    input_image = Image.open(image_path).convert("L")

    noise_image = None
    match noise_type:
        case "salt_pepper":
            # Add salt and pepper noise
            noisy_image = add_salt_and_pepper_noise(
                input_image, noise_param, noise_param)
        case "poisson":
            # Add Poisson noise
            noisy_image = add_poisson_noise(input_image)

    image_dirname = path.dirname(image_path)
    image_basename = path.basename(image_path)

    # Save the noisy image
    noisy_image.save(f"{image_dirname}/noise_{image_basename}")


if __name__ == "__main__":
    main()
