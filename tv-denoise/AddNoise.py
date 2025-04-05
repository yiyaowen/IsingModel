from os import path
from PIL import Image
from sys import argv
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

if __name__ == "__main__":
    for image_path in argv[1:]:
        image_path = path.abspath(image_path)

        # Load binary image
        input_image = Image.open(image_path).convert("L")

        # Add salt and pepper noise
        salt_probability = 0.1 # Adjust as needed
        pepper_probability = 0.1 # Adjust as needed

        noisy_image = add_salt_and_pepper_noise(
            input_image, salt_probability, pepper_probability)

        image_dirname = path.dirname(image_path)
        image_basename = path.basename(image_path)

        # Save the noisy image
        noisy_image.save(f"{image_dirname}/noise_{image_basename}")
