from PIL import Image
from sys import argv
from termcolor import cprint
import numpy as np
import skimage.metrics as skm


def calc_metrics(base_path, noisy_paths):
    metrics = dict()

    # Load the base image
    base = np.array(Image.open(base_path).convert('L'))

    # Iterate through noisy images and calculate metrics
    for noisy_path in noisy_paths:
        noisy = np.array(Image.open(noisy_path).convert('L'))

        # Calculate PSNR
        psnr_value = skm.peak_signal_noise_ratio(base, noisy, data_range=255)
        # Calculate SSIM
        ssim_value = skm.structural_similarity(base, noisy, data_range=255)

        metrics[noisy_path] = { "psnr": psnr_value, "ssim": ssim_value }

    return metrics


def show_metrics(metrics_items):
    for noisy_path, values in metrics_items:
        psnr_value = values["psnr"]
        ssim_value = values["ssim"]

        name_color = None
        psnr_color = None
        ssim_color = None
        # Highlight the best PSNR and SSIM
        if noisy_path == max_psnr_path:
            name_color = "blue"
            psnr_color = "blue"
        if noisy_path == max_ssim_path:
            name_color = "blue"
            ssim_color = "blue"
        if noisy_path == max_psnr_path and noisy_path == max_ssim_path:
            name_color = "green"
            psnr_color = "green"
            ssim_color = "green"

        cprint(f"\nImage: {noisy_path}\n", name_color)
        cprint(f"  PSNR: {psnr_value:.2f}", psnr_color)
        cprint(f"  SSIM: {ssim_value:.4f}", ssim_color)


if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: python CalcPSNR.py <base_image> <noisy_image1> [<noisy_image2> ...]")
        exit(1)
    base_image = argv[1]
    noisy_images = argv[2:]
    metrics = calc_metrics(base_image, noisy_images)

    # Sort metrics by PSNR first, then by SSIM in descending order
    reorder = lambda item: (item[1]["psnr"], item[1]["ssim"])
    metrics_items = sorted(metrics.items(), key=reorder, reverse=True)

    # Find the noisy_path with the maximum PSNR and SSIM
    max_psnr_path = max(metrics, key=lambda path: metrics[path]["psnr"])
    max_ssim_path = max(metrics, key=lambda path: metrics[path]["ssim"])

    show_metrics(metrics_items)
