import cv2
import numpy as np
from pathlib import Path
from spectral.svd import truncated_svd


def load_image(file_path, grayscale):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR_RGB
    image = cv2.imread(file_path, flag)
    if image is None:
        raise FileNotFoundError(f"Could not load or find the image at: {file_path}")
    return image


def calculate_psnr(original, reconstructed):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def svd_compression():
    path = r"image_path"
    img = load_image(path, grayscale=False)
    b, g, r = cv2.split(img)
    height, width = img.shape[:2]

    print(f"Original image shape: {img.shape}")
    print(f"Red channel shape:    {r.shape}")
    print(f"Green channel shape:  {g.shape}")
    print(f"Blue channel shape:   {b.shape}")

    output_dir = Path("outputs/compression")
    output_dir.mkdir(parents=True, exist_ok=True)

    ranks = [5, 120, 50]
    original_values = height * width * 3

    for k in ranks:
        reconstructed_channels = []

        for channel in [b, g, r]:
            result = truncated_svd(channel, k)
            reconstructed_channels.append(result["A_approx"])

        reconstructed = cv2.merge(reconstructed_channels)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        stored_values = 3 * k * (height + width + 1)
        compression_percentage = (1 - stored_values / original_values) * 100
        psnr = calculate_psnr(img, reconstructed)

        if psnr >= 35:
            quality = "near original"
        elif psnr >= 25:
            quality = "decent"
        else:
            quality = "blurry"

        print(
            f"Rank-{k:<3} | Compression: {compression_percentage:.1f}% | "
            f"PSNR: {psnr:.1f} dB [{quality}]"
        )

        output_path = output_dir / f"rank_{k}.png"
        cv2.imwrite(str(output_path), reconstructed)

    print(f"Reconstructed images saved to: {output_dir}/")


svd_compression()
