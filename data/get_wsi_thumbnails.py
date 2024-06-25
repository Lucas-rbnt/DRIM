import pyvips
import os
import tqdm
import cv2
import numpy as np
from typing import Tuple
from PIL import Image
import argparse


def segment(
    img_rgba: np.ndarray,
    sthresh: int = 25,
    sthresh_up: int = 255,
    mthresh: int = 9,
    otsu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    # img_med = filters.median(img_hsv))#, np.ones((3, 3,3))
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring
    # img_med = img_hsv
    if otsu:
        _, img_otsu = cv2.threshold(
            img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(img, img, mask=img_otsu)
    return masked_image, img_otsu


def main(data_path, downscale_factor):
    """
    This script creates thumbnails and low resolution masks of the WSI images.
    The data folder must be organized as follows (from a TCGA download):
    - data_path
        - id1
            - filename1.svs
            - ...
        - id2
            - filename2.svs
            - ...
        - ...

    Args:
        data_path: Path to the input directory.
        downscale_factor: Downscale factor for x20 magnification.

    Returns:
        None

    ----
    Example usage:
        python get_wsi_thumbnails.py --data_path /data/ --downscale_factor 6

    """
    subdirectories = os.listdir(data_path)
    for subdirectory in tqdm.tqdm(subdirectories):
        subdirectory_path = os.path.join(data_path, subdirectory)
        filenames = os.listdir(subdirectory_path)
        wsi_filename = [f for f in filenames if f.endswith("svs") or f.endswith("tif")][
            0
        ]
        slide = pyvips.Image.new_from_file(
            os.path.join(subdirectory_path, wsi_filename)
        )
        if int(float(slide.get("aperio.AppMag"))) == 40:
            d = downscale_factor + 1
        else:
            d = downscale_factor
        thumbnail = pyvips.Image.thumbnail(
            os.path.join(subdirectory_path, wsi_filename),
            slide.width / (2**d),
            height=slide.height / (2**d),
        ).numpy()

        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
        thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        # to filter out felt-tip marks
        mask_hsv = np.tile(thumbnail_hsv[:, :, 1] < 160, (3, 1, 1)).transpose(1, 2, 0)
        thumbnail *= mask_hsv
        masked_image, mask = segment(thumbnail)
        masked_image = Image.fromarray(masked_image).convert("RGB")
        # save
        masked_image.save(os.path.join(subdirectory_path, "thumbnail.jpg"))
        np.save(os.path.join(subdirectory_path, "mask.npy"), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create thumbnails and low resolution masks of the WSI images."
    )
    parser.add_argument("--data_path", "-p", help="Path to the input directory.")
    parser.add_argument(
        "--downscale_factor",
        "-d",
        type=int,
        help="Downscale factor for x20 magnification.",
    )
    args = parser.parse_args()
    main(args.data_path, args.downscale_factor)
