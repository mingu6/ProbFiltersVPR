import os
import argparse
import sys
import itertools
from multiprocessing import Pool

from tqdm import tqdm, trange
import PIL

from src.thirdparty.robotcar_dataset_sdk import image, camera_model
from src.settings import RAW_PATH, READY_PATH
from src import params, utils


def validate_images(traverse, camera):
    """
    TODO:
    Check set of images against timestamps file to identify any missing images.
    """
    return None


def process_and_save_image(paths):
    img_path = paths[0]
    save_path = paths[1]
    cm = paths[2]
    arr = image.load_image(img_path, cm)
    img = PIL.Image.fromarray(arr)
    img.save(save_path)
    return


def ready_images(traverse, camera, n_workers=4, resume=True):
    # load and undistort images
    image_folder_path = os.path.join(RAW_PATH, traverse, camera)
    cm = camera_model.CameraModel(
        utils.models_path, image_folder_path
    )  # this is a variable used in process_and_save_image
    try:
        os.listdir(image_folder_path)
    except FileNotFoundError:
        print(
            "Folder {} not found, please check that this traverse/camera"
            "combination  exists."
        )
        return
    ready_folder_path = os.path.join(READY_PATH, traverse, camera)
    if resume and os.path.exists(ready_folder_path):
        ready_fnames = os.listdir(ready_folder_path)
        img_fnames = [
            img_path
            for img_path in os.listdir(image_folder_path)
            if img_path.endswith(".png") and img_path not in ready_fnames
        ]
    else:
        img_fnames = [
            img_path
            for img_path in os.listdir(image_folder_path)
            if img_path.endswith(".png")
        ]
        if not os.path.exists(ready_folder_path):
            os.makedirs(ready_folder_path)
    try:
        full_raw_path = [os.path.join(image_folder_path, fname) for fname in img_fnames]
        full_ready_path = [
            os.path.join(ready_folder_path, fname) for fname in img_fnames
        ]
        with Pool(n_workers) as pool:
            list(
                tqdm(
                    pool.imap(
                        process_and_save_image,
                        zip(full_raw_path, full_ready_path, itertools.repeat(cm)),
                    ),
                    total=len(full_raw_path),
                )
            )
    except FileNotFoundError as e:
        print(e)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Undistorts images from the RobotCar dataset"
    )
    parser.add_argument(
        "-n", "--n-workers", type=int, default=4, help="Number of workers to use"
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Skip images already undistored in traverse/camera combinations",
    )
    args = parser.parse_args()

    for traverse in params.traverses.values():
        ready_images(traverse, "stereo/left", args.n_cpu, args.resume)
