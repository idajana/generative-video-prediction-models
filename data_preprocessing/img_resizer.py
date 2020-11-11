"""
This script takes as input images, crops the redundant
frame and resizes to ratio 1:1 and selected dimension
"""
import os
import argparse
import cv2
import yaml
from utils.dir_utils import makedir


def main(input_dir, output_dir, in_width, in_height, resize_dim, crop_width, verbose=True):
    """
    :param input_dir: Input folder, assumed are two levels of subfolders
    :param output_dir: Output directory
    :param in_width: Width of the input images
    :param in_height: Height of the input images
    :param resize_dim: Resize output in ratio 1x1
    :param crop_width: Crop width for croping the black frame in the dataset
    """
    subdirs_1 = os.listdir(input_dir)
    makedir(output_dir)

    for subdir_1 in subdirs_1:
        makedir(os.path.join(output_dir, subdir_1))
        subdirs_2 = os.listdir(os.path.join(input_dir, subdir_1))
        for subdir_2 in subdirs_2:
            output_path_dir = os.path.join(output_dir, subdir_1, subdir_2)
            makedir(output_path_dir)

            input_path = os.path.join(input_dir, subdir_1, subdir_2)
            files = os.listdir(input_path)

            for file in files:
                if "right" in file or "left" in file:
                    output_path_file = os.path.join(output_path_dir, file.split(".")[0] + ".png")
                    filepath = os.path.join(input_path, file)

                    img = cv2.imread(filepath)
                    if img is None:
                        print("Skipped corrupted file: ", file)
                        continue
                    if crop_width:
                        left_width = int(in_width / 2) - int(crop_width/2)
                        right_width = int(in_width / 2) + int(crop_width / 2)
                    else:
                        #Default crops to ratio 1:1
                        left_width = int(in_width / 2) - int(in_height / 2)
                        right_width = int(in_width / 2) + int(in_height / 2)

                    cropped = img[:, left_width:right_width, :]
                    resized = cv2.resize(cropped, (resize_dim, resize_dim),
                                         interpolation=cv2.INTER_NEAREST)
                    if verbose:
                        print(output_path_file)
                    cv2.imwrite(output_path_file, resized)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--params", help="Path to YAML file with parameters",
                   default="params/data_preprocessing.yaml")
    args = a.parse_args()
    with open(args.params, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print(params)
        except yaml.YAMLError as exc:
            print(exc)

    main(params["data_path"], params["path_resized"], params["in_width"],
         params["in_height"], params["resize_dim"], params["crop_width"], params["verbose"])
