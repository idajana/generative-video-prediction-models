import argparse
import numpy as np
import os
import csv
import cv2
"""
This script transforms csv + imgs to npz rollouts
"""

def str2array(input_str):
    """
    Convert string to array
    :param input: str
    action in string format
    :return: float np.array
    action
    """
    action = input_str.split("[")
    action = action[1].split("]")
    action = action[0].split(" ")
    action_final = []
    for el in action:
        if not el == '':
            action_final.append(float(el))
    return action_final

def generate_rollouts(csv_path, output_dir, resize_dim):
    """
    Generate npz rollouts and save to output dir
    :param csv_path: str
    :param output_dir: str
    :param resize_dim: False or int
    :return:
    """
    csv_files = os.listdir(csv_path)
    for csv_file in csv_files:
        with open(os.path.join(csv_path ,csv_file)) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            first = True
            actions = []
            observations = []
            for row in readCSV:
                if first:
                    first = False
                    continue
                action = np.array(str2array(row[1]))
                actions.append(action)
                img_name = row[2]

                obs = cv2.imread(img_name)
                if resize_dim:
                    obs = cv2.resize(obs, (resize_dim, resize_dim),
                                         interpolation=cv2.INTER_NEAREST)
                im_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                observations.append(im_rgb)
            rollout_name = img_name.split("/")[-2]

            np.savez(os.path.join(output_dir, rollout_name),
                                     observations=np.array(observations),
                                     actions=np.array(actions))

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--dir", help="Data path")
    a.add_argument("--resize", default=False)
    args = a.parse_args()
    data_dir = args.dir
    csv_dir = os.path.join(data_dir, "csv_files")
    out_dir = os.path.join(data_dir, "rollouts_npz")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if args.resize:
        resize_param = int(args.resize)
    else:
        resize_param = False
    generate_rollouts(csv_dir, out_dir, resize_param)


