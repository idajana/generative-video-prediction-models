import os
import argparse
import json
import numpy as np
import yaml

from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from utils.img_utils import load_image
from utils.dir_utils import makedir

np.set_printoptions(suppress=True)

LIMIT = 50000000.0
hist_dict = {
    0: "Q Real part",
    1: "Q Vector i",
    2: "Q Vector j",
    3: "Q Vector k",
    4: "Pose vector x",
    5: "Pose vecotr y",
    6: "Pose vector z"
}

def calculate_delta(pos1, pos2):
    """
    Calculate delta between two positions
    :param pos1: np.array
    Position of the shape [q1...q4, x, y, z]
    :param pos2: np.array
    Position of the shape [q1...q4, x, y, z]
    :return: np.array
    Delta position
    """
    q_1 = Quaternion(pos1[0:4])
    q_2 = Quaternion(pos2[0:4])

    p_1 = np.array(pos1[4:-1])
    p_2 = np.array(pos2[4:-1])
    qd2 = q_2 * q_1.conjugate
    p_3 = p_2 - p_1

    outlier = False
    if LIMIT < abs(p_3[0]) or LIMIT < abs(p_3[1]) or LIMIT < abs(p_3[2]):
        outlier = True
    element = np.concatenate((qd2.q, p_3))

    return element, outlier

def plot_histogram(actions):
    """
    Plot histogram
    :param actions: Numpy array
    Array with actions
    :return:
    """
    for i in range(0, actions.T[0].shape[0]):
        counts, bins = np.histogram(actions[i])
        print(hist_dict[i])
        print(counts)
        print(bins)
        plt.hist(actions[i], bins=bins)
        plt.title(hist_dict[i])
        plt.show()

def sort_by_index(names):
    """
    Sort images by index
    :param names: List with names
    :return: List of names sorted by index
    """
    names_dict = dict()

    for name in names:
        if "right" in name:
            index = name.split("right")[1].split(".png")[0]
        else:
            index = name.split("left")[1].split(".png")[0]
        names_dict[int(index)] = name

    keys = list(names_dict.keys())
    keys.sort()
    sorted_images = []
    for key in keys:
        sorted_images.append(names_dict[key])

    return sorted_images

def save_npz(actions, observations, images, json_filename, rollouts_dir, name):
    """
    Saves the .npz files
    :param actions: Array with all actions in rollout
    :param observations: Array with all images in rollout
    :param images: List with image names
    :param json_filename: JSON name to take the name from
    :param rollouts_dir: Output dir for rollouts
    :param name: rollout name
    :return:
    """
    if "test" in images[0]:
        group = "test"
    if "val" in images[0]:
        group = "val"
    if "train" in images[0]:
        group = "train"

    rollout_filename = group + "_" + json_filename.split(".json")[0] + name +".npz"
    np.savez(os.path.join(rollouts_dir, rollout_filename),
             observations=np.array(observations),
             actions=np.array(actions))

def generate_rollout(report_path, rollouts_dir, min_len, plot_hist=True, save_data=False):
    """
    Generate .npz files for training
    :param report_path: path with previously generated json with positions and images
    :param rollouts_dir: dir for saving rollouts
    :param plot_hist: plot histogram for each element of pose vector in complete dataset
    :param save_data: bool
    Save .npz files, use False in case when analyzing the histogramm
    :return:
    """
    json_files = os.listdir(report_path)
    all_actions = []
    outlier = False
    num_of_files = 0
    for json_filename in json_files:
        actions = []
        observations = []
        roll_suffix = -1
        with open(os.path.join(report_path, json_filename)) as json_file:
            data = json.load(json_file)
        print(json_filename)
        images = list(data.keys())
        images.sort()
        images = sort_by_index(images)
        num_of_files += len(images)
        i = 0
        j = 1
        while i + j < len(images) - 1:
            pos1 = np.array(data[images[i]]).astype(float)
            pos2 = np.array(data[images[i + j]]).astype(float)
            action, outlier = calculate_delta(pos1, pos2)
            if outlier:
                outlier = False
                j += 1
                if j > 5:
                    #We want to break this rollout into two rollouts
                    # first we save the current and start from this point
                    if save_data and len(actions) > min_len:
                        roll_suffix += 1
                        save_npz(actions, observations, images, json_filename,
                                 rollouts_dir, str(roll_suffix))
                    actions = []
                    observations = []
                    i += 1
                    j = 1

                continue
            if save_data:
                observations.append(load_image(images[i]))
            actions.append(action)
            all_actions.append(action)
            j = 1
            i += 1

        if save_data and len(actions) > min_len:
            roll_suffix += 1
            save_npz(actions, observations, images, json_filename, rollouts_dir, str(roll_suffix))

        json_file.close()
    if plot_hist:
        act_np = np.array(all_actions)
        plot_histogram(act_np.T)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--params", help="Path to YAML file with parameters",
                   default="params/data_preprocessing.yaml")
    args = a.parse_args()
    with open(args.params_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print(params)
        except yaml.YAMLError as exc:
            print(exc)

    img_path = params["path_resized"]
    xml_path = params["data_path"]
    npz_output = os.path.join(params["out_path"], "data")
    report_path = os.path.join(params["out_path"], "report")
    makedir(npz_output)

    if params["limit"]:
        LIMIT = params["limit"]

    generate_rollout(report_path, npz_output, params["min_rollout_len"], plot_hist=params["plot_histogram"],
                     save_data=params["save_data"])
