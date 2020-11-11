import os
import argparse
import pdb
import numpy as np
import yaml
from utils.dir_utils import makedir, sort_data

def find_min_max(data_dir):
    """
    :param data_dir: str
    Directory with npz rollouts
    :return: float, float
    Minimum and maximum value of position
    """
    max_q = []
    min_q = []
    for filename in os.listdir(data_dir):
        try:
            data = np.load(os.path.join(data_dir, filename))
        except ValueError:
            print(filename)
        actions = data["actions"]
        try:
            max_q.append(np.max(actions[:, :], axis=0))
            min_q.append(np.min(actions[:, :], axis=0))
        except IndexError:
            print(actions.shape)
            os.remove(os.path.join(data_dir, filename))
        print(filename)
    maximum = np.max(np.array(max_q), axis=0)
    minimum = np.min(np.array(min_q), axis=0)
    print("Maximum: {}".format(np.max(maximum[4:])))
    print("Minimum: {}".format(np.min(minimum[4:])))
    return np.min(minimum[4:]), np.max(maximum[4:])

def normalize(minimum, maximum, value):
    """
    Normalize the array
    """
    return 2*(value-minimum)/(maximum-minimum)-1

def data_norm(data_dir, out_dir, minimum, maximum, min_len):
    """
    Normalize the data
    :param data_dir: str
    Directory with npz rollouts
    :param out_dir: str
    Directory with normalized npz rollouts
    :param minimum: float
    minimum value of pose
    :param maximum: float
    maximum value of pose
    :return:
    """
    for filename in os.listdir(data_dir):
        try:
            data = np.load(os.path.join(data_dir, filename))
        except ValueError:
            print(filename)
        obser = data["observations"]
        unnorm_actions = data["actions"]

        print(filename)
        try:
            norm_pos = normalize(minimum, maximum, unnorm_actions[:, 4:])
            quat = unnorm_actions[:, :4]
            actions = np.concatenate([np.array(quat), np.array(norm_pos)], axis=1)
            if actions.shape[0] > min_len:
                np.savez(os.path.join(out_dir, filename), observations=obser, actions=actions)
            else:
                print("Skipped")
                print(actions.shape)
        except IndexError:
            pdb.set_trace()

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
    data_path = os.path.join(params["out_path"], "data")
    norm_data_path = os.path.join(params["out_path"], "rollouts_norm")
    min_len = params["min_rollout_len"]
    min_val, max_val = find_min_max(data_path)
    makedir(norm_data_path)
    data_norm(data_path, norm_data_path, min_val, max_val, min_len)
    sort_data(norm_data_path)
