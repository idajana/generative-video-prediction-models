"""
This script matches
"""

import os
from xml.dom import minidom
import json
import argparse
import yaml
import numpy as np
from utils.dir_utils import makedir


def get_from_xml(xml_index, data_path):
    """
    Reads the endoscope position the XML with the specific index
    :param xml_index: int
        index of XML file
    :param data_path: str
        path to folder with XML files
    :return: str
        the position of the endoscope
    """
    xml_name = "PolarisData" + xml_index + ".xml"
    xml_path = os.path.join(data_path, xml_name)
    xmldoc = minidom.parse(xml_path)
    position = xmldoc.getElementsByTagName('position_Endoscope')
    y_p = position[0].childNodes[0]
    val = y_p.nodeValue
    val = val.replace("\n", "").split("\t")
    val = val[2:-1]
    return val




def map_files(in_dir, xml_dir, out_dir):
    """
    Map the exact filepath to
    :param in_dir: Input images dir (resized images)
    :param xml_dir: The path to Phantomversuch which has the Polaris data and the original images
    :param out_dir: Report_output dir
    :return:
    """

    for dir_type in os.listdir(in_dir):
        subdir1 = os.path.join(in_dir, dir_type)
        rollout_dirs = os.listdir(subdir1)
        for rollout_dir in rollout_dirs:
            dict_left = dict()
            dict_right = dict()
            full_path = os.path.join(subdir1, rollout_dir)
            filenames = os.listdir(full_path)
            filenames.sort()
            print(rollout_dir)

            date, number, _ = rollout_dir.split("_")
            xml_path = os.path.join(xml_dir, date, number)
            for filename in filenames:
                file_path = os.path.join(full_path, filename)
                if "right" in filename:
                    index = filename.split("scene_right")[1].split(".png")[0]
                    pose = get_from_xml(index, xml_path)
                    pos1 = np.array(pose).astype(float)
                    if not pos1[-1]:
                        continue
                    dict_right[file_path] = pose
                elif "left" in filename:
                    index = filename.split("scene_left")[1].split(".png")[0]
                    pose = get_from_xml(index, xml_path)
                    pos1 = np.array(pose).astype(float)
                    if not pos1[-1]:
                        continue
                    dict_left[file_path] = pose
            with open(os.path.join(out_dir, rollout_dir+"_right.json"), 'w') as outfile:
                json.dump(dict_right, outfile)
            with open(os.path.join(out_dir, rollout_dir+"_left.json"), 'w') as outfile2:
                json.dump(dict_left, outfile2)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--params", help="Path to YAML file with parameters",
                   default="params/data_preprocessing_params.yaml")
    args = a.parse_args()
    with open(args.params, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print(params)
        except yaml.YAMLError as exc:
            print(exc)

    img_path = params["path_resized"]
    xml_path = params["data_path"]
    out_path = params["out_path"]
    report_path = os.path.join(out_path, "report")
    makedir(out_path)
    makedir(report_path)

    map_files(img_path, xml_path, report_path)
