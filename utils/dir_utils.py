import os
from shutil import move

def makedir(new_dir):
    """
    :param new_dir: str
    Data with rollouts to sort
    :return:
    """
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

def sort_data(data_dir):
    """
    Sort data to train, val and test directories
    :param data_dir: str
    Data with rollouts to sort
    :return:
    """
    filenames = os.listdir(data_dir)
    makedir(os.path.join(data_dir,"val"))
    makedir(os.path.join(data_dir,"test"))
    makedir(os.path.join(data_dir,"train"))

    for name in filenames:
        if "test_" in name:
            move(os.path.join(data_dir, name), os.path.join(data_dir,"val", name))
        if "val_" in name:
            move(os.path.join(data_dir, name), os.path.join(data_dir,"test", name))
        if "train_" in name:
            move(os.path.join(data_dir, name), os.path.join(data_dir,"train", name))