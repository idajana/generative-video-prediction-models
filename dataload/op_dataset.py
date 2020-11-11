import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def join_function(file_tuple):
    return os.path.join(file_tuple[0], file_tuple[1])


def get_image_names(input_path):
    dir_by_tryout = os.listdir(input_path)
    image_names = []
    for date_dir in dir_by_tryout:
        file_path = os.path.join(input_path, date_dir)
        filenames = os.listdir(file_path)
        filepath_vec = [file_path]*len(filenames)
        filenames_wpath = list(map(join_function, zip(filepath_vec, filenames)))
        image_names += filenames_wpath
    return image_names

class OPDataset(Dataset):
    """Dataset from surgeries."""

    def __init__(self, input_dir, transform=None):

        self.input_dir = input_dir
        self.image_names = get_image_names(input_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image



