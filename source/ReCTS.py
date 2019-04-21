import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np 

import utils

class ReCTS(Dataset):
    def __init__(self, data_dir="./data/ReCTS/", dir_list=[1, 2, 3, 4], img_size=(300, 300)):
        self.sub_dirs = ["ReCTS_part{0}".format(i) for i in dir_list]
        self.img_size = img_size
        image_list = []
        label_list = []
        for sub_dir in self.sub_dirs:
            tmp_label_list = []
            tmp_label_list.extend([label if label[:2] != "._" else label[2:] for label in os.listdir(os.path.join(data_dir, sub_dir, "gt"))])
            image_list.extend([os.path.join(data_dir, sub_dir, "img", label[:-5]+".jpg") for label in tmp_label_list])
            label_list.extend([os.path.join(data_dir, sub_dir, "gt", label) for label in tmp_label_list])
        self.label_list = label_list
        self.image_list = image_list
    
    def __len__(self):
        return len(self.label_list)
    
    
    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label_path = self.label_list[idx]
        image = Image.open(image_path)
        w, h = image.size
        image = image.resize(self.img_size)
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        with open(label_path, "r") as f:
            label_json = json.load(f)

        lines = []
        for line in label_json["lines"]:
            points = []
            for i, point in enumerate(line['points']):
                if i % 2 == 0:
                    points.append(point / w * self.img_size[0])
                else:
                    points.append(point / h * self.img_size[1])
            lines.append(points)

        return image, lines

if __name__ == "__main__":
    dataset = ReCTS()
    print(len(dataset))
    dataLoader = DataLoader(dataset, 10, shuffle=True, num_workers=1, collate_fn=utils.detection_collate)
    max_item = 0
    for bacth_data in dataLoader:
        images, lines = bacth_data
        print(images.shape, len(lines[0]))