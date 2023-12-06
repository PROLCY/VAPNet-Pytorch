import os
import json

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config import Config

# best crop dataset for training(FCDB, GAICD)
class BCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.best_crop_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'best_training_set.json')

        self.image_list, self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        best_crop_bounding_box = self.data_list[index]
        return image_name, best_crop_bounding_box

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        best_crop_list = []
        for data in data_list:
            image_list.append(data['name'])
            best_crop_list.append(data['crop'])
        return image_list, best_crop_list
    
# unlabeled dataset for training(Open Images)
class UnlabledDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unlabeled_training_set.json')

        self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['name']
        suggestion_label = data['suggestion']
        adjustment_label = data['adjustment']
        magnitude_label = data['magnitude']
        return image_name, suggestion_label, adjustment_label, magnitude_label

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        return data_list
    
# Labeled dataset for test(FCDB, GAICD)
class LabledDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = os.path.join(self.cfg.image_dir, 'labeled_vapnet')
        self.dataset_path = self.cfg.labeled_data
        
        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'labeled_testing_set.json')

        self.data_list = self.build_data_list()

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['name']
        image = Image.open(os.path.join(self.image_dir, image_name))
        image_size = torch.tensor(image.size)

        if len(image.getbands()) == 1:
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image, (0, 0, image.width, image.height))
            image = rgb_image
        np_image = np.array(image)
        np_image = cv2.resize(np_image, self.cfg.image_size)
        transformed_image = self.transformer(np_image)
        
        bounding_box = torch.tensor(data['bounding_box']).float()
        perturbed_bounding_box = torch.tensor(data['perturbed_bounding_box']).float()
        suggestion_label = torch.tensor(data['suggestion'])
        adjustment_label = torch.tensor(data['adjustment'])
        magnitude_label = torch.tensor(data['magnitude'])
        return transformed_image, image_size, bounding_box, perturbed_bounding_box, suggestion_label, adjustment_label, magnitude_label

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        return data_list
    
if __name__ == '__main__':
    cfg = Config()
    bc_dataset = BCDataset('train', cfg)
    unlabeled_dataset = UnlabledDataset('train', cfg)
    labeled_dataset = LabledDataset('test', cfg)
    print(bc_dataset.__getitem__(0))
    print(unlabeled_dataset.__getitem__(0))
    print(labeled_dataset.__getitem__(0))