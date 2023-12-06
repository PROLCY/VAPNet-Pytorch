import os

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .csnet import CSNet

class CSNetDemo(object):
    def __init__(self, model):
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.image_size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def inference(self, image_list):
        score = self.model(self.convert_image_list_to_tensor(image_list).to(self.device))
        return score


    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            np_image = np.array(image)
            np_image = cv2.resize(np_image, self.image_size)
            tensor.append(self.transformer(np_image))
        tensor = torch.stack(tensor, dim=0)
        return tensor

if __name__ == '__main__':

    model = CSNet()
    model.eval()
    weight_file = os.path.join('./weight' 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

    demo = CSNetDemo(model)