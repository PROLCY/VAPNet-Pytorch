import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from config import Config
from vapnet import VAPNet

class Demo(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

    def inference(self, image_name_list):
        image_list = [Image.open(os.path.join(image_dir_path, image_name)) for image_name in image_name_list]
        output = self.model(self.convert_image_list_to_tensor(image_list))
        return output

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            np_image = np.array(image)
            np_image = cv2.resize(np_image, self.cfg.image_size)
            tensor.append(self.transformer(np_image))
        tensor = torch.stack(tensor, dim=0)
        return tensor

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    model.eval()
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

    demo = Demo(model, cfg)
    image_dir_path = sys.argv[1]
    image_name_list = os.listdir(image_dir_path)

    suggestion_threshold = 0.8

    output_tuple = demo.inference(image_name_list)
    type_list = ['Left', 'Right', 'Up', 'Down', 'Zoom-in', 'Zoom-out', 'Clockwise', 'Counter-Clockwise']
    print(f'Predicted_Result')
    for idx in range(len(output_tuple[0])):
        suggesiton = output_tuple[0][idx].item()
        adjustment = output_tuple[1][idx].tolist()
        magnitude = output_tuple[2][idx].tolist()

        print(f'{image_name_list[idx]}')
        print(f'Model Output\n{suggesiton}\n{adjustment}\n{magnitude}')
        if suggesiton < suggestion_threshold:
            print(f'Not-Needed View Adjustment')
        else:
            adjustment_index = adjustment.index(max(adjustment))
            adjustment_type = type_list[adjustment_index]
            adjustment_magnitude = magnitude[adjustment_index]
            print(f'{adjustment_type}, {adjustment_magnitude}')
        print()

    
