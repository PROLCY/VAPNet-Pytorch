import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import Config
from image_utils.image_perturbation import get_perturbed_image
from dataset import BCDataset, UnlabledDataset
from test import test_while_training
from vapnet import VAPNet

Image.MAX_IMAGE_PIXELS = None

def not_convert_to_tesnor(batch):
    return batch

def build_dataloader(cfg):
    bc_dataset = BCDataset('train', cfg)
    labeled_loader = DataLoader(dataset=bc_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    unlabeled_dataset = UnlabledDataset('train', cfg)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    return labeled_loader, unlabeled_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir
        self.unlabeled_image_dir = os.path.join(self.image_dir, 'open_images')
        
        self.adjustment_count = self.cfg.adjustment_count

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))

        self.batch_size = self.cfg.batch_size

        self.adjustment_type = self.cfg.adjustment_type_list

        self.bc_loader, self.unlabeled_loader = build_dataloader(self.cfg)

        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')
        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)
        
        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.train_iter = 0
        self.suggested_case_iter = 0
        self.suggestion_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.magnitude_loss_sum = 0

    def training(self):
        print('\n======train start======\n')
        self.model.train().to(self.device)
        bc_iter = iter(self.bc_loader)
        for index, data in enumerate(self.unlabeled_loader):
            self.train_iter += 1
            try:
                bc_data_list = next(bc_iter)
            except:
                bc_iter = iter(self.bc_loader)
            unlabeled_data_list = data

            # get train data for labeled dataset
            l_image_list, l_suggestion_label_list, l_adjustment_label_list, l_magnitude_label_list = self.get_labeled_data_list(bc_data_list)

            # get train data for unlabeld dataset
            ul_image_list = [Image.open(os.path.join(self.unlabeled_image_dir, x[0])).convert('RGB') for x in unlabeled_data_list]
            ul_suggestion_label_list = [x[1] for x in unlabeled_data_list]
            ul_adjustment_label_list = [x[2] for x in unlabeled_data_list]
            ul_magnitude_label_list = [x[3] for x in unlabeled_data_list]

            # combine
            image_list = l_image_list + ul_image_list
            gt_suggestion_list = l_suggestion_label_list + ul_suggestion_label_list
            gt_adjustment_list = l_adjustment_label_list + ul_adjustment_label_list
            gt_magnitude_list = l_magnitude_label_list + ul_magnitude_label_list
            
            # shuffle
            combined_list = list(zip(image_list, gt_suggestion_list, gt_adjustment_list, gt_magnitude_list))
            random.shuffle(combined_list)
            image_list, gt_suggestion_list, gt_adjustment_list, gt_magnitude_list = [list(x) for x in zip(*combined_list)]
            
            # model inference
            predicted_suggestion, predicted_adjustment, predicted_magnitude = self.model(self.convert_image_list_to_tensor(image_list).to(self.device))

            # calculate suggestion loss using BCELoss
            gt_suggestion_list = torch.tensor(gt_suggestion_list).to(self.device)       
            suggestion_loss = self.suggestion_loss_fn(predicted_suggestion, gt_suggestion_list)
            self.suggestion_loss_sum += suggestion_loss.item()

            # filter no suggestion cases
            suggested_index = [i for i in range(len(gt_suggestion_list)) if gt_suggestion_list[i] == 1]
            
            # there are only no suggestion cases
            if len(suggested_index) == 0:
                train_log = f'suggestion loss:{suggestion_loss:.5f}'
                print(train_log)
                self.optimizer.zero_grad()
                suggestion_loss.backward()
                self.optimizer.step()
                continue

            self.suggested_case_iter += 1

            suggested_index = torch.tensor(suggested_index).to(self.device)
            gt_adjustment_list = torch.tensor(gt_adjustment_list).to(self.device)
            gt_magnitude_list = torch.tensor(gt_magnitude_list).to(self.device)

            # remove no-suggestion cases
            predicted_adjustment = torch.index_select(predicted_adjustment, dim=0, index=suggested_index)
            predicted_magnitude = torch.index_select(predicted_magnitude, dim=0, index=suggested_index)
            gt_adjustment_list = torch.index_select(gt_adjustment_list, dim=0, index=suggested_index)
            gt_magnitude_list = torch.index_select(gt_magnitude_list, dim=0, index=suggested_index)

            # calculate adjustment loss using CrossEntropyLoss
            adjustment_loss = self.adjustment_loss_fn(predicted_adjustment, gt_adjustment_list)

            # calculate magnitude loss using L1Loss
            magnitude_loss = self.magnitude_loss_fn(predicted_magnitude, gt_magnitude_list)

            total_loss = suggestion_loss + adjustment_loss + magnitude_loss
            train_log = f'suggestion loss:{suggestion_loss.item():.5f}/adjustment loss:{adjustment_loss.item():.5f}/magnitude loss:{magnitude_loss.item():.5f}/total loss:{total_loss:.5f}'
            print(train_log)

            self.adjustment_loss_sum += adjustment_loss.item()
            self.magnitude_loss_sum += magnitude_loss.item()

            # back propagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        print('\n======train end======\n')

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

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')

            epoch_log = 'epoch: %d / %d, lr: %8f' % (self.epoch, self.max_epoch, self.optimizer.param_groups[0]['lr'])
            print(epoch_log)

            average_suggestion_loss = self.suggestion_loss_sum / self.train_iter
            average_adjustment_loss = self.adjustment_loss_sum / self.suggested_case_iter
            average_magnitude_loss = self.magnitude_loss_sum / self.suggested_case_iter

            train_log = f'ave_suggestion_loss:{average_suggestion_loss}\nave_adjustment_loss:{average_adjustment_loss}\nave_magnitude_loss:{average_magnitude_loss}'
            print(train_log)

            test_while_training()

            self.suggestion_loss_sum = 0
            self.adjustment_loss_sum = 0
            self.magnitude_loss_sum = 0
            self.train_iter = 0
            self.suggested_case_iter = 0

    def get_perturbed_image(self, data):
        image_name = data[0]
        best_crop_bounding_box = data[1]

        image = Image.open(os.path.join(self.image_dir, image_name))
        best_crop_image = image.crop(best_crop_bounding_box)

        suggestion_label = [0.0]
        adjustment_index = -1
        adjustment_label = [0.0] * self.adjustment_count
        magnitude_label = [0.0] * self.adjustment_count

        type_list = self.adjustment_type
        selected_type = random.choice(type_list)
        selected_type_index = type_list.index(selected_type)

        if selected_type == 'best':
            return best_crop_image, suggestion_label, adjustment_label, magnitude_label
        else:
            suggestion_label = [1.0]
            output = get_perturbed_image(image=image, bounding_box=best_crop_bounding_box, allow_zero_pixel=False, type=selected_type)
        
        if output == None:
            return None
        
        perturbed_image, operator = output

        adjustment_index = selected_type_index * 2 if operator[selected_type_index] > 0 else selected_type_index * 2 + 1
        adjustment_label[adjustment_index] = 1.0
        
        if selected_type == 'zoom':
            magnitude_label[adjustment_index] = abs(operator[selected_type_index] / (1 + operator[selected_type_index]))
        else:
            magnitude_label[adjustment_index] = abs(operator[selected_type_index])
        
        return perturbed_image, suggestion_label, adjustment_label, magnitude_label

    def get_labeled_data_list(self, bc_data_list):
        image_list = []
        suggestion_label_list = []
        adjustment_label_list = []
        magnitude_label_list = []

        for data in bc_data_list:
            labeled_data = self.get_perturbed_image(data)
            if labeled_data == None:
                continue
            image_list.append(labeled_data[0])
            suggestion_label_list.append(labeled_data[1])
            adjustment_label_list.append(labeled_data[2])
            magnitude_label_list.append(labeled_data[3])

        return image_list, suggestion_label_list, adjustment_label_list, magnitude_label_list

if __name__ == '__main__':
    cfg = Config()
    
    model = VAPNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()