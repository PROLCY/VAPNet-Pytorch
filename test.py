import os

from shapely.geometry import Polygon
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score as f1
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import LabledDataset
from image_utils.box_perturbation import get_perturbed_box
from vapnet import VAPNet

def build_dataloader(cfg):
    labeled_dataset = LabledDataset('test', cfg)
    data_loader = DataLoader(dataset=labeled_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers)
    return data_loader

class Tester(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = os.path.join(self.cfg.image_dir, 'image_labeled_vapnet')

        self.data_loader = build_dataloader(self.cfg)
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))

        self.batch_size = self.cfg.batch_size

        self.adjustment_count = self.cfg.adjustment_count

        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')

        self.data_length = self.data_loader.__len__()

        self.fpr_limit = self.cfg.fpr_limit

        self.suggestion_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.magnitude_loss_sum = 0

    def run(self, custom_threshold=0):
        print('\n======test start======\n')

        total_gt_suggestion_label = np.array([])
        total_gt_adjustment_label = np.array([])
        total_gt_magnitude_label = np.array([])
        total_predicted_suggestion = np.array([])
        total_predicted_adjustment = np.array([])
        total_predicted_magnitude = np.array([])

        total_gt_perturbed_bounding_box = []
        total_gt_bounding_box = []
        total_image_size = []

        self.model.eval().to(self.device)
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.data_loader), total=self.data_length):
                # data split
                image = data[0].to(self.device)
                image_size = data[1].tolist()
                gt_bounding_box = data[2].tolist()
                gt_perturbed_bounding_box = data[3].tolist()
                gt_suggestion_label = data[4].to(self.device)
                gt_adjustment_label = data[5].to(self.device)
                gt_magnitude_label = data[6].to(self.device)

                # model inference
                predicted_suggestion, predicted_adjustment, predicted_magnitude = self.model(image.to(self.device))

                # caculate loss
                self.suggestion_loss_sum += self.suggestion_loss_fn(predicted_suggestion, gt_suggestion_label)
                self.adjustment_loss_sum += self.adjustment_loss_fn(predicted_adjustment, gt_adjustment_label)
                self.magnitude_loss_sum += self.magnitude_loss_fn(predicted_magnitude, gt_magnitude_label)

                # convert tensor to numpy for using sklearn metrics
                gt_suggestion_label = gt_suggestion_label.to('cpu').numpy()
                gt_adjustment_label = gt_adjustment_label.to('cpu').numpy()
                gt_magnitude_label = gt_magnitude_label.to('cpu').numpy()
                predicted_suggestion = predicted_suggestion.to('cpu').numpy()
                predicted_adjustment = predicted_adjustment.to('cpu').numpy()
                predicted_magnitude = predicted_magnitude.to('cpu').numpy()

                total_gt_suggestion_label = self.add_to_total(gt_suggestion_label, total_gt_suggestion_label)
                total_gt_adjustment_label = self.add_to_total(gt_adjustment_label, total_gt_adjustment_label)
                total_gt_magnitude_label = self.add_to_total(gt_magnitude_label, total_gt_magnitude_label)
                total_predicted_suggestion = self.add_to_total(predicted_suggestion, total_predicted_suggestion)
                total_predicted_adjustment = self.add_to_total(predicted_adjustment, total_predicted_adjustment)
                total_predicted_magnitude = self.add_to_total(predicted_magnitude, total_predicted_magnitude)
                
                total_gt_bounding_box += gt_bounding_box
                total_gt_perturbed_bounding_box += gt_perturbed_bounding_box
                total_image_size += image_size

        # calculate auc, tpr, and threshold for suggestion
        auc_score, tpr_score, threshold = self.calculate_suggestion_accuracy(total_gt_suggestion_label, total_predicted_suggestion)
        if custom_threshold != 0:
            threshold = custom_threshold

        # remove no-suggested elements
        suggested_index = np.where(total_predicted_suggestion >= threshold)[0]

        # get one_hot encoded for predicted adjustment label
        one_hot_predicted_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=total_predicted_adjustment)
        for idx, adjustment in enumerate(one_hot_predicted_adjustment):
            if idx not in suggested_index:
                one_hot_predicted_adjustment[idx] = np.array([0.0] * self.adjustment_count)
        # calculate f1 score for each adjustment
        f1_score = list(self.calculate_f1_score(total_gt_adjustment_label, one_hot_predicted_adjustment))

        # get one_hot encoded for total adjustment label
        one_hot_predicted_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=total_predicted_adjustment)

        # conver index nparray of no-suggestd elements to python list
        suggested_index = list(suggested_index)
        
        # get predicted bounding box
        predicted_bounding_box = []
        
        for index, gt_perturbed_box in enumerate(total_gt_perturbed_bounding_box):
            # no-suggestion case
            if index not in suggested_index:
                predicted_bounding_box.append(gt_perturbed_box)
                continue
        
            adjustment_index = np.where(one_hot_predicted_adjustment[index] == 1.0)[0][0]
            magnitude = total_predicted_magnitude[index][adjustment_index]

            type_index = adjustment_index // 2
            operator = [0.0] * 4
            operator[type_index] = (-1 if adjustment_index % 2 == 0 else 1) * magnitude
            predicted_box = get_perturbed_box(image_size=total_image_size[index], \
                                              bounding_box_corners=gt_perturbed_box, \
                                              operator=operator)

            predicted_bounding_box.append(predicted_box)
            

        # calculate average iou score for each bounding box pairs
        iou_score = self.calculate_ave_iou_score(total_gt_bounding_box, predicted_bounding_box)

        print('\n======test end======\n')

        # calculate ave score
        ave_suggestion_loss = self.suggestion_loss_sum / self.data_length
        ave_adjustment_loss = self.adjustment_loss_sum / self.data_length
        ave_magnitude_loss = self.magnitude_loss_sum  / self.data_length
        
        print(f'threshold:{threshold}')
        with open('./threshold.csv', 'a') as f:
            f.writelines(f'{threshold}\n')
        loss_log = f'{ave_suggestion_loss}/{ave_adjustment_loss}/{ave_magnitude_loss}'
        accuracy_log = f'{auc_score:.5f}/{tpr_score:.5f}/{f1_score}/{iou_score:.5f}'
    
        print(loss_log)
        print(accuracy_log)
        
    def add_to_total(self, target_np_array, total_np_array):
        if total_np_array.shape == (0,):
            total_np_array = target_np_array
        else:
            total_np_array = np.concatenate((total_np_array, target_np_array))
        return total_np_array

    def calculate_suggestion_accuracy(self, gt_suggestion, predicted_suggestion):
        def find_idx_for_fpr(fpr):
            idices = np.where(np.abs(fpr - self.fpr_limit) == np.min(np.abs(fpr - self.fpr_limit)))
            return np.max(idices)

        gt_suggestion = np.array(gt_suggestion).flatten()
        predicted_suggestion = predicted_suggestion.flatten()
        fpr, tpr, cut = roc_curve(gt_suggestion, predicted_suggestion)
        auc_score = auc(fpr, tpr)
        idx = find_idx_for_fpr(fpr)

        tpr_score = tpr[idx]
        threshold = cut[idx]

        return auc_score, tpr_score, threshold
    
    def convert_array_to_one_hot_encoded(self, array):
        largest_value = np.max(array)
        one_hot_encoded = np.zeros_like(array)
        one_hot_encoded[array == largest_value] = 1
        return one_hot_encoded
    
    def calculate_f1_score(self, gt_adjustment, predicted_adjustment):

        def convert_one_hot_encoded_to_index(array):
            if np.all(array == 0):
                return np.array(self.adjustment_count)
            one_hot_index = np.where(array == 1)[0][0]
            return one_hot_index
        
        if len(gt_adjustment) == 0:
            return [0.0] * self.adjustment_count
        
        # one_hot_encoded_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=predicted_adjustment)
        gt_label_list = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=gt_adjustment)
        predicted_label_list = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=predicted_adjustment)
        labels = [i for i in range(0, self.adjustment_count + 1)]
        f1_score = f1(gt_label_list, predicted_label_list, labels=labels, average=None, zero_division=0.0)

        return f1_score
        
    def calculate_ave_iou_score(self, boudning_box_list, perturbed_box_list):
        # box format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (counter-clockwise order)
        def calculate_iou_score(box1, box2):
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            if poly1.intersects(poly2) == False:
                return 0
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area

            iou = intersection_area / union_area if union_area > 0 else 0.0
            return iou
        
        iou_sum = 0
        for i in range(len(boudning_box_list)):
            iou_sum += calculate_iou_score(boudning_box_list[i], perturbed_box_list[i])
        
        ave_iou = iou_sum / len(boudning_box_list)
        return ave_iou

def test_while_training(threshold=0):
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run(custom_threshold=threshold)

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()