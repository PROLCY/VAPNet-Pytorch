import math
import os
import sys

import json
from PIL import Image
import torch
import tqdm

from config import Config
from CSNet.csnet import CSNet
from CSNet.csnet_demo import CSNetDemo
from image_utils.box_perturbation import get_perturbed_box
from image_utils.image_perturbation import get_perturbed_image

Image.MAX_IMAGE_PIXELS = None

cfg = Config()

# get CSNet demo for score inference
model = CSNet()
model.eval()
model.to(torch.device('cuda:0'))
weight_file = os.path.join('./CSNet/weight', 'checkpoint-weight.pth')
model.load_state_dict(torch.load(weight_file))
csnet_demo = CSNetDemo(model)

def make_pseudo_label(image_path):
    
    image = Image.open(image_path).convert('RGB')
    image_name = image_path.split('/')[-1]

    left_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    right_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]

    up_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    down_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]

    zoom_in_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    zoom_out_magnitude = [x * 0.05 for x in range(1, 10, 1)]

    clockwise_magnitude = [-x * math.pi / 36 for x in range(1, 10, 1)]
    counter_clokwise_magnitude = [x * math.pi / 36 for x in range(1, 10, 1)]

    candidate_magnitude_list_by_each_adjustment = [left_shift_magnitude, \
                                                right_shift_magnitude, \
                                                up_shift_magnitude, \
                                                down_shift_magnitude, \
                                                zoom_in_magnitude, \
                                                zoom_out_magnitude, \
                                                clockwise_magnitude, \
                                                counter_clokwise_magnitude \
                                                ]
    
    pseudo_data_list = []
    adjustment_label_list = []
    magnitude_label_list = []
    perturbed_image_list = []

    adjustment_type_list = cfg.adjustment_type_list
    
    for index, candidate_magnitude_list in enumerate(candidate_magnitude_list_by_each_adjustment):
        adjustment_label = [0.0] * len(candidate_magnitude_list_by_each_adjustment)
        adjustment_label[index] = 1.0

        for mag in candidate_magnitude_list:

            mag = round(mag, 2)
            magnitude_label = [0.0] * len(adjustment_label)
            magnitude_label[index] = abs(mag)
            
            adjustment_type = adjustment_type_list[index // 2]
            pseudo_image, operator = get_perturbed_image(image=image, bounding_box=[0, 0, image.size[0], image.size[1]], allow_zero_pixel=True, type=adjustment_type, magnitude=mag)

            perturbed_image_list.append(pseudo_image)
            adjustment_label_list.append(adjustment_label)
            magnitude_label_list.append(magnitude_label)

    score_list = csnet_demo.inference(perturbed_image_list).tolist()
    pseudo_data_list = [(x[0], y, z, img) for x, y, z, img in zip(score_list, adjustment_label_list, magnitude_label_list, perturbed_image_list)]
    
    # sort in desceding order by csnet score
    pseudo_data_list.sort(reverse=True)

    original_image_score = csnet_demo.inference([image])[0].item()
    best_adjustment_label = pseudo_data_list[0]
    best_adjustment_score = best_adjustment_label[0]

    # get perturbed image as dataset
    if original_image_score + 0.2 < best_adjustment_score:
        return {
            'name': image_name,
            'suggestion': [1.0],
            'adjustment': best_adjustment_label[1],
            'magnitude': best_adjustment_label[2]
        }
    # get original image as dataset
    else:
        return {
            'name': image_name,
            'suggestion': [0.0],
            'adjustment': [0.0] * len(candidate_magnitude_list_by_each_adjustment),
            'magnitude': [0.0] * len(candidate_magnitude_list_by_each_adjustment)
        }
    
def make_annotations_for_unlabeled(image_list, image_dir_path, annotation_path):

    annotation_list = []

    pertubed_cnt = 0
    no_perturbed_cnt = 0
    adjustment_cnt = [0] * cfg.adjustment_count
    
    for image_name in tqdm.tqdm(image_list):
        image_path = os.path.join(image_dir_path, image_name)
        try:
            annotation = make_pseudo_label(image_path)
            if annotation['suggestion'] == [1.0]:
                pertubed_cnt += 1
                adjustment_cnt[annotation['adjustment'].index(1.0)] += 1
            else:
                no_perturbed_cnt += 1
            annotation_list.append(annotation)
            with open(os.path.join(annotation_path, 'unlabeled_training_set.csv'), 'a') as f:
                f.writelines(f'{annotation}\n')
        except Exception as e:
            print(f'Exception while processing {image_name}')
            print(e)
    
    print(f'perturbed_cnt:{pertubed_cnt}')
    print(f'cnt_by_adjustment_label:{adjustment_cnt}')
    print(f'no-perturbed_cnt:{no_perturbed_cnt}')
    
    print('Start saving annotations...')
    with open(os.path.join(annotation_path, 'unlabeled_training_set.json'), 'w') as f:
        json.dump(annotation_list, f, indent=2)
    print('Annotations Saved...')

    return

def perturbing_for_labeled_data(image, bounding_box, box_corners, type):
    output = None
    for i in range(0, 100, 1):
        output = get_perturbed_image(image, bounding_box, allow_zero_pixel=False, type=type)
        if output != None:
            break
    if output == None:
        return None
    perturbed_image, operator = output

    perturbed_box_corners = get_perturbed_box(image.crop(bounding_box).size, box_corners, operator)

    adjustment_label = [0.0] * cfg.adjustment_count
    magnitude_label = [0.0] * cfg.adjustment_count

    adjustment_index = -1

    for idx, mag in enumerate(operator):
        if mag != 0:
            selected_operator_index = idx
            adjustment_index = idx * 2 if mag > 0 else idx * 2 + 1
            break

    adjustment_label[adjustment_index] = 1.0
    if type == 'zoom':
        magnitude_label[adjustment_index] = abs(operator[selected_operator_index] / (1 + operator[selected_operator_index]))
    else:
        magnitude_label[adjustment_index] = abs(operator[selected_operator_index])

    return perturbed_image, perturbed_box_corners, adjustment_label, magnitude_label

def make_annotation_for_labeled(image_dir_path, image_name, bounding_box):
    image_path = os.path.join(image_dir_path, image_name)
    image = Image.open(image_path)
    image_name = image_name.split('.')[0]

    annotation_list = []

    # generate no-suggestion case
    best_crop = image.crop(bounding_box)
    best_crop.save(os.path.join(image_dir_path, image_name + f'_-1.jpg'))

    box_corners = [
        [bounding_box[0], bounding_box[1]],
        [bounding_box[0], bounding_box[3]],
        [bounding_box[2], bounding_box[3]],
        [bounding_box[2], bounding_box[1]],
    ]

    annotation_list.append({
        'name': image_name + f'_-1.jpg',
        'bounding_box': box_corners,
        'perturbed_bounding_box': box_corners,
        'suggestion': [0.0],
        'adjustment': [0.0] * cfg.adjustment_count,
        'magnitude': [0.0] * cfg.adjustment_count
    })

    type_list = ['horizontal_shift', 'vertical_shift', 'zoom', 'rotate']
    for idx, type in enumerate(type_list):
        for i in range(2):
            output = perturbing_for_labeled_data(image, bounding_box, box_corners, type)
            if output == None:
                continue
            perturbed_image = output[0]
            perturbed_box_corners = output[1]
            adjustment_label = output[2]
            magnitude_label = output[3]
            adjustment_index = adjustment_label.index(max(adjustment_label))
            perturbed_image_name = image_name + f'_{adjustment_index}_{i}.jpg'
            annotation = {
                'name': perturbed_image_name,
                'bounding_box': box_corners,
                'perturbed_bounding_box': perturbed_box_corners,
                'suggestion': [1.0],
                'adjustment': adjustment_label,
                'magnitude': magnitude_label
            }
            perturbed_image.save(os.path.join(image_dir_path, perturbed_image_name))
            annotation_list.append(annotation)
    
    return annotation_list

def make_annotations_for_labeled(best_crop_annotation_list, image_dir_path, annotation_path):
    annotation_list = []

    pertubed_cnt = 0
    no_perturbed_cnt = 0
    adjustment_cnt = [0] * cfg.adjustment_count

    for data in tqdm.tqdm(best_crop_annotation_list):
        try:
            image_name = data['name']
            bounding_box = data['crop']
            annotation_for_one_image = make_annotation_for_labeled(image_dir_path, image_name, bounding_box)
            annotation_list += annotation_for_one_image

            for annotation in annotation_for_one_image:
                if annotation['suggestion'] == [1.0]:
                    pertubed_cnt += 1
                    adjustment_cnt[annotation['adjustment'].index(1.0)] += 1
                else:
                    no_perturbed_cnt += 1
        except Exception as e:
            print(f'Exception while processing {image_name}')
            print(e)

    print(f'perturbed_cnt:{pertubed_cnt}')
    print(f'cnt_by_adjustment_label:{adjustment_cnt}')
    print(f'no-perturbed_cnt:{no_perturbed_cnt}')

    print('Start saving annotations...')
    with open(os.path.join(annotation_path, 'labeled_testing_set.json'), 'w') as f:
        json.dump(annotation_list, f, indent=2)
    print('Annotations Saved...')

    return

if __name__ == '__main__':
    option = sys.argv[1]
    if option == '-u':
        image_dir_path = sys.argv[2]
        annotation_path = sys.argv[3]
        image_list = os.listdir(image_dir_path)
        make_annotations_for_unlabeled(image_list, image_dir_path, annotation_path)
    if option == '-l':
        best_crop_annotation_path = sys.argv[2]
        with open(best_crop_annotation_path, 'r') as f:
            best_crop_annotation_list = json.load(f)
        image_dir_path = sys.argv[3]
        annotation_path = sys.argv[4]
        make_annotations_for_labeled(best_crop_annotation_list, image_dir_path, annotation_path)