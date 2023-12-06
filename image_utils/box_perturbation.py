import math

from .image_perturbation import rotate
    
def get_shifted_box(image_size, bounding_box_corners, operator):

    if operator[0] != 0:
        mag = image_size[0] * operator[0]
        min_x = bounding_box_corners[0][0] + mag
        max_x = bounding_box_corners[2][0] + mag
        min_y = bounding_box_corners[0][1]
        max_y = bounding_box_corners[2][1]

    elif operator[1] != 0:
        mag = image_size[1] * operator[1]
        min_x = bounding_box_corners[0][0]
        max_x = bounding_box_corners[2][0]
        min_y = bounding_box_corners[0][1] + mag
        max_y = bounding_box_corners[2][1] + mag
    
    new_box_corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
    ]

    return new_box_corners

def get_zoomed_box(image_size, bounding_box_corners, operator):

    x_mag = image_size[0] * operator[2] / 2
    y_mag = image_size[1] * operator[2] / 2

    min_x = bounding_box_corners[0][0] - x_mag
    max_x = bounding_box_corners[2][0] + x_mag
    min_y = bounding_box_corners[0][1] - y_mag
    max_y = bounding_box_corners[2][1] + y_mag

    new_box_corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
    ]

    return new_box_corners

def get_rotated_box(bounding_box_corners, operator):

    cx = (bounding_box_corners[0][0] + bounding_box_corners[2][0]) / 2
    cy = (bounding_box_corners[0][1] + bounding_box_corners[2][1]) / 2
    w = (bounding_box_corners[2][0] - bounding_box_corners[0][0])
    h = (bounding_box_corners[2][1] - bounding_box_corners[0][1])

    oa = operator[3]
    new_box_corners = rotate([cx, cy, w, h], oa)

    return new_box_corners

def get_perturbed_box(image_size, bounding_box_corners, operator):
    if operator[0] != 0 or operator[1] != 0:
        return get_shifted_box(image_size, bounding_box_corners, operator)
    elif operator[2] != 0:
        return get_zoomed_box(image_size, bounding_box_corners, operator)
    elif operator[3] != 0:
        return get_rotated_box(bounding_box_corners, operator)

if __name__ == '__main__':
    bounding_box = [100, 100, 400, 400]

    box_size = [bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]]

    bounding_box_corners = [
        [bounding_box[0], bounding_box[1]],
        [bounding_box[0], bounding_box[3]],
        [bounding_box[2], bounding_box[3]],
        [bounding_box[2], bounding_box[1]],   
    ]

    operator = [0.0, 0.0, 0.0, math.pi / 4]
    rotated_box = get_rotated_box(bounding_box_corners, operator)

    operator = [0.2, 0.0, 0.0, 0.0]
    horizontal_shifted_box = get_shifted_box(box_size, bounding_box_corners, operator)

    operator = [0.0, -0.25, 0.0, 0.0]
    vertical_shifted_box = get_shifted_box(box_size, bounding_box_corners, operator)

    operator = [0.0, 0.0, 0.05, 0.0]
    zoomed_box = get_zoomed_box(box_size, bounding_box_corners, operator)

    print(rotated_box)
    print(horizontal_shifted_box)
    print(vertical_shifted_box)
    print(zoomed_box)