import torch
import math
import numpy as np 

def detection_collate(batch):
    # 重写组 batch 函数
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def create_defaul_boxes(rates=[1, 2, 3, 5, 7, 10], total_S = 1/2, scales=[37, 19, 10, 5, 3, 1]):
    rate_boxes = []
    for rate in rates:
        h = math.sqrt(total_S / rate)
        w = rate * h
        rate_boxes.append([w, h])
    
    defaul_boxes = None
    for scale in scales:
        scale_boxes = np.zeros((scale, scale, 48))
        cell_w = 1 / scale
        cell_h = 1 / scale
        for h_index in range(scale):
            for w_index in range(scale):
                center_x = (h_index+0.5) * cell_w
                center_y = (w_index+0.5) * cell_h
                for box_index, box in enumerate(rate_boxes):
                    scale_boxes[h_index, w_index, box_index*4:(box_index+1)*4] = center_x, center_y, box[0]/scale, box[1]/scale
                    scale_boxes[h_index, w_index, (box_index+6)*4:(box_index+7)*4] = center_x, center_y+cell_h/2, box[0]/scale, box[1]/scale
        scale_boxes = scale_boxes.reshape(-1, 48)
        if defaul_boxes is None:
            defaul_boxes = scale_boxes
        else:
            defaul_boxes = np.concatenate((defaul_boxes, scale_boxes), axis=0)
    return defaul_boxes

if __name__ == "__main__":
    print(create_defaul_boxes().shape)
    default_boxes = torch.from_numpy(create_defaul_boxes())
    x_box = default_boxes[:,::4]
    print(x_box.shape)
    default_boxes = default_boxes.unsqueeze(0).repeat((10, 1, 1))
    print(default_boxes.shape)