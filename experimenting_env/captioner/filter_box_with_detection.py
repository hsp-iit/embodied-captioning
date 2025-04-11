import torch
import pandas as pd
import numpy as np
import glob
import detectron2
from detectron2.structures import Instances, Boxes
import tqdm
import torch
from scipy.special import softmax
import os
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import json
import argparse

def calculate_intersection_area(box_a, box_b):
    """
    Calculate the intersection area of two bounding boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Width and height of the intersection rectangle
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height

def calculate_area(box):
    """
    Calculate the area of a bounding box.
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def find_relevant_boxes(bounding_boxes, overlap_threshold=0.8, area_threshold=8000):
    """
    Find indices of bounding boxes that either:
    1. Have at least `overlap_threshold` overlap with another box.
    2. Have an area less than `area_threshold`.
    The indices are determined after sorting boxes by their area.
    """
    # Calculate areas and sort bounding boxes by area
    boxes_with_area = [(i, box, calculate_area(box)) for i, box in enumerate(bounding_boxes)]
    boxes_with_area.sort(key=lambda x: x[2])  # Sort by area (ascending)

    relevant_indices = set()

    # Process boxes in sorted order
    for idx_a, box_a, area_a in boxes_with_area:
        # Check if the box's area is below the threshold
        if area_a < area_threshold:
            relevant_indices.add(idx_a)
            continue

        # Check for overlap with larger boxes only
        for idx_b, box_b, area_b in boxes_with_area:
            if idx_a == idx_b or area_b <= area_a:
                continue

            intersection_area = calculate_intersection_area(box_a, box_b)
            smaller_box_area = area_a  # Since we're processing smaller boxes first

            # Check overlap condition
            if intersection_area / smaller_box_area >= overlap_threshold:
                relevant_indices.add(idx_a)
                break

    return sorted(relevant_indices)  # Return sorted indices

def read_bbs():
    path = "/home/tommaso/Desktop/Workspace/SImCa/data/psuedolabeler/test"
    file_path_list = os.listdir(path)
    data_list = []
    
    for file_path in tqdm.tqdm(file_path_list):
        data = np.load(os.path.join(path, file_path), allow_pickle=True)['arr_0'].item()
        data_list.append(data)
        
    return data_list

def IoU(box1, box2):
    
    x1, y1, a1, b1 = box1
    x2, y2, a2, b2 = box2
    
    area1 = (a1 - x1)*(b1 - y1)
    area2 = (a2 - x2)*(b2 - y2)
    
    xx = max(x1, x2)
    yy = max(y1, y2)
    aa = min(a1, a2)
    bb = min(b1, b2)
    
    w = max(0, aa - xx)
    h = max(0, bb - yy)
    
    intersection_area = w*h
    
    union_area = area1 + area2 - intersection_area
    
    IoU = intersection_area / union_area
    
    return IoU

def compute_all_ious(bboxes):
    """
    Compute the IoU for every pair of bounding boxes.

    Args:
        bboxes: List of bounding boxes, each defined as [x1, y1, x2, y2].

    Returns:
        A 2D NumPy array containing IoU values.
    """
    n = len(bboxes)
    iou_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Optimize by avoiding redundant computations
            iou = IoU(bboxes[i], bboxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou  # Symmetric

    return iou_matrix

def get_high_iou_pairs(iou_matrix, threshold):
    """
    Get indices offinetune_models_wandb
        iou_matrix: A 2D NumPy array containing IoU values.
        threshold: The IoU threshold.

    Returns:
        A list of tuples, where each tuple contains the indices (i, j) of bounding boxes
        with IoU > threshold.
    """
    pairs = []
    n = iou_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):  # Avoid self-comparison and duplicate pairs
            if iou_matrix[i, j] > threshold:
                pairs.append((i, j))

    return pairs

def predict_new_boxes(img, model, processor):
    inputs = processor(img, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[(1280, 1280)], 
                                                            threshold=0.9, return_binary_maps=True)
    
    class_indices = [56, 57, 58, 59, 60, 61, 62]
    boxes = []
    
    for idx, segment in enumerate(results[0]['segments_info']):
                if segment['label_id'] in class_indices:
                    bb = cv2.boundingRect(results[0]['segmentation'][idx].cpu().numpy().astype('uint8'))
                    x, y, w, h = bb
                    boxes.append([x, y, x + w, y + h])

    return boxes

def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str,
                        default="...",
                        )
    parser.add_argument('--dst_dir',
                        type=str,
                        default="...",
                        )
    return parser.parse_args()

if __name__ == '__main__':
    # Load args
    args = get_args()
    
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance").to('cuda:0')
    
    bbs_path_list = glob.glob(src_dir + '/*.npz')

    # bbs_path_list = ['/home/tommaso/Desktop/Workspace/SImCa/data/psuedolabeler/test/episode_103_step_1.npz']
    
    deleted_box_dict = {}
    
    for bbs_path in tqdm.tqdm(bbs_path_list):
        try:
            bbs = np.load(bbs_path, allow_pickle=True)['arr_0'].item()
        except:
            print("nope")
            continue
        instances = bbs['instances']    
        if len(instances):
            img = bbs['image']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes_tensor = bbs['instances'].pred_boxes.tensor
            bboxes = [box for box in bboxes_tensor]
            
            new_boxes = predict_new_boxes(img, model, processor)
            keep = set()
            deleted_box = []
            for box1 in new_boxes:
                for i, box2 in enumerate(bboxes):
                    if calculate_intersection_area(box1, box2) / min(calculate_area(box1), calculate_area(box2)) > 0.7:
                        keep.add(i)
                    else:
                        if box2.tolist() not in deleted_box:
                            deleted_box.append(box2.tolist())
            
            num_deleted_items = len(bboxes) - len(keep)
            if num_deleted_items > 0:
                print(f"Deleted {num_deleted_items} instances")
                deleted_box_dict[bbs_path] = deleted_box
            
            gt_classes = torch.tensor([instances.gt_classes[i] for i in keep])
            gt_masks =  torch.from_numpy(np.array([np.array(instances.gt_masks[i]) for i in keep]))
            gt_logits =  torch.from_numpy(np.array([np.array(instances.gt_logits[i]) for i in keep]))
            infos = [instances.infos[i] for i in keep]
            pred_boxes = Boxes([instances.pred_boxes[i].tensor[0].tolist() for i in keep])
            episodes = [instances.episode[i] for i in keep]
            embeddings = torch.tensor([instances.embeddings[i].tolist() for i in keep])
            captions = [instances.captions[i] for i in keep]
            new_instances = Instances(image_size = (1280, 1280), gt_classes=gt_classes, gt_masks=gt_masks, gt_logits=gt_logits,
                                    infos=infos, pred_boxes=pred_boxes, episode=episodes, embeddings=embeddings, captions=captions)
            
            bbs['instances'] = new_instances
            bbs['image'] = img
            
        bbs_path = os.path.join(dst_dir, os.path.basename(bbs_path))
        print("Saving {}".format(bbs_path))
        np.savez_compressed(bbs_path, bbs)
    with open('/work/tgalliena/SImCa/deleted_boxes.json', 'w') as f:
        json.dump(deleted_box_dict, f)
    