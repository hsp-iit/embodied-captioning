"""  This script performs the following operations:
 - Runs a captioner on images cropped in the annotated bounding boxes
 - Saves the box overlayed on the image, the caption, and the captioner perplexity in a .csv file
The user can select the captioner to use.
"""

import argparse
import os
import torch
import cv2
import numpy as np
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
from experimenting_env.captioner.captioning_datasets.detection_dataset import DetectionDataset
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.utils.utils_captioner import select_captioner
from experimenting_env.captioner.utils.utils_file import CsvFile, make_dir


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch_name', type=str,
                        default="coca",
                        )
    parser.add_argument('--data_name', type=str,
                        default="replica",
                        )
    parser.add_argument('--data_dir', type=str,
                        default="/media/tapicella/Win11_OS/Users/tapicella/Downloads/replica_sim",
                        # default="/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset/gibson_finetuning",
                        )
    parser.add_argument('--visualise_overlay', type=bool, default=False)
    parser.add_argument('--save_overlay', type=bool, default=True)
    parser.add_argument('--save_caption', type=bool, default=True)
    parser.add_argument('--save_perplexity', type=bool, default=True)
    parser.add_argument('--dest_dir', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Replica/CoCa"
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    arch_name = args.arch_name
    data_name = args.data_name
    data_dir = args.data_dir
    visualise_overlay = args.visualise_overlay
    save_overlay = args.save_overlay
    save_caption = args.save_caption
    save_perplexity = args.save_perplexity
    dest_dir = args.dest_dir

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Print parameters
    print("========================")
    print("gpu_id:", gpu_id)
    print("arch_name:", arch_name)
    print("data_name:", data_name)
    print("data_dir:", data_dir)
    print("visualise_overlay:", visualise_overlay)
    print("save_overlay:", save_overlay)
    print("save_caption:", save_caption)
    print("save_perplexity:", save_perplexity)
    print("dest_dir:", dest_dir)
    print("Using ", device)
    print("========================")

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # Load model
    if arch_name == "coca":
        cfg = Configuration(arch_name='coca', model_name='coca_ViT-L-14',
                            checkpoint_name='mscoco_finetuned_laion2B-s13B-b90k', height=224, width=224)
    elif arch_name == "blip2":
        cfg = Configuration(arch_name='blip2', model_name='Salesforce/blip2-opt-2.7b', height=224, width=224)
    model = select_captioner(cfg.captioner)
    model.eval()
    model.to(device)

    # Load data
    dataset = DetectionDataset(data_dir)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialise .csv file
    if save_perplexity or save_caption:
        dest_file = os.path.join(dest_dir, "{}_{}_captions_perplexity.csv".format(arch_name, data_name))
        header = ["filename", "caption", "perplexity"]
        csv_file = CsvFile()
        csv_file.init_header(header, dest_file)

    if save_overlay:
        make_dir(os.path.join(dest_dir, "vis"))

    # Inference
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        # if i < 34613:
        #     continue
        # Load filename
        filename = sample_batch['filename'][0]

        # Don't process frames without detected objects
        if sample_batch['object_id'] == -1:
            # print(filename)
            continue

        # Load image
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Load bounding box
        bboxes = list(sample_batch['bboxes'].values())

        # Crop image
        for bbox_ind, bbox in enumerate(bboxes):
            xmin, ymin, width, height = bbox[0].cpu().detach().numpy()[0].astype(int)
            xmin = int(xmin)
            ymin = int(ymin)
            width = int(width)
            height = int(height)
            xmax = xmin + width
            ymax = ymin + height
            # if xmin > xmax:
            #     print(filename)
            #     continue
            image_crop = image[ymin:ymax, xmin:xmax, :]

            # Load to device
            image_crop_tensor = torch.from_numpy(np.expand_dims(image_crop, axis=0))
            image_crop_tensor = image_crop_tensor.to(device)

            # Inference
            output = model(image_crop_tensor)
            caption = output["text"]
            perplexity = model.compute_perplexity().item()

            if visualise_overlay:
                cv2.imshow("Crop", image_crop)
                overlay_vis = image.copy()
                start_point = (xmin, ymin)
                end_point = (xmax, ymax)
                color = (0, 0, 255)
                thickness = 2
                overlay_vis = cv2.rectangle(overlay_vis, start_point, end_point, color, thickness)
                cv2.imshow("Bounding box", overlay_vis)
                print("Caption: ", output["text"])
                print("Perplexity: ", perplexity)
                cv2.waitKey(0)

            row = None
            image_filename = (filename[0:25] + "_{}".format(bbox_ind) + ".png")
            if save_caption and save_perplexity:
                row = [image_filename, caption, "{}".format(perplexity)]
            elif save_perplexity:
                row = [image_filename, "{}".format(-1), "{}".format(perplexity)]
            elif save_caption:
                row = [image_filename, caption, "{}".format(-1)]

            if save_caption or save_perplexity:
                csv_file.append_row(row, dest_file)

            if save_overlay:
                overlay_vis = image.copy()
                start_point = (xmin, ymin)
                end_point = (xmax, ymax)
                color = (0, 0, 255)
                thickness = 2
                overlay_vis = cv2.rectangle(overlay_vis, start_point, end_point, color, thickness)
                cv2.imwrite(os.path.join(dest_dir, "vis", image_filename), overlay_vis)
