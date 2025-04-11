"""  This script performs the following operations:
 - Runs a captioner on images cropped in the annotated boundin boxes
 - Saves the box overlayed on the image, the caption, and the captioner perplexity in a .csv file
The user can select the captioner to use.
"""

import argparse
import os
import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from experimenting_env.captioner.captioning_datasets.detection_dataset import DetectionDataset
from experimenting_env.captioner.test_pseudo_caption_generation import generate_pseudo_caption
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.utils.utils_captioner import select_captioner
from experimenting_env.captioner.utils.utils_file import CsvFile


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch_name', type=str,
                        default="coca",
                        )
    parser.add_argument('--data_name', type=str,
                        default="gibson",
                        )
    parser.add_argument('--data_dir', type=str,
                        default="/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset/gibson_finetuning",
                        )
    parser.add_argument('--save_pseudo_caption', type=bool, default=False)
    parser.add_argument('--dest_dir', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa"
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    arch_name = args.arch_name
    data_name = args.data_name
    data_dir = args.data_dir
    save_pseudo_caption = args.save_pseudo_caption
    dest_dir = args.dest_dir

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Print parameters
    print("========================")
    print("gpu_id:", gpu_id)
    print("arch_name:", arch_name)
    print("data_name:", data_name)
    print("data_dir:", data_dir)
    print("save_pseudo_caption:", save_pseudo_caption)
    print("dest_dir:", dest_dir)
    print("Using ", device)
    print("========================")

    # Load model
    if arch_name == "coca":
        cfg = Configuration(arch_name='coca', model_name='coca_ViT-L-14',
                            checkpoint_name='mscoco_finetuned_laion2B-s13B-b90k', height=224, width=224)
    elif arch_name == "blip2":
        cfg = Configuration(arch_name='blip2', model_name='Salesforce/blip2-opt-2.7b', height=224, width=224)
    model = select_captioner(cfg)
    model.eval()
    model.to(device)

    # Load data
    dataset = DetectionDataset(data_dir)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if save_pseudo_caption:
        # Initialise .csv file
        dest_file = os.path.join(dest_dir, "{}_{}_pseudo_caption.csv".format(arch_name, data_name))
        header = ["episode_id", "object_id", "pseudo_caption"]
        csv_file = CsvFile()
        csv_file.init_header(header, dest_file)

    episode_list = []
    object_list = []
    logits_dict = {}
    captions_dict = {}

    # Inference
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        episode_id = sample_batch["episode_id"].numpy()[0]
        object_id = sample_batch["object_id"].numpy()[0]

        # Generate pseudo caption
        if len(object_list) == 3: # episode_id not in episode_list and len(object_list) != 0:
            print("==========")
            for obj_ind in logits_dict.keys():
                print("Episode_id: {}".format(episode_id))
                print("Object_id: {}".format(obj_ind))
                logits_list = logits_dict[obj_ind]
                probs_list = []
                for s in range(len(logits_list)):
                    print("Caption: {}".format(captions_dict[obj_ind][s]))
                    sentence_logits = torch.stack(logits_list[s], dim=0).squeeze()
                    probs = torch.softmax(sentence_logits, dim=-1)
                    probs_list.append(probs)
                # Generate pseudo-caption
                pseudo_caption = generate_pseudo_caption(probs_list, th=0.25)
                print("Pseudo-caption: ", pseudo_caption)
                row = None
                if save_pseudo_caption:
                    row = [episode_id, object_id, pseudo_caption]
                    csv_file.append_row(row, dest_file)
            object_list = []
            logits_dict = {}
            captions_dict = {}

        if episode_id not in episode_list:
            episode_list.append(episode_id)
        if object_id not in object_list:
            object_list.append(object_id)
            logits_dict[object_id] = []
            captions_dict[object_id] = []

        # Load filename
        filename = sample_batch['filename'][0]
        image_filename = filename[0:25] + ".png"
        # Load image
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load bounding box
        bbox = list(sample_batch['bboxes'].values())[0][0].cpu().detach().numpy()[0].astype(np.int)

        # Crop image
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        image_crop = image[ymin:ymax, xmin:xmax, :]

        # Load to device
        image_crop_tensor = torch.from_numpy(np.expand_dims(image_crop, axis=0))
        image_crop_tensor = image_crop_tensor.to(device)

        # Inference
        output = model(image_crop_tensor)
        logits_dict[object_id].append(output["logits"])
        captions_dict[object_id].append(output["text"])
