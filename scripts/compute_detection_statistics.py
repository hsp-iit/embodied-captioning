import argparse
import os
import numpy as np
import ast
import pandas as pd
import albumentations as A
import cv2
import csv
import random
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class SimpleImageCaptioningDataset(Dataset):
    """Custom dataset with image, caption pairs"""

    def __init__(self, annotations_file, margin=10):
        annotations_temp = pd.read_csv(annotations_file)
        self.margin = margin

        # Precompute all the paths to check
        # image_paths = annotations_temp['filename'].apply(lambda x: x.replace(
        #     "/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data"))
        image_paths = annotations_temp['filename'].apply(lambda x: x.replace(
            "/work/tgalliena/SImCa/data/sampled_images", "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset"))

        # Efficiently identify rows with invalid paths
        valid_paths_mask = image_paths.apply(os.path.exists)

        # Filter out rows with invalid paths
        self.annotations = annotations_temp[valid_paths_mask].copy()

        print("Dataset built!")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # "/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data"
        img_path = self.annotations.iloc[idx]['filename'].replace("/work/tgalliena/SImCa/data/sampled_images",
            "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset"
        )
        caption = self.annotations.iloc[idx]['caption']
        episode_id = self.annotations.iloc[idx]['episode_id']
        object_id = self.annotations.iloc[idx]['object_id']

        img_array_original = np.load(img_path)
        # img_array_original = np.load(img_path, allow_pickle=True)['arr_0'].item()['image']
        bb = ast.literal_eval(self.annotations.iloc[idx]['bounding_box'])
        bbox_exp_original = [bb[0] - self.margin if (bb[0] - self.margin) >= 0 else 0,
                             bb[1] - self.margin if (bb[1] - self.margin) >= 0 else 0,
                             bb[2] + self.margin if (bb[2] + self.margin) < img_array_original.shape[0] else (
                                     img_array_original.shape[0] - 1),
                             bb[3] + self.margin if (bb[3] + self.margin) < img_array_original.shape[1] else (
                                     img_array_original.shape[1] - 1)]

        sample = {}
        sample["image_path"] = img_path
        sample["text"] = caption
        sample["episode_id"] = episode_id
        sample["object_id"] = object_id
        sample["img_array_original"] = img_array_original
        sample["bbox_exp_original"] = bbox_exp_original
        return sample


def show_image(img_array_original, bbox_exp_original):
    cv2.imshow("RGB original", cv2.resize(cv2.cvtColor(img_array_original, cv2.COLOR_RGB2BGR),
                                          (img_array_original.shape[1] // 2, img_array_original.shape[0] // 2)))
    img_rect_original = cv2.rectangle(cv2.cvtColor(img_array_original, cv2.COLOR_RGB2BGR),
                                      (int(bbox_exp_original[0]), int(bbox_exp_original[1])),
                                      (int(bbox_exp_original[2]), int(bbox_exp_original[3])), (0, 255, 0), 3)
    cv2.imshow("Bbox original",
               cv2.resize(img_rect_original,
                          (img_rect_original.shape[1] // 2, img_rect_original.shape[0] // 2)))
    cv2.waitKey(0)

def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',
                        type=str,
                        default="/media/tapicella/Data/data/SImCa_test/fine_tuning/val_coca_ens_clip_gibson.csv",
                        )
    parser.add_argument('--threshold_ratio',
                        type=float,
                        default=0.0
                        )
    parser.add_argument('--threshold_area',
                        type=float,
                        default=0.0
                        )
    parser.add_argument('--threshold_side',
                        type=float,
                        default=0.0
                        )
    parser.add_argument('--visualize_ratio',
                        type=bool,
                        default=False
                        )
    parser.add_argument('--visualize_area',
                        type=bool,
                        default=False
                        )
    parser.add_argument('--visualize_min_side',
                        type=bool,
                        default=False
                        )
    parser.add_argument('--visualize_plots',
                        type=bool,
                        default=False
                        )
    parser.add_argument('--save_res',
                        type=bool,
                        default=True
                        )
    parser.add_argument('--save_plots',
                        type=bool,
                        default=True
                        )
    parser.add_argument('--stats_dest_path',
                        type=str,
                        default="stats.csv",
                        )
    parser.add_argument('--fig_dest_path',
                        type=str,
                        default='stats_gibson_mask2former.png',
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    csv_path = args.csv_path
    threshold_ratio = args.threshold_ratio
    threshold_area = args.threshold_area
    threshold_side = args.threshold_side
    visualize_ratio = args.visualize_ratio
    visualize_area = args.visualize_area
    visualize_min_side = args.visualize_min_side
    visualize_plots = args.visualize_plots
    save_plots = args.save_plots
    save_res = args.save_res
    stats_dest_path = args.stats_dest_path
    fig_dest_path = args.fig_dest_path

    dataset = SimpleImageCaptioningDataset(annotations_file=csv_path)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ar_statistics = []

    if save_res:
        with open(stats_dest_path, 'w+', newline='') as csvfile:
            fieldnames = ['name', 'w_h_ar', 'h_w_ar', "area", "min_side"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    w_h = []
    h_w = []
    areas = []
    minimum_side = []

    # Visualise some samples
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        # Load data
        enc_batch = sample_batch
        img_path = enc_batch["image_path"]
        img_array_original = enc_batch["img_array_original"]
        bbox_exp_original = enc_batch["bbox_exp_original"]
        text = enc_batch["text"]
        episode_id = enc_batch["episode_id"]
        object_id = enc_batch["object_id"]

        xmin, ymin, xmax, ymax = bbox_exp_original
        xmin = xmin.item()
        ymin = ymin.item()
        xmax = xmax.item()
        ymax = ymax.item()

        width = xmax - xmin
        height = ymax - ymin

        r = 0
        a = width * height
        min_s = 0
        if width / height <= 1:
            r = width / height
            line = [img_path[0], r, "-", a, min_s]
            w_h.append(r)
            min_s = width
        else:
            r = height / width
            h_w.append(r)
            min_s = height
            line = [img_path[0], "-", r, a, min_s]
        areas.append(a)
        minimum_side.append(min_s)

        if save_res:
            with open(stats_dest_path, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(line)

            # Visualise
            if visualize_ratio or visualize_area or visualize_min_side:
                print(f"{img_path=}")
                img_array_original = img_array_original.detach().numpy()[0]
                show = False
                if r > threshold_ratio:
                    print(f"{r=}")
                    show = True
                if min_s > threshold_side:
                    print(f"{min_s=}")
                    how = True
                if a > threshold_area:
                    print(f"{a=}")
                    show = True
                if show:
                    show_image(img_array_original, [xmin, ymin, xmax, ymax])

    w_h = np.asarray(w_h)
    h_w = -np.asarray(h_w)
    ratios = np.concatenate([w_h, h_w], axis=0)
    areas = np.asarray(areas)
    minimum_side =np.asarray(minimum_side)
    fig, ax = plt.subplots(nrows=3,ncols=1, tight_layout=True)
    ax[0].hist(ratios, bins=50)
    ax[0].set_title("Aspect ratios: -h/w, +w/h")
    ax[1].hist(areas, bins=50)
    ax[1].set_title("Areas [pixels]")
    ax[2].hist(minimum_side, bins=50)
    ax[2].set_title("Minimum side [pixels]")
    if visualize_plots:
        plt.show()
    if save_plots:
        plt.savefig(fig_dest_path)
