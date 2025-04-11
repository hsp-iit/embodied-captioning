import argparse
import torch
import cv2
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from experimenting_env.captioner.captioning_datasets.base_dataset import CaptioningBaseDataset
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.utils.utils_captioner import select_captioner


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch_name', type=str,
                        default="coca",
                        )
    parser.add_argument('--data_dir', type=str,
                        default="/home/tapicella/Downloads/temp_data/rgb",
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    arch_name = args.arch_name
    data_dir = args.data_dir

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Print parameters
    print("========================")
    print("gpu_id:", gpu_id)
    print("arch_name:", arch_name)
    print("data_dir:", data_dir)
    print("Using ", device)
    print("========================")

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
    test_dataset = CaptioningBaseDataset(
        data_dir,
        augmentation=None,
        preprocessing=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    for i, sample_batch in enumerate(tqdm(test_loader)):
        imgs = sample_batch['image']

        # Load to device
        imgs = imgs.to(device)

        # Inference
        output = model(imgs)

        # Load data
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", image)
        print("Caption: ", output["text"])
        print("Perplexity: ", model.compute_perplexity())
        cv2.waitKey(0)
