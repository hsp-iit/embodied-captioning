import argparse
import torch.nn.functional as F
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
    parser.add_argument('--arch_name', type=str,
                        default="CoCa",
                        )
    parser.add_argument('--visualise', type=bool,
                        default=False
                        )
    parser.add_argument('--save', type=bool,
                        default=True
                        )
    parser.add_argument('--dest_file', type=str,
                        default="episode0.npy"
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    arch_name = args.arch_name
    visualise = args.visualise
    save = args.save
    dest_file = args.dest_file

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Load model
    if arch_name == "CoCa":
        cfg = Configuration(arch_name='coca', model_name='coca_ViT-L-14',
                            checkpoint_name='mscoco_finetuned_laion2B-s13B-b90k', height=224, width=224)
    elif arch_name == "BLIP2":
        cfg = Configuration(arch_name='blip2', model_name='Salesforce/blip2-opt-2.7b', height=224, width=224)
    model = select_captioner(cfg)
    model.eval()
    model.to(device)

    # Load data
    images_dir = "/home/tapicella/Downloads/temp_data/rgb"
    test_dataset = CaptioningBaseDataset(
        images_dir,
        augmentation=None,
        preprocessing=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logits_dict = {"image_name": [], "object_id": [], "logits": [], "sentence":[]}

    # Inference
    for i, sample_batch in enumerate(tqdm(test_loader)):
        # Load image
        imgs = sample_batch['image']

        # Load to device
        imgs = imgs.to(device)

        # Inference
        outputs = model(imgs)

        # Visualise imagbe and caption
        if visualise:
            image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("RGB", image)
            print("Caption: ", outputs["text"])
            print("Perplexity: ", model.compute_perplexity())
            cv2.waitKey(0)

        if save:
            # Retrieve logits
            logits_temp = outputs["logits"]
            logits = torch.empty((logits_temp[0].shape[0], len(logits_temp), logits_temp[0].shape[1]))
            for ind in range(logits_temp[0].shape[0]):
                for seq in range(len(logits_temp)):
                    logits[ind, seq] = logits_temp[seq][ind]

            # Save logits
            logits_dict["image_name"].append(sample_batch['filename'][0])
            logits_dict["object_id"].append(0)
            logits_dict["logits"].append(logits.cpu().detach().numpy())
            logits_dict["sentence"].append(outputs["text"])
    np.save(dest_file, logits_dict)

