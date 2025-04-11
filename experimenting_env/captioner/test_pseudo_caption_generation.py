"""  This script performs the following operations:
 - Loads a set of captioner outputs
 - Computes the pseudo-caption
The user can select the captioner to use.
"""

import argparse
import os

import open_clip
import torch
import cv2
import numpy as np

from tqdm import tqdm


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--logits_path', type=str,
                        default="/media/tapicella/Data/code/SImCa/episode0.npy",
                        )
    return parser.parse_args()


def compute_max_tokens_probability(probs):
    # Retrieve maximum probability in for each token in the sentence
    return torch.max(probs, dim=0)[0]


def compute_average_tokens_probability(token_probs):
    # Compute tokens probability average
    probs_sum = torch.sum(token_probs, dim=0)
    a = probs_sum.cpu().detach().numpy()
    num_el = token_probs.shape[0]
    return probs_sum / num_el


def generate_pseudo_caption(probs, th):
    """ Pseudo caption generation from probability tensor.
    Args:
        probs: probabilities tensors of each word in a captions set
    """
    token_probs = []
    for s in range(len(probs)):
        # Compute token probabilities
        token_probs.append(compute_max_tokens_probability(probs[s]))

    # Compute pseudo-caption tokens
    token_probs = torch.stack(token_probs, dim=0)
    # a = token_probs.cpu().detach().numpy()
    pseudo_tokens = compute_average_tokens_probability(token_probs)
    # a = pseudo_tokens.cpu().detach().numpy()

    # Threshold probabilities
    pseudo_tokens = torch.where(pseudo_tokens > th)[0]

    # Convert tokens into words
    pseudo_caption = open_clip.decode(pseudo_tokens).split("<end_of_text>")[0].replace("<start_of_text>", "")

    return pseudo_caption


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    logits_path = args.logits_path

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Print parameters
    print("========================")
    print("gpu_id:", gpu_id)
    print("data_dir:", logits_path)
    print("Using ", device)
    print("========================")

    # Load captions
    caption_dict = np.load(logits_path, allow_pickle=True).item()
    logits = caption_dict["logits"]
    sentences = caption_dict["sentence"]
    probs = []
    for s in range(len(logits)):
        print("Sentence {}: {}".format(s, sentences[s]))
        sentence_logits = torch.from_numpy(logits[s][0])
        probs.append(torch.softmax(sentence_logits, dim=-1))

    # Generate pseudo-caption
    pseudo_caption = generate_pseudo_caption(probs, th=0.5)

    # Visualise pseudo-caption
    print("Pseudo caption: ", pseudo_caption)
