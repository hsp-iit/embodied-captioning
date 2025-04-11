""" This script ...
"""

import argparse
import matplotlib.pyplot as plt
import pandas
import os
import open_clip
import torch


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_file', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa/coca_gibson_captions_perplexity_ann.csv",
                        )
    parser.add_argument('--arch_name', type=str,
                        default="coca",
                        )
    parser.add_argument('--visualise', type=bool,
                        default=False
                        )
    parser.add_argument('--save', type=bool,
                        default=False
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    captions_file = args.captions_file
    arch_name = args.arch_name
    visualise = args.visualise
    save = args.save

    # Print parameters
    print("========================")
    print("captions_file:", captions_file)
    print("arch_name:", arch_name)
    print("visualise:", visualise)
    print("save:", save)
    print("========================")

    # Select tokenizer based on captioner
    if arch_name == "coca":
        tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
    elif arch_name == "blip2":
        tokenizer = ...

    # Read .csv file
    captions_perplexity_ann = pandas.read_csv(captions_file).values
    captions_filename = os.path.basename(captions_file)
    method_name = captions_filename.split("_")[0]
    dataset = captions_filename.split("_")[1]

    # Select correct and wrong captions.
    # The correct ones are the pseudo-captions, while the wrong ones are used for negative learning
    labels = ('Correct', 'Wrong')
    correct_captions = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 1, 1]
    wrong_captions = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 0, 1]
    pseudo_captions = correct_captions[0]
    wrong_captions = wrong_captions[0:5]

    # Extract tokens
    tokenized_correct = tokenizer(pseudo_captions)
    tokenized_wrong = tokenizer(wrong_captions)

    # Compare tokens between right and wrong sets of captions
    correct_list = tokenized_correct
    compared_tensor = []
    for sentence in tokenized_wrong:
        sentence = torch.unsqueeze(sentence, dim=0)
        compared_tensor.append(torch.eq(sentence, correct_list)[0])
    compared_tensor = torch.stack(compared_tensor, dim=0)

    # Keep only negative tokens
    negative_tokens = compared_tensor
    j = 1