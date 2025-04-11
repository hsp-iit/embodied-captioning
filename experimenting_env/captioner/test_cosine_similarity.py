""" This script computes the cosine similarity between a pseudo-caption and other captions."""

import numpy as np
import argparse
import pandas
import os
import torch

from sentence_transformers import SentenceTransformer, util


def calculate_cosine_similarity(captions, model):
    # Encode the captions
    embeddings = model.encode(captions, batch_size=16)

    # Calculate similarity
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    return similarity_matrix


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_file', type=str,
                        default="...",
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print parameters
    print("========================")
    print("captions_file:", captions_file)
    print("arch_name:", arch_name)
    print("visualise:", visualise)
    print("save:", save)
    print("device: ", device)
    print("========================")

    # Read .csv file
    captions_perplexity_ann = pandas.read_csv(captions_file).values
    captions_filename = os.path.basename(captions_file)
    method_name = captions_filename.split("_")[0]
    dataset = captions_filename.split("_")[1]

    # Select correct and wrong captions.
    # The correct ones are the pseudo-captions, while the wrong ones are used for negative learning
    labels = ('Correct', 'Wrong')
    # pseudo_captions = ["A brown table with a vase on it."]
    # wrong_captions = ["A black table with a vase on it.", "A brown bed with a vase on it.", "A brown table with a cup on it.", "A human standing in the snow."]
    pseudo_captions = ["A bed with a red pillow on it."]
    wrong_captions = ["A bed with circular pillow on it", "A sofa with a red pillow on it", "A bed with white linen.",
                      "A horse running in the field"]
    correct_captions = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 1, 1]
    wrong_captions = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 0, 1]
    pseudo_captions = correct_captions[0]
    wrong_captions = wrong_captions[0:5]

    print("pseudo caption: ", pseudo_captions[0])
    for i in range(len(wrong_captions)):
        captions = np.concatenate((pseudo_captions, [wrong_captions[i]]))

        # Compute cosine similarity among BERT embeddings
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

        similarity_matrix = calculate_cosine_similarity(captions, model)

        # Extract the upper triangular part of the matrix, excluding the diagonal
        num_pairs = len(captions)
        upper_tri_indices = np.triu_indices(num_pairs, k=1)
        similarity_values = similarity_matrix[upper_tri_indices]

        # Calculate mean and standard deviation of the similarity scores
        mean_similarity = np.mean(similarity_values)
        std_similarity = np.std(similarity_values)

        # Print the mean and standard deviation
        print("==============", wrong_captions[i])
        print(f'Mean cosine similarity: {mean_similarity:.4f}')
        print(f'Standard deviation of cosine similarity: {std_similarity:.4f}')
