""" This script analyses the perplexity in captions previously selected with running the scripts 'text_captioner_perplexity.py' and 'select_images_captions.py'.
    Distribution of captions annotated as 'correct' and 'wrong' are visualised or saved.
"""

import argparse
import matplotlib.pyplot as plt
import pandas
import os


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_file', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa/coca_gibson_captions_perplexity_ann.csv",
                        )
    parser.add_argument('--visualise', type=bool,
                        default=False
                        )
    parser.add_argument('--save', type=bool,
                        default=True
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    captions_file = args.captions_file
    visualise = args.visualise
    save = args.save

    # Read .csv file
    captions_perplexity_ann = pandas.read_csv(captions_file).values
    captions_filename = os.path.basename(captions_file)
    method_name = captions_filename.split("_")[0]
    dataset = captions_filename.split("_")[1]

    labels = ('Correct', 'Wrong')
    correct = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 1, 2]
    wrong = captions_perplexity_ann[captions_perplexity_ann[:, 3] == 0, 2]

    correct = correct[0:30]
    wrong = wrong[0:30]

    bins = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]

    fig, ax = plt.subplots()
    hist_correct = ax.hist(correct, bins=bins, label=labels[0], alpha=0.5)
    hist_wrong = ax.hist(wrong, bins=bins, label=labels[1], alpha=0.5)

    ax.set_ylabel('Density')
    ax.set_xlabel('Perplexity')
    ax.set_ylim([0, 20])
    ax.set_title('{}_{}'.format(dataset, method_name))
    fig.legend()
    fig.tight_layout()

    # Visualise plot
    if visualise:
        plt.show()

    # Save plot
    if save:
        plt.savefig('{}_{}.png'.format(dataset, method_name))
