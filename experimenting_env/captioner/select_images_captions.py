""" This script reads images and captions and allows the user to select a caption as 'correct', 'wrong', or discard it.
    Results are saved in a .csv file.
"""

import argparse
import cv2
import os
import pandas
import sys

from tqdm import tqdm
from experimenting_env.captioner.utils.utils_file import CsvFile


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa",
                        )
    parser.add_argument('--captions_file', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa/coca_gibson_captions_perplexity.csv",
                        )
    parser.add_argument('--dest_dir', type=str,
                        default="/media/tapicella/Data/data/SImCa_test/Gibson/CoCa"
                        )
    return parser.parse_args()

if __name__ == '__main__':
    # Load args
    args = get_args()
    data_dir = args.data_dir
    captions_file = args.captions_file
    dest_dir = args.dest_dir

    print("Select if the image caption is 1: 'correct', 0: 'wrong' or 3:'discard'(go to the next)")
    print("Press 's' to start, 'e' to end and save results:")
    while True:
        key = input()
        # print(key)
        if key == "e":
            # Exit program
            print("Exiting demo...")
            sys.exit(0)
        elif key != "s":
            print("Press 's' to start:")
            continue
        break

    # Read src .csv file
    src_file = pandas.read_csv(captions_file)
    # Initialise destination .csv file
    dest_file = os.path.join(dest_dir, os.path.basename(captions_file).replace(".csv", "_ann.csv"))
    csv_file = CsvFile()
    if not os.path.exists(dest_file):
        header = ["filename", "caption", "perplexity", "annotation"]
        csv_file.init_header(header, dest_file)

    for i, src_file_row in enumerate(tqdm(src_file.values)):
        if i < 8000:
            continue
        row = None
        # Load filename
        filename = src_file_row[0]
        # Load image
        image = cv2.imread(os.path.join(data_dir, filename))
        # Load caption
        caption = src_file_row[1]
        # Load perplexity
        perplexity = src_file_row[2]
        # Visualise image
        cv2.imshow("Object", image)
        # Visualise caption
        print(caption)
        cv2.waitKey(2000)
        key = input()
        if key == "e":
            # Exit program
            print("Exiting demo...")
            sys.exit(0)
        elif key == "1":
            # Store 'right'
            row = [filename, caption, perplexity, "{}".format(1)]
        elif key == "0":
            # Store 'wrong'
            row = [filename, caption, perplexity, "{}".format(0)]
        else:
            continue
        # Append result to .csv
        csv_file.append_row(row, dest_file)

    print("End of demo!")