import csv
import os

class CsvFile:
    def __init__(self):
        pass

    def init_header(self, header, dest_file):
        # Code modified from https://www.geeksforgeeks.org/writing-csv-files-in-python/
        with open(dest_file, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the fields
            csvwriter.writerow(header)

    def append_row(self, row, dest_file):
        # Code modified from https://www.geeksforgeeks.org/writing-csv-files-in-python/
        with open(dest_file, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the row
            csvwriter.writerow(row)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
