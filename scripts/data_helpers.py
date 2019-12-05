import csv
import numpy as np
import scipy.sparse as sp


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def read_csv_sample(path):
    """
    Reads the sample_submission file and extracts the couples (item, user) for which the rating has to be predicted.
    Argument: name (string name of .csv input file)
    """
    
    items = []
    users = []
    
    data = read_txt(path)[1:]
    
    for line in data:
        item, user, _ = deal_line(line)
        items.append(item)
        users.append(user)
     
    return [items, users]


def create_csv_submission(ids, predictions, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (
               ratings (predicted ratings)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2, r3 in zip(ids[0], ids[1], predictions):
            writer.writerow({'Id':'r' + str(r1) + '_c' + str(r2),'Prediction':r3})