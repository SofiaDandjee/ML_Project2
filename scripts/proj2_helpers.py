import csv
import numpy as np
import pandas as pd


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    
    ##Global mean 
    global_mean = np.sum(train[np.nonzero(train)])/train.nnz
    
    to_predict = test[np.nonzero(test)].todense()
    
    mse = calculate_mse(to_predict, global_mean)
    
    return np.sqrt(mse / test.nnz)


def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    user_mean = np.zeros(num_users)
    
    for i in range (num_users):
        train_i = train[:,i]
        mean = np.mean(train_i[train_i.nonzero()])
        
        test_i = test[:,i]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
    
    return np.sqrt(mse/test.nnz)


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape

    item_mean = np.zeros(num_items)
    
    for i in range (num_items):
        train_i = train[i,:]
        mean = np.mean(train_i[train_i.nonzero()])
        
        test_i = test[i,:]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
    
    return np.sqrt(mse/test.nnz)


def read_csv_sample(name):
    """
    Reads the sample_submission file and extracts the couples (item, user) for which the rating has to be predicted.
    Argument: name (string name of .csv input file)
    """
    items = []
    users = []
    
    with open(name, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)
        for row in rows:
            id = row[0].split("_")
            item, user = id[0], id[1]
            item, user = int(item[1:]), int(user[1:])
            
            items.append(item)
            users.append(user)
            
    return items, users


def create_csv_submission(ids, predictions, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (
               ratings (predicted ratings)
               name (string name of .csv output file to be created)
    """
    #TOMODIFY
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})