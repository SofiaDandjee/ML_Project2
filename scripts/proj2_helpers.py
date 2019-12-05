import numpy as np
import pandas as pd


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def baseline_global_mean(train, test, ids):
    """baseline method: use the global mean."""
    
    #Global mean
    global_mean = np.sum(train[np.nonzero(train)])/train.nnz  
    to_predict = test[np.nonzero(test)].todense()
    mse = calculate_mse(to_predict, global_mean)
    rmse = np.sqrt(mse / test.nnz)
    
    #Prediction
    predictions = []
    global_mean = int(round(global_mean))
    for i in range(len(ids[0])):
        predictions.append(global_mean)
    
    return rmse, predictions


def baseline_user_mean(train, test, ids):
    """baseline method: use the user means as the prediction."""
    #User means
    mse = 0
    _, num_users = train.shape
    user_means = np.zeros(num_users)
    
    for i in range(num_users):
        train_i = train[:,i]
        mean = np.mean(train_i[train_i.nonzero()])
        user_means[i] = mean
        
        test_i = test[:,i]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
        
    rmse = np.sqrt(mse/test.nnz)
    
    #Prediction
    predictions = []
    for i in range(len(ids[0])):
        user = ids[1][i]
        mean = int(round(user_means[user-1]))
        predictions.append(mean)
        
    return rmse, predictions


def baseline_item_mean(train, test, ids):
    """baseline method: use item means as the prediction."""
    #Item means
    mse = 0
    num_items,_ = train.shape
    item_means = np.zeros(num_items)
    
    for i in range (num_items):
        train_i = train[i,:]
        mean = np.mean(train_i[train_i.nonzero()])
        item_means[i] = mean
        
        test_i = test[i,:]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
        
    rmse = np.sqrt(mse/test.nnz)
    
    #Prediction
    predictions = []
    for i in range(len(ids[0])):
        item = ids[0][i]
        mean = int(round(item_means[item-1]))
        predictions.append(mean)
    
    return rmse, predictions