from surprise import NormalPredictor
from proj2_helpers import *
from surprise import accuracy
from surprise import BaselineOnly
from surprise import SVD
from surprise import SVDpp
from surprise import SlopeOne
from surprise import KNNBaseline

def global_mean(train, test, ids):
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

def user_mean(train, test, ids):
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

def item_mean(train, test, ids):
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

def normal_predictor(trainset, ids):
    """
    Generates predictions according to a normal distribution estimated from the training set
    """
    algo = NormalPredictor()
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions

def baseline_only(trainset, ids):
    """
    Combines global mean with user and item biases
    """
    bsl_options = {'method': 'als',
               'n_epochs': 100,
               'reg_u': 15,
               'reg_i': 0.01
               }

    algo = BaselineOnly(bsl_options=bsl_options)
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions

def knn_item(trainset, ids):
    
    bsl_options = {'method': 'als',
               'n_epochs': 100,
               'reg_u': 15,
               'reg_i': 0.01
               }
    
    return

def knn_movies(trainset, ids):
    bsl_options = {'method': 'als',
               'n_epochs': 100,
               'reg_u': 15,
               'reg_i': 0.01
               }

    sim_option = {'name': 'pearson_baseline',
                              'min_support': 1,
                              'user_based': False }

    algo = KNNBaseline(k = 100, bsl_options= bsl_option, sim_options= sim_option)
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions


def svd(trainset, ids):
    """
    Matrix-factorization taking biases into account
    """
    algo = SVD(n_epochs=40, lr_bu=0.0015, lr_bi= 0.05 ,lr_pu=0.01, lr_qi=0.01, reg_bu= 0, reg_bi=0, reg_pu = 0, 
               reg_qi = 0, n_factors = 100 , random_state=30)
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions

def svdpp(trainset, ids):
    algo = SVDpp(n_epochs=40, lr_bu=0.0015, lr_bi= 0.05 ,lr_pu=0.01, lr_qi=0.01, reg_bu= 0, reg_bi=0, reg_pu = 0, 
                 reg_qi = 0, n_factors = 100 , lr_yj = 0, reg_yj= 0, random_state=30)
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions


def slopeone(trainset, ids):
    algo = SlopeOne()
    #Train algorithm on training set
    algo.fit(trainset)
    
    #Predict on test set
    predictions = load_predictions(ids,algo)
    #Compute RMSE on training set
    rmse = train_rmse(trainset, algo)
    return rmse, predictions

