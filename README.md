# The EPFL Recommender System Challenge 2019
Machine Learning - Project 2

**`Tshtsh_club`**: Marie Anselmet, Sofia Dandjee, Héloïse Monnet

## Install the surprise library

~~~~shell
pip install scikit-surprise
~~~~

OR

~~~~shell
conda install scikit-surprise
~~~~

## Run the project
1. Make sure that ```Python >= 3.7```, ```NumPy >= 1.16``` and ```sklearn >= 0.22``` are installed
2. Go to `script\` folder and run ```run.py```. You will get ```submission.csv``` for Kaggle in the ```data\``` folder.

~~~~shell
cd script
python run.py
~~~~

## Script files

### ```proj2_helpers.py```

- `calculate_mse`: Computes mean-squared error between predicted and known ratings
- `calculate_rmse`: Computes root-mean-squared error between predicted and known ratings

### ```data_helpers.py```

- `read_csv_sample` : Reads the sample_submission file and extracts the couples (item, user) for which the rating has to be predicted.
- `create_csv_submission`: Creates an output file in csv format for submission to kaggle
- `build_surprise_data`: Loads the training and test set for it to be usable by the surprise library
- `split_data`: Loads the training and test set that are used for the custom models

### ```implementations.py```

- `global_mean`: Use the global mean as the prediction.
- `user_mean`: Use the user means as the prediction.
- `item_mean`: Use the item means as the prediction.
- `normal_predictor`: Generates predictions according to a normal distribution estimated from the training set.
- `baseline_only`: Combines user and item mean with user and item biases.
- `knn_baseline_user`: Nearest neighbour approach between users taking into account baseline ratings.
- `knn_baseline_movie`: Nearest neighbour approach between movies taking into account baseline ratings.
- `svd`: Matrix factorization algorithm taking biases into account.
- `svdpp`: Extension of svd taking into account implicit ratings.
- `slopeone`: Item-based algorithm based on similarity between users that rated the same movie.
- `nmf`: Non-negative matrix factorization.
- `blending`: Computes a ridge regression to find optimal weights for each of the fed models
- `co_clustering`: Users and items are assigned to clusters and co-clusters.
- `knn_centered_user`: Nearest neighbour approach taking into account the mean ratings of each user
- `knn_centered_movie`: Nearest neighbour approach taking into account the mean ratings of each movie
- `knn_basic_user`: Nearest neighbour basic approach on users
- `knn_basic_movie`: Nearest neighbour basic approach on movies

### ```run.py```

Script to produce the same .csv predictions used in the best submission on the Kaggle platform.



