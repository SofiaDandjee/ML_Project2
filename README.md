# The EPFL Recommender System Challenge 2019
Machine Learning - Project 2

**`Tshtsh_club`**: Marie Anselmet, Sofia Dandjee, Héloïse Monnet

## Install the required libraries

1. Install the Surprise library
2. Install the sklearn library

## Run the project
1. Make sure that ```Python >= 3.7``` and ```NumPy >= 1.16``` are installed
2. Go to `script\` folder and run ```run.py```. You will get ```submission.csv``` for Kaggle in the ```submission\``` folder.

~~~~shell
cd script
python run.py
~~~~

## Script files

### ```proj2_helpers.py```

- `calculate_mse`: Computes mean-squared error
- `calculate_rmse`: Computes root-mean-squared error

### ```data_helpers.py```

- `read_csv_sample : Reads the sample_submission file and extracts the couples (item, user) for which the rating has to be predicted.
- `create_csv_submission`: Creates an output file in csv format for submission to kaggle
- `build_surprise_data`: Loads the training set for it to be usable by the surprise library

### ```implementations.py```

- `global_mean`: Use the global mean as the prediction.
- `user_mean`: Use the user means as the prediction.
- `item_mean`: Use the item means as the prediction.
- `normal_predictor`: Generates predictions according to a normal distribution estimated from the training set
- `baseline_only`: Combines global mean with user and item biases
- `knn_user`: Nearest neighbour approach between users taking into account a baseline rating
- `knn_movie`: Nearest neighbour approach between movies taking into account a baseline rating
- `svd`: Matrix factorization
- `svdpp`: Extension of svd taking into account implicit ratings
- `slopeone`:
- `nmf`: Non-negative matrix factorization
- `blending`: Computes a ridge regression to find optimal weights for each of the previous models



### ```run.py```

Script to produce the same .csv predictions used in the best submission on the Kaggle platform.



