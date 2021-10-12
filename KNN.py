# These are imports and you do not need to modify these.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import sklearn
import random
import math

# ===================================================
# You only need to complete or modify the code below.

def process_data(data, labels):
	"""
	Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
    	data: numpy array
    		An array of N strings.
    	labels: numpy array
    		An array of N integer labels.

    Returns
    -------
    train_X: numpy array
		Array with shape (N, D) of N inputs.
    train_Y:
    	Array with shape (N,) of N labels.
    val_X:
		Array with shape (M, D) of M inputs.
    val_Y:
    	Array with shape (M,) of M labels.
    test_X:
		Array with shape (M, D) of M inputs.
    test_Y:
    	Array with shape (M,) of M labels.
	"""
	
	# Split the dataset of string into train, validation, and test 
	# Use a 70/15/15 split
	# train_test_split shuffles the data before splitting it 
	# Stratify keeps the proportion of labels the same in each split

	# -- WRITE THE SPLITTING CODE HERE --
	# Split the data into 70 percent train and 30 percent test and validate data
	train_X, test_X_split, train_Y, test_Y_split = train_test_split(data, labels, test_size=0.30, stratify=labels,random_state= 1)
	# Split the remaining 30 percent data into 15 percent test and validate data each
	test_X, val_X, test_Y, val_Y = train_test_split(test_X_split, test_Y_split, test_size=0.50, stratify=test_Y_split, random_state= 1)

	# Preprocess each dataset of strings into a dataset of feature vectors
	# using the CountVectorizer function. 
	# Note, fit the Vectorizer using the training set only, and then
	# transform the validation and test sets.

	# -- WRITE THE PROCESSING CODE HERE --
	# Preprocess dataset using CountVectorizer from ngram range of 1 to 3
	vector = CountVectorizer(ngram_range=(1,3))
	# Fit data on train dataset
	train_X = vector.fit_transform(train_X)
	# Transform data on test dataset
	test_X = vector.transform(test_X)
	# Transform data on validate dataset.
	val_X = vector.transform(val_X)
	# Return the training, validation, and test set inputs and labels
	return train_X, train_Y, val_X, val_Y, test_X, test_Y
	# -- RETURN THE ARRAYS HERE -- 


def select_knn_model(train_X, val_X, train_Y, val_Y):
	"""
	Test k in {1, ..., 20} and return the a k-NN model
	fitted to the training set with the best validation loss.

    Parameters
    ----------
    	train_X: numpy array
    		Array with shape (N, D) of N inputs.
    	train_X: numpy array
    		Array with shape (M, D) of M inputs.
    	train_Y: numpy array
    		Array with shape (N,) of N labels.
    	val_Y: numpy array
    		Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
    	The best k-NN classifier fit on the training data 
    	and selected according to validation loss.
  	best_k : int
    	The best k value according to validation loss.
	"""
	# acc and model array to store accuracy and models
	acc = []
	model = []
	# Iterate 1 to 20 in search of best model and it's corresponding K value
	for k in range(1, 21):
		# knn model created with the 'k' neighbors
		knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
		# Fit the model on train data
		test = knn.fit(train_X, train_Y)
		# Predict using validation data
		predict_v = test.predict(val_X)
		# Store the accuracy of original and predicted value in acc array
		acc.append(sklearn.metrics.accuracy_score(val_Y, predict_v))
		# Store the model of each iteration in model array
		model.append(knn)
	# return the K value and it's corresponding best model
	return model[acc.index(max(acc))], acc.index(max(acc)) + 1



# You DO NOT need to complete or modify the code below this line.
# ===============================================================


# Set random seed
np.random.seed(3142021)
random.seed(3142021)

def load_data():
	# Load the data
	with open('./clean_fake.txt', 'r') as f:
		fake = [l.strip() for l in f.readlines()]
	with open('./clean_real.txt', 'r') as f:
		real = [l.strip() for l in f.readlines()]

	# Each element is a string, corresponding to a headline
	data = np.array(real + fake)
	labels = np.array([0]*len(real) + [1]*len(fake))
	return data, labels


def main():
	data, labels = load_data()
	train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)

	best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
	test_accuracy = best_model.score(test_X, test_Y)
	print("Selected K: {}".format(best_k))
	print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
	main()
