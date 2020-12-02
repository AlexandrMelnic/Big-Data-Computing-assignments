# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:40:38 2020

@author: sasha
"""
from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold
from surprise import accuracy
from lsh_jaccard import lsh_jaccard



train_file_path = "train.csv"
test_file_path = "test.csv"
reader = Reader(line_format='user item rating timestamp', sep=',')
#data = Dataset.load_from_file(train_file_path, reader=reader)
data = Dataset.load_from_folds([(train_file_path, test_file_path)], reader=reader)
pkf = PredefinedKFold()


    
algo = lsh_jaccard(threshold = 0.1)
for trainset, testset in pkf.split(data):
    
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
