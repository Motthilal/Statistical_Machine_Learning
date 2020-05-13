# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:10:35 2020

@author: motth
"""

import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import scipy.io
import math
from scipy.stats import multivariate_normal as mvn

# gaussian function
def cal_Probability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def Naive_Bayes():
    Matfile = scipy.io.loadmat('mnist_data.mat') 
    
    trY = Matfile['trY']
    trX = Matfile['trX']
    tsX = Matfile['tsX']
    tsY = Matfile['tsY']
    
    
    labels_x = []
    for item in trY:
      for i in item:
        labels_x.append(int(i))
        
        
    
    ###    seperating training data    ###
    l7 = 0
    l8 = 0
    
    list_ts = []
    
    for item in tsY:
      for i in item:
        if(int(i) == 0):
          l7+=1
        else:
          l8+=1
        list_ts.append(int(i))
    
    
    ###    calculate prior    ###
    prior_7 = l7 / (l7 + l8)
    prior_8 = l8 / (l7 + l8)
    
    mean_x7 = []
    std_x7 = []
    mean_x8 = []
    std_x8 = []
    
    for i, num in enumerate(trX):
      if(labels_x[i] == 0):
        mean_x7.append(num.mean())
        std_x7.append(num.std())
      else:
        mean_x8.append(num.mean())
        std_x8.append(num.std())
        
    mean_mean_7 = sum(mean_x7) / len(mean_x7)
    std_mean_7 = st.stdev(mean_x7)
    mean_std_7 = sum(std_x7) / len(std_x7)
    std_std_7 = st.stdev(std_x7)
    
    
    mean_mean_8 = sum(mean_x8) / len(mean_x8) 
    std_mean_8 = st.stdev(mean_x8)
    mean_std_8 = sum(std_x8) / len(std_x8)
    std_std_8 = st.stdev(std_x8)
    
    
    mean7 = sum(mean_x7) / len(mean_x7)
    std7 = st.stdev(mean_x7)
    
    mean8 = sum(mean_x8) / len(mean_x8)
    std8 = st.stdev(mean_x8)
    
    
    ###    Predict    ###
    prediction_list = []
    
    for item in tsX:  
      avg = item.mean()
      std = item.std()
      prob_avg_8 = cal_Probability(avg, mean_mean_8, std_mean_8)
      prob_std_8 = cal_Probability(std, mean_std_8, std_std_8)
      
      prob_avg_7 = cal_Probability(avg, mean_mean_7, std_mean_7)
      prob_std_7 = cal_Probability(std, mean_std_7, std_std_7)
      
      final_prob_8 = prob_avg_8 * prob_std_8 * prior_8
      final_prob_7 = prob_avg_7 * prob_std_7 * prior_7
      
      if(final_prob_8 < final_prob_7):
        prediction = 0
      else:
        prediction = 1
      
      
      prediction_list.append(prediction)
        
    
    
    ###   Accuracy Calculation   ###
    correct = 0
    wrong = 0
    
    correct7 = 0
    wrong7 = 0
    correct8 = 0
    wrong8 = 0
    
    for i, val in enumerate(prediction_list):
      if(list_ts[i] == 0):
        if(list_ts[i] == val):
          correct += 1
          correct7 += 1
        else:
          wrong += 1
          wrong7 += 1
      if(list_ts[i] == 1):
        if(list_ts[i] == val):
          correct += 1
          correct8 += 1
        else:
          wrong += 1
          wrong8 += 1
        
        
    ###    Final accuracy    ###
    print("Naive Bayes Overall Accuracy:",(correct / (correct + wrong)))
    


def log_likelihood(X, labels_, weights):
	log_like = np.sum( labels_*np.dot(X, weights) - np.log(1 + np.exp(np.dot(X, weights))))
	return log_like


def logistic_function_sigmoid(s):
	return 1 / (1 + np.exp(-s))


def Training_data(X, labels, iterations, lr):
	
	
	W = np.zeros(X.shape[1])
	
	for iter in range(iterations):

		
		s = np.dot(X, W)
		predicted_labels = logistic_function_sigmoid(s)

		
		error = labels - predicted_labels
		W += lr * np.dot(X.T, error)
		
		### For every 100 iterations, print log-likelihood 
#		if iter % 100 == 0:
#			print(log_likelihood(X, labels, W))
		
	return W


def Logistic_regression():

	Matfile = scipy.io.loadmat('mnist_data.mat')

	trained_weights = Training_data(Matfile['trX'],Matfile['trY'][0],10000,3e-3)

	Y = Matfile['tsY'][0]
	X = Matfile['tsX']

	test_data_predictions = np.round(logistic_function_sigmoid(np.dot(X, trained_weights)))
	print("Logistic Regression Overall Accuracy:",((test_data_predictions == Y).sum().astype(float) / len(test_data_predictions)))
    
if __name__ == "__main__":
    Naive_Bayes()
    Logistic_regression()