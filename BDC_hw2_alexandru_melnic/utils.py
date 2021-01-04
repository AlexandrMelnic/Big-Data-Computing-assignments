# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:14:09 2020

@author: sasha
"""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from wordcloud import WordCloud

    
def return_top_terms(cluster_centers, feature_names, 
                     return_list = False, top_k = 10,
                     print_terms=False):
    
    '''Given the cluster centroids gets the top_k largest terms. Takes
    in input the cluster centers and the feature_name dictionary provided by 
    the vectorizer. The paraemeter return_list is to return or not the dictionary
    needed to print the word clouds and print_terms is to print or not the top_k
    terms in order.'''
    
    top_terms_id = np.argsort(cluster_centers)[:, :-top_k - 1:-1]
    top_terms_tfidf = np.sort(cluster_centers)[:, :-top_k - 1:-1]
    
    terms_list = []
    for i in range(cluster_centers.shape[0]):
        
        if print_terms:
            print("Cluster %d:" % i, end='')
        
        term_tfidf = {}
        for ind, tf in zip(top_terms_id[i], top_terms_tfidf[i]):
            
            term_tfidf[feature_names[ind]] = tf
            
            if print_terms:
                print(' %s' % feature_names[ind], end='')
            
            
        terms_list.append(term_tfidf)
        print()
        
    if return_list:
        return terms_list

def return_clustering_metrics(cl_labels, labels, X):
    
    '''Prints the metrics for a given clustering. Takes in input
    the cluster labels and the true labels. '''
    
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, cl_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, cl_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, cl_labels))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, cl_labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, cl_labels, sample_size=1000))
    print('Accuracy: %0.3f' % metrics.accuracy_score(labels, cl_labels))
    
    
def plot_word_clouds(top_terms, save_fig = None):
    
    ''' Plots the wordcloud for a given clustering. Takes in input a dictionary term:score, 
    as given optionally in output from the return_top_terms function. If save fig is the path to
    save the plot, if it is None then the plot is not performed.'''
    
    cloud_cluster_0 = WordCloud()
    cloud_cluster_0.generate_from_frequencies(frequencies=top_terms[0])
    cloud_cluster_1 = WordCloud()
    cloud_cluster_1.generate_from_frequencies(frequencies=top_terms[1])
    
    plt.figure( figsize=(15,12) )
    
    plt.subplot(1,2,1)
    plt.imshow(cloud_cluster_0, interpolation="bilinear")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(cloud_cluster_1, interpolation="bilinear")
    plt.axis("off")
    
    if save_fig:
        plt.save(save_fig+'.png')
    plt.show()
    

