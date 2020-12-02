# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 20:41:42 2020

@author: sasha
"""
from surprise import AlgoBase
from datasketch import MinHash, MinHashLSH
import numpy as np
from tqdm import tqdm



class lsh_jaccard(AlgoBase):
    

    def __init__(self, threshold = 0.1, n_perm = 128):
        AlgoBase.__init__(self)
        self.tr = threshold
        self.n_perm = n_perm
        
        
    def compute_minhash_signature(self, user):
        
        '''Returns the signature of a user.'''
        
        m = MinHash(num_perm=self.n_perm)
        for item, _ in self.trainset.ur[user]:
            #m.update(self.trainset.to_raw_iid(item).encode('utf8'))
            m.update(bytes(item))
        return m        
        
    
    def fit(self, trainset):
        
        '''Computes the  signature matrix for the training set.'''
        
        AlgoBase.fit(self, trainset)
        
        self.lsh = MinHashLSH(threshold=self.tr, num_perm = self.n_perm)
        for user in tqdm(self.trainset.ur, desc='Computing LSH'):
            self.lsh.insert(user, self.compute_minhash_signature(user))
            
        return self
        
    def compute_baseline_score(self, user):
        
        '''Returns the baseline score for user given by the average ratings of that user. '''
        
        return np.mean([r for i,r in self.trainset.ur[user]])
    
    def jaccard_sim(self, a, b):
        intersection = a.intersection(b)
        return len(intersection)/(len(a) + len(b) - len(intersection))
    
    def estimate(self, u, i):
        
        '''Computes the predicted rating for user u and item i. For every user computes the neighbors, 
        then selects those users that rated the item i and returns the weighted average of the ratings
        with weights given by the jaccard similarity. The jaccard similarity is computed with the Min Hash.'''
        
        m_current = self.compute_minhash_signature(u)

        neighbors = self.lsh.query(m_current)

        neigh_rating = {user:rating for user, rating in self.trainset.ir[i] if user in neighbors}
 
        tot_sum_sim = 0
        
        baseline_score = self.compute_baseline_score(u)
        predicted_rating = 0
        
        if len(neigh_rating) == 0:
            return baseline_score
        
        for user in tqdm(neigh_rating, desc='Computing predictions'):
            
            m_temp = self.compute_minhash_signature(user)
            temp_sim = m_temp.jaccard(m_current)
            
            predicted_rating += (neigh_rating[user]-self.compute_baseline_score(user))*temp_sim
            tot_sum_sim += temp_sim
        
        predicted_rating = predicted_rating/tot_sum_sim + baseline_score
        return predicted_rating