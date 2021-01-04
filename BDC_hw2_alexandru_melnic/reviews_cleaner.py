# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:03:54 2020

@author: sasha
"""

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from string import punctuation
from tqdm import tqdm



class TextProcessing:
    
    def __init__(self, DATA_PATH, TARGET_PATH, frac_review, 
                 min_word_len = 2, min_text_len = 15, select_nouns=False):
        
        '''
        DATA_PATH is the path to the text file and TARGET_PATH to the labels;
        frac_review refers to the % of texts needed to be sampled, if equal to
        100 it means select all texts;
        min_word_len is the minimum length of the word to be included in the 
        processed text;
        min_text_len is the minimum number of characters in a text;
        if select_nouns is True it means only the nouns are selected from the
        texts (NN tag in nltk).
        
        '''
        
        self.DATA_PATH = DATA_PATH
        self.TARGET_PATH = TARGET_PATH
        self.frac = frac_review
        self.min_word_len = min_word_len
        self.min_text_len = min_text_len
        self.select_nouns = select_nouns
        self.lemmatizer = WordNetLemmatizer()
        
    def import_and_clean(self):
        
        '''
        Method that imports the texts looping on the individual elements and
        processing them with the method clean_text.
        
        '''
        
        self.processed_texts = []
        self.true_target = []
        
        # open files
        with open(self.DATA_PATH, 'r') as text_file, open(self.TARGET_PATH, 'r') as target_file:
            
            row_idx = 0 # row idx counter
            
            # loop on the text rows
            for text, target in tqdm(zip(text_file, target_file), desc = 'processing text'):
                
                # sampling a frac % of the original texts
                if hash(str(row_idx)) % 100 < self.frac:
                    
                   processed_text = self.clean_text(text)
                   
                   # add only texts that are longer than min_text_len characters
                   if len(processed_text) > self.min_text_len:
                       self.processed_texts.append(processed_text)
                       self.true_target.append(int(target[0]))
                row_idx += 1
                
            text_file.close()
            target_file.close()
            
 
        
                                            
    def clean_text(self, text):
        
        '''
        Takes in input an entire sentence and returns the processed version of it.
        
        '''    
        
        word_list = word_tokenize(text)
        
        # if the condition is true create a dict
        # word:tag, to verify if the word is a noun
        if self.select_nouns:
             tags = {word:tag for word, tag in pos_tag(word_list)}
            
        lemmatized_text = ''
        
        # loop on all the words
        for word in word_list:
             
            # if the condition is true verify if the current word is a noun
            # if it is not skip the word
            if self.select_nouns:
                
                if tags[word] != 'NN' :
                    continue

            # remove punctuation and lower all characters              
            cleaned_word = self.punctuation_remover(word, punctuation).lower()
            
            # check if the word is longer than the minimum and if there are no numbers in it
            if len(cleaned_word) > self.min_word_len and self.isnumeric_control(cleaned_word) == False:
                
                # lemmatize
                lemmatized_text = lemmatized_text + ' ' + self.lemmatizer.lemmatize(cleaned_word)
                
        return lemmatized_text
    
    def punctuation_remover(self, string, punctuation = punctuation):
        
        '''Takes in input a string and delets from it all the punctuation.
           Returns the cleaned string.'''
           
        new_string = ''
        for char in string:
            if char not in punctuation:
                new_string = new_string + char
                
        return new_string
    
    def isnumeric_control(self, word):
    
        '''Takes a string as input and checks if there is a number, if yes
            then returns True. '''
        
        for char in word:
            if char.isnumeric():
                return True
            
        return False
    

