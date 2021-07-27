import unicodedata
import re
import json
from requests import get
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from requests import get
from bs4 import BeautifulSoup
import os
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')

def basic_clean(string):
    '''Takes in a string and normalizes it '''
    
    # normalizes data into unicode
    string = unicodedata.normalize('NFKD', string)\
                        .encode('ascii', 'ignore')\
                        .decode('utf-8', 'ignore')
    # removes whitespace 
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string

def tokenize(string):
    '''Takes in a string and TokTok tokenizes it'''
    # init tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # tokenize and return the string
    tokenizer.tokenize(string, return_str = True)
    return string

def stem(string):
    '''Applies Porter Stemming to a string'''
    # init porter stemming
    ps = nltk.porter.PorterStemmer()
    
    # stem each word in the string
    stems = [ps.stem(word) for word in string.split()]
    
    # join and return 
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(string):
    '''Lemmatizes a string'''
    # init lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # lemmatize the words in the string
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # join lemmatized words back to the data
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''removes stopwords from a string. Has the ability to include or exclude words
    '''
    stopword_list = stopwords.words('english')
    
    # remove excluded words from the set.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # unionize extra words to add to he list
    stopword_list = stopword_list.union(set(extra_words))
    print(string)
    # split it up
    words = string.split()
    
    # create a list of words not in stopword list.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # join words back and return them
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords
    
def prep_text_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(stem)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['lemmatized'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(lemmatize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]    