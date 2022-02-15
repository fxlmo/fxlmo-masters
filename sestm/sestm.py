import json
import sys
import re
import os
import numpy as np
from numpy.linalg import inv
from pathlib import Path
from bs4 import BeautifulSoup as bs
# from textblob import TextBlob as tb
import datetime
from nbformat import read
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import unittest

# constants
STOP_WORDS = set(stopwords.words('english'))
DEBUG = True
NEW_ARTICLES = False
TESTING = False

# hyper params
ALPHA_PLUS = 0.05
ALPHA_MINUS = 0.02
KAPPA = 10

# TESTING METHODS
class TestMethods:

    def test_process_raw(self):
        print('Implement me')

    def test_freq_word(self):
        d = [{'word1': 1, 'word2': 2, 'word3': 3}, {'word1': 1, 'word2': 2, 'word4': 4}, {'word2': 2, 'word3': 3}]
        sgn = [1,0,1]
        (pos_j, total_j) = freq_word(d,sgn)
        self.assertEqual(pos_j, [{'word1': 1}, {'word2': 4}, {'word3': 6}, {'word4': 0}])
        self.assertEqual(total_j, [{'word1': 2}, {'word2': 6}, {'word3': 6}, {'word4': 4}])

    def test_process_article(self):
        arts_raw = [
            {'date': '2021-12-23 12:58:45.061000+00:00', 'ticker': 'ABDN', 'mrkt_info': {'open': 233.7, 'close': 200.3}, 'html': '<p>John likes to watch movies.\nMary likes movies too.</p>'},
            {'date': '2022-01-26 07:11:46.774000+00:00', 'ticker': 'ABDN', 'mrkt_info': {'open': 229.2, 'close': 241.0}, 'html': '<p>Mary also likes to watch football games.</p>'},
            {'date': '2021-10-14 11:23:02.340000+00:00', 'ticker': 'ABDN', 'mrkt_info': {'open': 250.3, 'close': 258.5}, 'html': '<p>John likes to watch movies.\nMary likes movies too.\nMary also likes to watch football games.</p>'},
            {'date': '2021-10-25 13:22:07.985000+00:00', 'ticker': 'ABDN', 'mrkt_info': {'open': 256.9, 'close': 251.9}, 'html': '<p>Carl likes to play football.</p>'}
        ]
        (d, sgn, y) = process_raw_article(arts_raw)
        print(d)
        # self.assertEqual(d, )
        self.assertEqual(sgn, [-1,1,1,-1])
        self.assertEqual(y, [233.7-200.3, 229.2 - 241.0, 250.3 - 258.5, 256.9 - 251.9])


def extract_raw_article(article_path):
    print(article_path)
    pathlist = Path(article_path).rglob('*.json')
    art_list = []
    for path in pathlist:
        # open json
        with open(str(path)) as json_file:
            data = json.load(json_file)
            art_list.append(data)
    # return processed_articles
    return art_list

def process_raw_article(arts_raw):
    # word count vector d
    d = []
    # article sentiment score vector
    sgn = []
    # realised return of article
    y = []
    global_bow = {}
    art_no = 0
    ARTS_TO_SEARCH = len(arts_raw)
    for a in arts_raw:
        raw_html = a['html']
        if raw_html:
            # get text from html and set to lower
            readable_text = bs(raw_html, 'lxml').get_text().lower()
            # substitute non alphabet chars (new lines become spaces)
            readable_text = re.sub(r'\n', ' ', readable_text)
            readable_text = re.sub(r'[^a-z ]', '', readable_text)
            # sub multiple spaces with one space
            readable_text = re.sub(r'\s+', ' ', readable_text)
            # tokenise text
            words = nltk.wordpunct_tokenize(readable_text)
            if len(words) > 0:
                # lemmatise, remove non-english, and remove stopwords
                lemmatizer = WordNetLemmatizer()
                lemmatised_words = []
                en_words = set(nltk.corpus.words.words())
                for w in words:
                    rootword = lemmatizer.lemmatize(w, pos="v")
                    if rootword not in STOP_WORDS and (rootword in en_words or not rootword.isalpha()):
                        lemmatised_words.append(rootword)
                # convert to bag of words
                bow_art = {}
                for l in lemmatised_words:
                    if l in global_bow:
                        global_bow[l] += 1
                    else:
                        global_bow[l] = 1
                    if l in bow_art:
                        bow_art[l] += 1
                    else:
                        bow_art[l] = 1
                d.append(bow_art)
                returns = a['mrkt_info']['open'] - a['mrkt_info']['close']
                y.append(returns)
                if (returns > 0):
                    sgn.append(1)
                else:
                    sgn.append(-1)
        # progress bar
        sys.stdout.write('\r')
        j = (art_no + 1) / ARTS_TO_SEARCH
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()
        art_no += 1
    return (d, sgn, y)

def freq_word(d, sgn):
    pos_j = {}
    total_j = {}
    for i in range(len(d)):
        for w in d[i]:
            pos_sent = 0
            if (sgn[i] == 1):
                pos_sent = 1
            if w in total_j:
                total_j[w] += d[i][w]
                pos_j[w] += d[i][w]*pos_sent
            else:
                total_j[w] = d[i][w]
                pos_j[w] = d[i][w]*pos_sent
    return (pos_j, total_j)

# return list of sentimentally charged words and neutral words in tuple
def get_sentiment_words(pos_j, total_j, pi):
    sentiment_words = []
    neutral_words = []
    for i in total_j:
        if ((pos_j[i]/total_j[i] >= pi + ALPHA_PLUS or pos_j[i]/total_j[i] <= pi - ALPHA_MINUS) and total_j[i] >= KAPPA):
            sentiment_words.append(i)
        else:
            neutral_words.append(i)
    return (sentiment_words, neutral_words)



def main():
    if TESTING:
        print("Doing testing")
        return
    if NEW_ARTICLES:
        # run this when new articles are added.
        print("Extracting raw articles...")
        arts_raw = extract_raw_article(os.getcwd() + '/sestm/articles-pulled')
        print("Processing html into BOW form...")
        (d, sgn, y) = process_raw_article(arts_raw)
        # write to file so I don't have to compute every time
        with open('prev-data/word-vectors.json', 'w') as fp:
            json.dump(d, fp)
        with open('prev-data/article-signs.json', 'w') as fp:
            json.dump(sgn, fp)
        with open('prev-data/article-returns.json', 'w') as fp:
            json.dump(y, fp)
    else:
        # read from stored file
        print("Reading from file")
        with open('prev-data/word-vectors.json') as f:
            d = json.load(f)
        with open('prev-data/article-signs.json') as f:
            sgn = json.load(f)
        with open('prev-data/article-returns.json') as f:
            y = json.load(f)

    # SCREENING FOR SENTIMENT CHARGED WORDS
    # fraction of articles with return set to +1
    pi = sgn.count(1) / len(sgn)
    if DEBUG:
        print("pi = " + str(pi))

    print("Calculating word frequencies...")
    (pos_j, total_j) = freq_word(d, sgn)

    if DEBUG:
        total_j = {k: v for k, v in sorted(total_j.items(), key=lambda item: item[1])}
        with open('prev-data/frac-pos.txt', "w") as f:
            for i in total_j:
                f.write(str(i) + ":\t" + str(round(pos_j[i] / total_j[i],3)) + "\t\t" + str(total_j[i]) + "\n")
    # if DEBUG:
    #     for i in total_j:
    #         print("Frac pos of " + str(i) + ": " +  str(pos_j[i]) + "/" + str(total_j[i]))

    print("Getting sentiment words")
    (sentiment_words, neutral_words) = get_sentiment_words(pos_j, total_j, pi)

    # LEARNING SENTIMENT TOPICS
    s = []                                          # ith element corresponds to total count of sentiment charged words for document i
    d_s = []                                        # ith element corresponds to list of word counts for each of the sentiment charged words for document i
    h = np.zeros((len(d), len(sentiment_words)))    # ith element corresponds to |S|x1 vector of word frequencies divided by total sentiment words in doc i
    print("Calculating s_i")
    for doc in d:
        s.append(sum(doc.get(val,0) for val in sentiment_words))
        d_s.append([doc.get(val,0) for val in sentiment_words])

    print("Calculating h_i")
    for i in range(len(d)):
        # subvector of sentiment words in d_i
        if (s[i] == 0) :
            h[i] = np.zeros(len(sentiment_words)).transpose()
        else:
            h[i] = np.array([(j/s[i]) for j in d_s[i]]).transpose()
        
    print("Calculating p_i")
    p = [(rank/len(y)) for (rank,_) in enumerate(sorted(range(len(y)), key=lambda y_temp: y[y_temp]))]

    print("Calculating O")
    p_inv = [(1-val) for val in p]
    W = np.column_stack((p, p_inv))
    W = W.transpose()
    ww = np.matmul(W,W.transpose())
    w2 = np.matmul(W.transpose(), inv(ww))
    O = np.matmul(h.transpose(),w2)
    # TODO: renormalise each column to have unit L1 norm (wtf does this mean lol)
    #Now we can score new articles



main()