from concurrent.futures import process
from dataclasses import dataclass
from datetime import date, datetime
import profile
import re
from numpy import result_type
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import yfinance as yf
import time
import os
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_SECRET = os.getenv('ACCESS_SECRET')
BEARER = os.getenv('BEARER')
tickers = ["#III", "#ABDN", "#ADM", "#AAF", "#AAL", "#ANTO", "#AHT", "#ABF", "#AZN", "#AUTO", "#AVST", "#AVV", "#AV.", "#BME", "#BA.", "#BARC", "#BDEV", "#BKG", "#BP.", "#BATS", "#BLND", "#BT.A", "#BNZL", "#BRBY", "#CCH", "#CPG", "#CRH", "#CRDA", "#DCC", "#DPH", "#DGE", "#ECM", "#ENT", "#EVR", "#EXPN", "#FERG", "#FLTR", "#FRES", "#GSK", "#GLEN", "#HLMA", "#HL.", "#HIK", "#HSBA", "#IMB", "#INF", "#IHG", "#ICP", "#IAG", "#ITRK", "#ITV", "#JD.", "#KGF", "#LAND", "#LGEN"]
# tickers = ["III", "ABDN", "ADM", "AAF", "AAL", "ANTO", "AHT", "ABF", "AZN", "AUTO", "AVST", "AVV", "AV.", "BME", "BA.", "BARC", "BDEV", "BKG", "BP.", "BATS", "BLND", "BT.A", "BNZL", "BRBY", "CCH", "CPG", "CRH", "CRDA", "DCC", "DPH", "DGE", "ECM", "ENT", "EVR", "EXPN", "FERG", "FLTR", "FRES", "GSK", "GLEN", "HLMA", "HL.", "HIK", "HSBA", "IMB", "INF", "IHG", "ICP", "IAG", "ITRK", "ITV", "JD.", "KGF", "LAND", "LGEN", "LLOY", "LSEG", "MNG", "MGGT", "MRO", "MNDI", "NG.", "NWG", "NXT", "OCDO", "PSON", "PSH", "PSN", "PHNX", "POLY", "PRU", "RKT", "REL", "RTO", "RMV", "RIO", "RR.", "RMG", "SGE", "SBRY", "SDR", "SMT", "SGRO", "SVT", "SHEL", "SN.", "SMDS", "SMIN", "SKG", "SPX", "SSE", "STJ", "STAN", "TW.", "TSCO", "ULVR", "UU.", "VOD", "WTB", "WPP"]

stop_words = set(stopwords.words('english'))

import tweepy

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)
client = tweepy.Client(bearer_token=BEARER,consumer_key=API_KEY,consumer_secret=API_SECRET,access_token=ACCESS_TOKEN, access_token_secret=ACCESS_SECRET)

# !hello tweepy: collects and prints tweets from timeline
# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)

#** normalise
#** @input: body of tweet
#** @output: list of normalised words for bag of words computing.
def normalise(text):
    text = text.lower()
    print("ORIGINAL TEXT: " + text + "\n")
    # remove emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    # lowercase text
    processed_text = emoji_pattern.sub(r'', text).lower()
    #TODO: expand contractions
    # process text to only contain numbers
    processed_text = re.sub(r'\n', ' ', processed_text)
    processed_text = re.sub(r'[^a-z ]', '', processed_text)
    # print("ALPHANUM TEXT: " + processed_text + "\n")
    # tokenize text
    words = nltk.wordpunct_tokenize(processed_text)
    # lemmatise and remove non-english and remove stopwords
    lemmatizer = WordNetLemmatizer()
    lemmatised_words = []
    en_words = set(nltk.corpus.words.words())
    for w in words:
        rootword = lemmatizer.lemmatize(w, pos="v")
        if rootword not in stop_words and (rootword in en_words or not rootword.isalpha()):
            lemmatised_words.append(rootword)
        # print(w + ": " + rootword)
    # processed_text = " ".join(w for w in lemmatised_words if w in en_words or not w.isalpha())
    # print("REMOVED NON ENGLISH: " + processed_text + "\n")
    print("OUTPUT: " + str(lemmatised_words) + "\n")
    #TODO: bag-of-words representation
    return lemmatised_words

def scrape():
    #example response for testing so I dont burn through tweets
    test_tweets = ['The Azeri Central East (#ACE) project is progressing according to the plan. The first 2022 ACE subsea operation was completed safely - two subsea spools successfully landed on the pre-installed crossing structures on the Central Azeri platform location. #bp https://t.co/EJUfCDZmMd', "Interested in working at the forefront of #AV technology?🌏 When you join the Panasonic KAIROS team, you'll be paving the way forward by providing clients with the latest industry trends. Learn more about the team on the #blog 👉 https://t.co/0kop3OCToj #LifeAtPanasonic https://t.co/p0iD1Ispsd", "It's #RaceEqualityWeek, take a look at our @NHSE_Diversity infographic which details key race equality challenges in the NHS. It also includes actions to help improve the experiences of #BME staff.\n\n#ActionNotJustWords #OurNHSPeople #WRES https://t.co/IOejWwyVaW https://t.co/l3DsDzGxEd", 'TV ratings body #BARC said it will be restarting issuing data on news channels from mid-March, over 17 months after suspending it.\nhttps://t.co/oRUC0ZbEiA', '🗞️ #BME welcomes @Atrys_Health as it moves from #BMEGrowth to the stock exchange. \n🔗https://t.co/YUmjA5nkVF https://t.co/dxNz9GDUTx', 'The Broadcast Audience Research Council (BARC) had in October 2020 "temporarily suspended" the ratings on the news genre following a \'cash for ratings\' scam.\n#BARC #TRP \n\n https://t.co/gqBPlfOxAw', 'Follow the new Film in Finland showcase page on LinkedIn! \nFilm in Finland page publishes industry news from Finnish productions, companies, commissions, and organizations. Film in Finland is operated by Business Finland.\n#filminfinland #audiovisual #av https://t.co/wRjVMNPxzP', '@MIB_India @meenaambwani #BARC will resume release of viewership data for news channels and comes after a directive of the \n\n@meenaambwani\nreports\n\nhttps://t.co/T04XQGOCdn', 'Television monitoring agency #BARC (Broadcast Audience Research Council) will resume ratings for news channels from 17 March, 2022.  https://t.co/DQXb0hp9K2', '#EarningsWithETNOW | GSK Pharma Q3 (Consolidated YoY) \n\n-Revenue from operations at Rs 816 cr vs Rs 792 cr \n-PAT at Rs 150 cr vs Rs 156 cr \n\n#StocksToWatch #GSKPharma #GSK @GSK https://t.co/ZqeAzVDbGT']

    # query = '("Apple Inc" OR apple) lang:en is:verified'
    from_date = datetime(2021,1,1) # jan 1
    to_date = datetime(2021,3,31) # march 31
    # format  KEYWORD1 KEYWORD2 ... -filter:FILTER1 AND -filter:FILTER2 AND ...
    query = "("
    for t in tickers:
        query += t + " OR "
    query = query[:-4] + ") is:verified -is:retweet lang:en"

    # NOTE: this is the important line that actually pulls tweets
    # tweets = client.search_recent_tweets(query=query, tweet_fields=["created_at","public_metrics"], max_results=10)

    # tweet_text = []
    # for t in tweets.data:
    #     tweet_text.append(t["text"])
    # print(tweet_text)

    for t in test_tweets:
        normalise(t)
        print("=====================================")

    # for t in tweets.data:
    #     tweet_tickers = []
    #     normalised_text = normalise(t["text"])

    #     for tick in tickers:
    #         if tick in t["text"]:
    #             tweet_tickers.append(tick.replace('"', "").replace("#", ""))
    #     tweet_tickers = json.dumps(tweet_tickers)
    #     tweet_dict = {
    #         "id": t["id"],
    #         "created_on": str(t["created_at"]),
    #         "collected_on": str(date.today()),
    #         # "followers": 
    #         # "impressions":t["public_metrics"]
    #         # "likes":
    #         "tickers": str(tweet_tickers),
    #         "pub_metrics": t["public_metrics"],
    #         "text": t["text"]
    #     }
    #     with open("tweets/" + str(t["id"]) + ".json", "w") as outfile:
    #         json.dump(tweet_dict, outfile)
    #     tweet_json = json.dumps(tweet_dict)
    #     # print(tweet_json)
    #     # print(t["created_at"])
    #     # print(t["text"])
    #     # print("=================================")


    # {
    #     "tweet" : {
    #         "id": <id>,
    #         "created_on": <date>,
    #         "collected": <date_collected>,
    #         "followers": <followers>,
    #         "likes": <likes>,
    #         "text": <text>,
    #         "tickers": <id1,id2,id3>
    #         "stock_info": <stock_info>
    #     }
    # }

# MAIN
scrape()