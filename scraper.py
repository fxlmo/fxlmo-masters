from dataclasses import dataclass
from datetime import datetime
from numpy import result_type
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import yfinance as yf
import time
import os

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_SECRET = os.getenv('ACCESS_SECRET')
BEARER = os.getenv('BEARER')

import tweepy

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)
client = tweepy.Client(bearer_token=BEARER,consumer_key=API_KEY,consumer_secret=API_SECRET,access_token=ACCESS_TOKEN, access_token_secret=ACCESS_SECRET)

# !hello tweepy: collects and prints tweets from timeline
# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)

# query = '("Apple Inc" OR apple) lang:en is:verified'
from_date = datetime(2021,1,1) # jan 1
to_date = datetime(2021, 3,31) # march 31
# format  KEYWORD1 KEYWORD2 ... -filter:FILTER1 AND -filter:FILTER2 AND ...
query = "(hello is:verified)"

# tweets = tweepy.Cursor(client.search_recent_tweets, q=query, result_type="recent").items(5)

tweets = client.search_recent_tweets(query=query)

# tweets = tweepy.Cursor(api.search_tweets,
#                         q=query,
#                         maxResults=500,
#                         until=to_date).items(500)

for t in tweets:
    print(t)
    print("=================================")