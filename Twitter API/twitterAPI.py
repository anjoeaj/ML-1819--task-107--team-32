# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:05:01 2018

@author: cjmcm
"""

#Import the necessary methods from tweepy library
import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler




consumer_key = "nOduyS52w0X7PaBeDPTsYlK9p"
consumer_secret = "eiAYX2jkWVtzztO9ejyuLCLhSC8bQ1JpRICldpgg6TKr6oZ7na"
access_token = "4528287733-Y7KOlcaaoo3vMTCd94JyWKCfWDVOAtQayrvG34S"
access_secret = "hyIo426W4CWn8fDpnGHW2aSB9adajAvBLBYHskXO354Z9"

class listener(StreamListener):

    def on_data(self, data):
        print(data)
        return(True)

    def on_error(self, status):
        print (status)S

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(languages=["en"], track=["a", "the", "i", "you", "u"])