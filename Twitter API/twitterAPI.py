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




consumer_key = "#"
consumer_secret = "#"
access_token = "#"
access_secret = "#"

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
