import tweepy
import common_functions as cf

twitter_credentials = cf.get_twitter_credentials('twitter_credentials')

auth = tweepy.OAuthHandler(twitter_credentials['api_key'], twitter_credentials['api_secret'])
auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])

api = tweepy.API(auth)
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
