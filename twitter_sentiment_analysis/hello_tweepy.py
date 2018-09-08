import tweepy
import common_functions as cf

twitter_credentials = cf.get_twitter_credentials('twitter_credentials')

auth = tweepy.OAuthHandler(twitter_credentials['api_key'], twitter_credentials['api_secret'])
auth.set_access_token(twitter_credentials['access_token'], twitter_credentials['access_token_secret'])

api = tweepy.API(auth)

#user timeline can be used to mine data from specific users
for tweet in api.user_timeline(screen_name = 'DanskeBank_UK', count = 1000):
    print(tweet.id)
