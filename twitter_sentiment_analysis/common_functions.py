import os

def get_twitter_credentials(env_var_name):

    try:
        twitter_key_string = os.environ.get(env_var_name)
    except Exception as e:
        print('Missing environment variable %s' % env_var_name)
    else:
        if twitter_key_string:
            twitter_key = dict(credential.split(':') for credential in twitter_key_string.split(';'))
        else:
            api_key = input('Paste the api_key:\n')
            api_secret = input('Paste the api_secret:\n')
            access_token = input('Paste the access_token:\n')
            access_token_secret = input('Paste the access_token_secret:\n')
            twitter_key = {\
            'api_key' : api_key,\
            'api_secret' : api_secret,\
            'access_token' : access_token,\
            'access_token_secret' : access_token_secret\
            }

    return twitter_key
