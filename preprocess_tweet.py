## Preprocessing Tweets Function

import nltk 
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

""" 
Load Stopword file
Instanciate Tokenizer

Define preprocess_tweet function:
Remove URLs
Remove @username
Remove hashtag character, but not hashtag words
Remove StopWords and punctuation
 """

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Create a Tweet tokenizer
tokenizer = TweetTokenizer()

# Preprocessing function for tweet data
def preprocess_tweet_2(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    
    # Remove usernames
    tweet = re.sub(r'@[A-Za-z0-9_]{1,15}', '', tweet)

    # Remove # symbol, but not hashtaged word
    tweet = re.sub(r'(?<=\s)#([A-Za-z0-9_]{1,2048})', lambda match: match.group(1), tweet, flags=re.IGNORECASE)
    # print(tweet)

    # Tokenize
    tokens = tokenizer.tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_tweet = ' '.join(tokens)
    
    # Return preprocessed tweet
    return preprocessed_tweet
