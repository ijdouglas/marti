# Use like: `from contextualize import *` to make sure variables get defined
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Get stop words and subset of words to actually remove from the stopwords
stops_ = list(stopwords.words('english'))
keep_words = [
  # Words originally in stop words that we don't want to ignore
  'below', 'above', 'over', 'under', 'most', 'more',
  'should','now', 'up', 'down'
]

def contextualize_all(text, target, window_size, stops = stops_):
  # text: a string, containing the full passage (so a string of length 1)
  # target (str): a string containing the target word
  # window_size: the full size of words before plus words after

  # Define w_ the number of words before = the number of words after target
  w_ = int(np.ceil(window_size / 2)) # actual # of words before and after

  # Get the stopwords from nltk
  # from nltk.corpus import stopwords
  # stops = list(stopwords.words('english'))
  # Remove some of the stopwords from the list
  stops = [x for x in stops if x not in keep_words]

  # Get all the words in a list, one element per word; to lowercase
  words_ = [x.lower() for x in text.split(' ')]
  # drop reduced set of stop words
  words_ = [x for x in words_ if x not in stops]

  # Locate all instances of the target
  loc_ = [i for i, x in enumerate(words_) if x == target.lower()]

  # Get word count to use as upper bound of after window
  wc_ = len(words_) # word count 
  
  # Loop through instances and extract before/after
  out = []
  for i in loc_:
    before_ = words_[np.max([0, i - w_]):i]
    after_ = words_[(i + 1):np.min([100000, i + 1 + w_])]
    out.append(pd.DataFrame({
      'target': [target],
      'window_size': [window_size],
      'location': [i],
      'before': [' '.join(before_)],
      'after': [' '.join(after_)]
    }))
  return pd.concat(out, axis = 0).reset_index(drop = True)