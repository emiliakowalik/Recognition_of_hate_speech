The project aims to analyze posts and comments on the Internet. It makes it possible to pre-classify whether an entry may be hate speech
​

Python libraries required:
​
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('point')
nltk.download('stopwords')
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import string
import pickle
​

The project requires entries and comments that we want to verify or classify as hate speech.
​

Source of data used to train the model:
https://www.kaggle.com/datasets/usharengaraju/dynamically-generated-hate-speech-dataset
