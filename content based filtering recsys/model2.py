import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from nltk.tokenize import word_tokenize

class RecommenderSystem:
    def __init__(self):
        self.df = pd.read_csv('content_by_metadata.csv')
        self.content_col= "Metadata"
        self.label_col = "Labels"
        self.encoder = None
        self.bank = None

    def recommend(self, label, topk=30):
        self.encoder= CountVectorizer(stop_words="english", tokenizer=word_tokenize, token_pattern=None)
        self.bank= self.encoder.fit_transform(self.df[self.content_col])
        idx = self.df.index[self.df[self.label_col] == label].tolist()

        if not idx:
            print(f"No videos found with the label: {label}")
            return None

        idx = idx[0]  
        content = self.df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx]

recsys = RecommenderSystem()

recsys.recommend("Family")