import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('content_by_metadata.csv')
metadata_column = 'Metadata'

vectorizer = TfidfVectorizer()
item_matrix = vectorizer.fit_transform(df[metadata_column].astype(str))

user_input = "Finance, Bullying, Child"
user_vector = vectorizer.transform([user_input])

cosine_similarities = linear_kernel(user_vector, item_matrix).flatten()
recommended_indices = cosine_similarities.argsort()[::-1][:20]
recommended_items = df.iloc[recommended_indices][['Video ID', 'Title', 'Labels', metadata_column]]

print(recommended_items)