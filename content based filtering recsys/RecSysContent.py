import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Read data
df = pd.read_csv('content_by_metadata.csv')

#Cleaning data
cols = ['Comments', 'Likes', 'Views']
df.drop(cols, axis=1, inplace=True)
df.isnull().sum()

#Model Content Based Filtering RecSys
vectorizer = TfidfVectorizer()
item_matrix = vectorizer.fit_transform(df['Metadata'].astype(str))

user_input = "Finance, Trauma, Child"
user_vector = vectorizer.transform([user_input])

#RecSys Testing
cosine_similarities = cosine_similarity(user_vector, item_matrix).flatten()
recommended_indices = cosine_similarities.argsort()[::-1][:20]
recommended_items = df.iloc[recommended_indices][['Video ID', 'Labels']]

print(recommended_items)