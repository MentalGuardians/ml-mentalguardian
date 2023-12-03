import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data and reset index
data = pd.read_csv('vid_youtube.csv')
data = data.reset_index(drop=True)

# Remove rows with NaN values in labels and Likes
data = data.dropna(subset=['Labels', 'Likes', 'Views']).reset_index(drop=True)

# Remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))

# Tokenize and stem the labels
stemmer = nltk.stem.PorterStemmer()
data['stemmed_labels'] = data['Labels'].apply(lambda x: ' '.join([stemmer.stem(word.lower()) for word in nltk.word_tokenize(x) if word.lower() not in stop_words]))

# Concatenate the stemmed labels and the views
data['labels_and_views'] = data['stemmed_labels'] + ' ' + data['Views'].astype(str)

# Vectorize the labels and views
vectorizer = TfidfVectorizer()
lc_matrix = vectorizer.fit_transform(data['labels_and_views'])

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(lc_matrix.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(lc_matrix.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(lc_matrix.toarray(), np.identity(lc_matrix.shape[0]), epochs=50, batch_size=32)

# Define a function to get recommendations
def get_recommendations(title):
    df_title = data[data['Labels'] == title.upper()]
    if not df_title.empty:
        idx = df_title.index[0]
        labels_vector = lc_matrix[idx].toarray()
        scores = model.predict(labels_vector)
        top_n_indices = np.argsort(scores[0])[::-1][1:11]
        recommendations_df = data.loc[top_n_indices, ['Labels', 'Views', 'Likes']]
        return recommendations_df
    else:
        return "The Labels is not found in the dataset."
