import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('content_by_metadata.csv')
data = data.reset_index(drop=True)
data = data.dropna(subset=['Labels', 'Likes', 'Views']).reset_index(drop=True)

data['Labels'] = data['Labels'].str.upper()

vectorizer = TfidfVectorizer()
lc_matrix = vectorizer.fit_transform(data['Labels'])

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(lc_matrix.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(lc_matrix.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(lc_matrix.toarray(), np.identity(lc_matrix.shape[0]), epochs=50, batch_size=32)

def get_recommendations(title):
    title = title.upper()
    df_title = data[data['Labels'] == title]
    
    if not df_title.empty:
        idx = df_title.index[0]
        labels_vector = lc_matrix[idx].toarray()
        scores = model.predict(labels_vector)
        top_n_indices = np.argsort(scores[0])[::-1][1:11]
        recommendations_df = data.loc[top_n_indices, ['Labels', 'Views', 'Likes']]
        return recommendations_df
    else:
        return "The Labels is not found in the dataset."

test = get_recommendations("love")
print(test)
