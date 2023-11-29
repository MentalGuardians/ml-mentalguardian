import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


data = pd.read_csv(
    "../data/sentiment_analysis_dataset.csv", encoding="ISO-8859-1", engine="python"
)
data.columns = ["label", "time", "date", "query", "username", "text"]
data = data[["text", "label"]]

data["label"][data["label"] == 4] = 1

data_pos = data[data["label"] == 1]
data_neg = data[data["label"] == 0]

data_pos = data_pos.iloc[: int(20000)]
data_neg = data_neg.iloc[: int(20000)]

data = pd.concat([data_pos, data_neg])
data["text"] = data["text"].str.lower()
data["text"].tail(5)

nltk.download("stopwords")
stopwords_list = stopwords.words("english")
from nltk.corpus import stopwords

", ".join(stopwords.words("english"))
STOPWORDS = set(stopwords.words("english"))


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


data["text"] = data["text"].apply(lambda text: cleaning_stopwords(text))
data["text"].head()

english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_punctuations(text):
    translator = str.maketrans("", "", punctuations_list)
    return text.translate(translator)
    data["text"] = data["text"].apply(lambda x: cleaning_punctuations(x))


data["text"].head()


def cleaning_repeating_char(text):
    return re.sub(r"(.)\1+", r"\1", text)


data["text"] = data["text"].astype(str)

data["text"] = data["text"].apply(cleaning_punctuations)
data["text"] = data["text"].apply(cleaning_repeating_char)
data["text"] = data["text"].apply(lambda x: cleaning_repeating_char(x))
data["text"].head()

nltk.download("wordnet")


def cleaning_email(data):
    return re.sub("@[^\s]+", " ", data)


data["text"] = data["text"].astype(str)


data["text"] = data["text"].apply(cleaning_punctuations)
data["text"] = data["text"].apply(cleaning_repeating_char)
data["text"] = data["text"].apply(lambda x: cleaning_email(x))
data["text"].tail()


def cleaning_URLs(data):
    return re.sub("((www\.[^\s]+)|(https?://[^\s]+))", " ", data)


data["text"] = data["text"].apply(lambda x: cleaning_URLs(x))
data["text"].tail()


def cleaning_numbers(data):
    return re.sub("[0-9]+", "", data)


data["text"] = data["text"].apply(lambda x: cleaning_numbers(x))
data["text"].tail()
tokenizer = RegexpTokenizer(r"\w+")
data["text"] = data["text"].apply(tokenizer.tokenize)
data["text"].head()
st = nltk.PorterStemmer()


def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data


data["text"] = data["text"].apply(lambda x: stemming_on_text(x))
data["text"].head()
lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data


data["text"] = data["text"].apply(lambda x: lemmatizer_on_text(x))

X = data.text
y = data.label
y

max_len = 500
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

X_train, X_test, Y_train, Y_test = train_test_split(
    sequences_matrix, y, test_size=0.3, random_state=2
)


def tensorflow_based_model():
    inputs = Input(name="inputs", shape=[max_len])  # step1
    layer = Embedding(2000, 50, input_length=max_len)(inputs)  # step2
    layer = LSTM(128, return_sequences=True)(layer)
    layer = LSTM(64)(layer)
    layer = Dense(256, name="FC1")(layer)  # step4
    layer = Activation("relu")(layer)  # step5
    layer = Dropout(0.5)(layer)  # step6
    layer = Dense(1, name="out_layer")(layer)
    layer = Activation("sigmoid")(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = tensorflow_based_model()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["accuracy"],
)

history = model.fit(X_train, Y_train, batch_size=80, epochs=6, validation_split=0.1)
print("Training finished !!")


def plot_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


# Call the function with the training history
plot_history(history)
