import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load the dataset
df = pd.read_csv("./TMDB_all_movies.csv")

# Select relevant columns
columns_to_keep = [
    "id", "title", "overview", "genres", 
    "cast", "director"
]
df = df[columns_to_keep]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word.lower() for word in tokens]  # Lowercasing
    tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
    tokens = [
        unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        for word in tokens  # Character Normalization
    ]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

def clean_special_characters(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'["]{2,}', '', text)  # Replace multiple quotes
    text = re.sub(r'^["]|["]$', '', text)  # Remove leading/trailing quotes
    text = re.sub(r'[^\w\s\.,!?\'"]', '', text)  # Remove unwanted special characters
    return text

# Fill missing text fields with an empty string
text_columns = ["title", "overview", "genres", "cast", "director"]
df[text_columns] = df[text_columns].fillna("").astype(str)

# Apply preprocessing
for col in text_columns[1:]:
    df[col] = df[col].apply(clean_special_characters).apply(preprocess_text)

df.replace(["null"], np.nan, inplace=True)
df["title"] = df["title"].apply(lambda x: x.lower())

# Combine text columns for search content
df["search_content"] = df[text_columns].apply(lambda row: " ".join(row), axis=1)

df["title"].replace("nan", "Nan", inplace=True)
df = df.dropna()
df = df[(df != "" ).all(axis=1)]

# Save preprocessed dataset
df.to_csv("./preprocessed_movies_clean.csv", index=False)



# def clean_text(text):
#     text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text

# def remove_stopwords(text):
#     return " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])



# lemmatizer = WordNetLemmatizer()

# def lemmatize_text(text):
#     return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# def preprocess_text(movies):
#     movies = [clean_text(movie.lower()) for movie in movies]
#     movies = [remove_stopwords(movie) for movie in movies]
#     movies = [lemmatize_text(movie) for movie in movies]
#     return list(set(movies))