import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# # Load the dataset
df = pd.read_csv("./TMDB_all_movies.csv")

# Select relevant columns
columns_to_keep = [
    "id", "title", "overview", "genres", "tagline", 
    "cast", "director", "writers", "production_companies"
]
df = df[columns_to_keep]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Removal of Stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Character Normalization (remove accents, special characters)
    tokens = [
        unicodedata.normalize('NFKD', word)
        .encode('ascii', 'ignore')
        .decode('utf-8', 'ignore') 
        for word in tokens
    ]
    # Stemming or Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return " ".join(tokens)

# Fill missing text fields with an empty string
text_columns = ["title", "overview", "genres", "tagline", "cast", "director", "writers", "production_companies"]
df[text_columns] = df[text_columns].fillna("")

# Apply preprocessing
for col in text_columns:
    df[col] = df[col].apply(preprocess_text)


# Combine title, overview, genres, tagline, cast, director, and writers
df["search_content"] = df[
    ["title", "overview", "genres", "tagline", "cast", "director", "writers"]
].apply(lambda row: " ".join(row), axis=1)


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