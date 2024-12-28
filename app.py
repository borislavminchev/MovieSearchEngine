from typing import Tuple
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from transformers.modeling_outputs import BaseModelOutput

from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import nltk

movies = [
    "The Godfather",
    "The Shawshank Redemption",
    "Schindler's List",
    "Forrest Gump",
    "The Matrix",
    "Inception",
    "The Dark Knight",
    "Pulp Fiction",
    "Fight Club",
    "Interstellar",
    "The Lord of the Rings: The Return of the King",
    "The Empire Strikes Back",
    "The Silence of the Lambs",
    "Parasite",
    "Gladiator"
]
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
    tokens = [token for token in tokens if token not in punctuation]
    return " ".join(tokens)

movies = preprocess_text(movies)



def movie_to_string(movie):
    return movie

class SearcEngine:
    def __init__(self, movies):
        self.movies = movies
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.embeddings = self.__calculate_embeddings(movies)

    @torch.no_grad
    def search(self, query, top_n=5):
        query_tokens = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        output = self.model(**query_tokens)
        query_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()

        similarities = self.__calculate_similarity(query_embedding, self.embeddings)
        top_indices = similarities.argsort()[-top_n:][::-1]

        return [(self.movies[i], similarities[i]) for i in top_indices]
    
    def __calculate_embeddings(self, movies):
        return np.array([self.__calculate_movie_embedding(movie) for movie in movies])

    
    @torch.no_grad
    def __calculate_movie_embedding(self, movie):
        tokens = self.tokenizer(movie_to_string(movie), return_tensors='pt', truncation=True, padding=True)
        output = self.model(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()
    
    def __calculate_similarity(self, query_embedding, movie_embeddings):
        return cosine_similarity([query_embedding], movie_embeddings)[0]

engine = SearcEngine(movies)
# res1 = engine.search("psycho")
res2 = engine.search("psycho")
# print(res1)
print(res2)
print(movies)

    





