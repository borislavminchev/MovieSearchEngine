import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import unicodedata


class MovieSearchEngine:
    def __init__(self, csv_file, text_column, model_name='distilbert-base-cased', batch_size=16):
        """
        Initialize the search engine.
        :param csv_file: Path to the preprocessed CSV file.
        :param text_column: Column name containing the search content.
        :param model_name: Pretrained model to use for encoding.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.text_column = text_column
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.batch_size = batch_size
        self.embeddings = self.__calculate_embeddings()

    def preprocess_text(self, text):
        """
        Preprocess the text for encoding.
        """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word.lower() for word in tokens]  # Lowercasing
        tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
        tokens = [
            unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for word in tokens  # Character normalization
        ]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        tokens = [token for token in tokens if token not in punctuation]  # Remove punctuation
        return " ".join(tokens)

    def __calculate_embeddings(self):
        """
        Calculate embeddings for all movies in the dataset.
        """
        embeddings = []
        for i, text in enumerate(self.dataframe[self.text_column]):
            if(i == 500):
                break
            print(i)
            embeddings.append(self.__calculate_text_embedding(text))
        return np.array(embeddings)

        return np.vstack(embeddings)  # Combine all batches into a single array
    
    @torch.no_grad()
    def __calculate_text_embedding(self, text):
        """
        Calculate the embedding for a single piece of text.
        """
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        output = self.model(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()

    def search(self, query, top_n=5):
        """
        Perform a search for the given query and return the top N results.
        :param query: The search query.
        :param top_n: Number of top results to return.
        """
        query = self.preprocess_text(query)
        query_embedding = self.__calculate_text_embedding(query)
        similarities = self.__calculate_similarity(query_embedding, self.embeddings)
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = self.dataframe.iloc[top_indices]
        return [(row['title'], similarities[idx]) for idx, row in results.iterrows()]

    def __calculate_similarity(self, query_embedding, movie_embeddings):
        """
        Calculate cosine similarity between the query embedding and movie embeddings.
        """
        return cosine_similarity([query_embedding], movie_embeddings)[0]


# Example usage
if __name__ == "__main__":
    # Path to the preprocessed CSV
    csv_file = "./preprocessed_movies_clean.csv"
    text_column = "title"

    # Initialize the search engine
    search_engine = MovieSearchEngine(csv_file, text_column)

    # Search for a query
    query = "science fiction adventure"
    top_results = search_engine.search(query, top_n=5)

    print("Top search results:")
    for title, similarity in top_results:
        print(f"{title} (Similarity: {similarity:.4f})")
