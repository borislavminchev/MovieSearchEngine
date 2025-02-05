import os
import pickle
import unicodedata
from string import punctuation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class MovieSearchEngineV2:
    def __init__(self, csv_file, text_column, model_name='distilbert-base-cased',
                 batch_size=64, embeddings_file='./embeddings.pkl', use_mean_pooling=True):
        """
        Initialize the search engine.
        :param csv_file: Path to the preprocessed CSV file.
        :param text_column: Column name containing the search content.
        :param model_name: Pretrained model to use for encoding.
        :param batch_size: Batch size for embedding computation.
        :param embeddings_file: Path to the pickle file for caching embeddings.
        :param use_mean_pooling: If True, use mean pooling over token embeddings instead of CLS token.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.text_column = text_column
        self.batch_size = batch_size
        self.embeddings_file = embeddings_file
        self.use_mean_pooling = use_mean_pooling

        # Device configuration: use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the tokenizer and model once, and move model to device.
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Initialize preprocessing resources once
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Load or compute embeddings
        self.embeddings = self.__load_embeddings()

    def preprocess_text(self, text):
        """
        Preprocess the text: lowercasing, tokenization, stopword removal, Unicode normalization,
        lemmatization, and punctuation removal.
        (Note: Transformers usually expect raw text. Use this only if you are sure that your
         dataset and queries should be preprocessed similarly.)
        """
        # Tokenization and lowercasing
        tokens = [word.lower() for word in word_tokenize(text)]
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        # Normalize characters and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(
                unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            ) for word in tokens
        ]
        # Remove punctuation tokens
        tokens = [token for token in tokens if token not in punctuation]
        return " ".join(tokens)

    def __load_embeddings(self):
        """
        Load precomputed embeddings if available; otherwise, calculate and save them.
        """
        if os.path.exists(self.embeddings_file):
            print(f"Loading embeddings from {self.embeddings_file}")
            with open(self.embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            # Convert loaded embeddings to a torch tensor on the correct device
            return torch.tensor(embeddings, dtype=torch.float, device=self.device)
        else:
            print("Calculating embeddings...")
            embeddings = self.__calculate_embeddings()
            print(f"Saving embeddings to {self.embeddings_file}")
            # Save as numpy array for easier serialization
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings.cpu().numpy(), f)
            return embeddings

    def __calculate_embeddings(self):
        """
        Calculate embeddings for all movies in the dataset using batch processing.
        """
        all_texts = self.dataframe[self.text_column].tolist()
        embeddings_list = []

        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i:i + self.batch_size]
            batch_embeddings = self.__calculate_batch_embeddings(batch_texts)
            embeddings_list.append(batch_embeddings)
            print(f"Processed batch {i // self.batch_size + 1} / {((len(all_texts) - 1) // self.batch_size) + 1}")

        # Concatenate all batch embeddings along the first dimension
        return torch.cat(embeddings_list, dim=0)

    @torch.no_grad()
    def __calculate_batch_embeddings(self, texts):
        """
        Calculate embeddings for a batch of texts.
        Uses either the CLS token or mean pooling over token embeddings.
        """
        # Optionally preprocess texts; ensure consistency with dataset creation.
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Tokenize batch (handles truncation and padding)
        tokens = self.tokenizer(
            processed_texts, return_tensors='pt', truncation=True, padding=True, max_length=512
        )
        # Move token tensors to the appropriate device
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        outputs = self.model(**tokens)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        attention_mask = tokens['attention_mask']       # Shape: (batch_size, seq_len)

        if self.use_mean_pooling:
            # Mean Pooling: sum embeddings over tokens and divide by number of non-padded tokens.
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            return pooled
        else:
            # Use the embedding of the first token (CLS) as the sentence representation.
            return last_hidden_state[:, 0, :]

    @torch.no_grad()
    def __calculate_text_embedding(self, text):
        """
        Calculate the embedding for a single piece of text.
        """
        processed_text = self.preprocess_text(text)
        tokens = self.tokenizer(processed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        outputs = self.model(**tokens)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = tokens['attention_mask']

        if self.use_mean_pooling:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            return pooled.squeeze(0)
        else:
            return last_hidden_state[:, 0, :].squeeze(0)

    @torch.no_grad()
    def search(self, query, top_n=5):
        """
        Perform a search for the given query and return the top N results.
        :param query: The search query.
        :param top_n: Number of top results to return.
        :return: List of movie IDs corresponding to the top search results.
        """
        # Preprocess and calculate embedding for the query
        query_embedding = self.__calculate_text_embedding(query)  # Shape: (hidden_dim,)

        # Ensure the query embedding is a 2D tensor for cosine similarity computations
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        movie_embeddings_np = self.embeddings.cpu().numpy()

        # Compute cosine similarities between the query and all movie embeddings
        similarities = cosine_similarity(query_embedding_np, movie_embeddings_np)[0]

        # Get indices of the top_n highest similarity scores
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        # Retrieve the corresponding movie IDs from the dataframe
        results = self.dataframe.iloc[top_indices]
        return results['id'].tolist()


# Example usage:
if __name__ == "__main__":
    engine = MovieSearchEngineV2(csv_file='./raw_movies_clean.csv', text_column='title', 
                                 embeddings_file='./raw_embeddings_v2.pkl', use_mean_pooling=True)
    query = "A space adventure with unexpected twists"
    top_movies = engine.search(query, top_n=5)
    print("Top matching movie IDs:", top_movies)
