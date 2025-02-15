import os
import pickle
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import nltk

# Ensure required NLTK resources are downloaded (ideally done once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class CombinedMovieSearchEngine:
    def __init__(self,
                 csv_file,
                 title_embeddings_file,
                 overview_embeddings_file,
                 model_name='distilbert-base-cased',
                 batch_size=64,
                 title_weight=0.5,
                 overview_weight=0.5):
        """
        Initialize the combined search engine.
        
        Parameters:
            csv_file (str): Path to the CSV file containing movie data.
            title_embeddings_file (str): Path to the precomputed title embeddings (.pkl).
            overview_embeddings_file (str): Path to the precomputed overview embeddings (.pkl).
            model_name (str): Pretrained transformer model name.
            batch_size (int): Batch size for query encoding.
            title_weight (float): Weight for the title cosine similarity.
            overview_weight (float): Weight for the overview cosine similarity.
        """
        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.title_weight = title_weight
        self.overview_weight = overview_weight

        # Initialize transformer model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Ensure required NLTK resources are downloaded (ideally done once)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        # Load precomputed embeddings from file (assumed to be numpy arrays)
        with open(title_embeddings_file, 'rb') as f:
            self.title_embeddings = pickle.load(f)
        with open(overview_embeddings_file, 'rb') as f:
            self.overview_embeddings = pickle.load(f)
        
        # Preprocessing resources for query text:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_df(self):
        return self.df
    
    def preprocess_text(self, text):
        """
        Preprocess query text: tokenize, lowercase, remove stopwords,
        normalize characters, lemmatize, and remove punctuation.
        """
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [
            unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for word in tokens
        ]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        tokens = [token for token in tokens if token not in punctuation]
        return " ".join(tokens)
    
    @torch.no_grad()
    def __calculate_query_embedding(self, query):
        """
        Compute the transformer-based embedding for the given query.
        Uses mean pooling over the token embeddings (with attention mask).
        """
        processed_query = self.preprocess_text(query)
        tokens = self.tokenizer(
            [processed_query],
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        outputs = self.model(**tokens)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        attention_mask = tokens['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled_embedding = sum_embeddings / sum_mask  # (batch_size, hidden_dim)
        return pooled_embedding

    def search(self, query, top_n=5):
        """
        Given a query, compute the weighted sum of cosine similarities between the query embedding
        and the precomputed title and overview embeddings. Return the top_n movie IDs.
        
        Parameters:
            query (str): The search query.
            top_n (int): Number of top results to return.
        
        Returns:
            list: List of movie IDs from self.df['id'].
        """
        # Compute query embedding (for both title and overview, using the same text)
        query_embedding = self.__calculate_query_embedding(query)
        query_np = query_embedding.cpu().numpy()  # Shape: (1, hidden_dim)
        
        # Compute cosine similarities
        cos_sim_title = cosine_similarity(query_np, self.title_embeddings)[0]
        cos_sim_overview = cosine_similarity(query_np, self.overview_embeddings)[0]
        
        # Compute weighted similarity
        weighted_sim = (self.title_weight * cos_sim_title +
                        self.overview_weight * cos_sim_overview)
        
        # Get indices of the top_n movies (highest weighted similarity)
        top_indices = np.argsort(weighted_sim)[-top_n:][::-1]
        return self.df.iloc[top_indices]['id'].tolist()