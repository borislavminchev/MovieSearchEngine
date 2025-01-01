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
    def __init__(self, csv_file, text_column, model_name='distilbert-base-cased', batch_size=64):
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
        Calculate embeddings for all movies in the dataset using batch processing.
        """
        all_texts = self.dataframe[self.text_column].tolist()
        embeddings = []

        # Process in batches
        for i in range(0, len(all_texts), self.batch_size):
            print(i)
            batch_texts = all_texts[i:i + self.batch_size]
            batch_embeddings = self.__calculate_batch_embeddings(batch_texts)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)  # Combine all batches into a single array

    @torch.no_grad()
    def __calculate_batch_embeddings(self, texts):
        """
        Calculate embeddings for a batch of texts.
        """
        tokens = self.tokenizer(
            texts, return_tensors='pt', truncation=True, padding=True, max_length=512
        )

        # Move tokens and model to GPU if available
        if torch.cuda.is_available():
            tokens = {key: val.cuda() for key, val in tokens.items()}
            self.model = self.model.cuda()

        output = self.model(**tokens)
        # Return only the CLS token embeddings (first token)
        return output.last_hidden_state[:, 0, :].cpu().numpy()

    # def __calculate_embeddings(self):
    #     """
    #     Calculate embeddings for all movies in the dataset.
    #     """
    #     embeddings = []
    #     for i, text in enumerate(self.dataframe[self.text_column]):
    #         if(i == 5000):
    #             break
    #         print(i)
    #         embeddings.append(self.__calculate_text_embedding(text))
    #     return np.array(embeddings)

    #     return np.vstack(embeddings)  # Combine all batches into a single array
    
    @torch.no_grad()
    def __calculate_text_embedding(self, text):
        """
        Calculate the embedding for a single piece of text.
        """
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        output = self.model(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()
    
    @torch.no_grad()
    def search(self, query, top_n=5):
        """
        Perform a search for the given query and return the top N results.
        :param query: The search query.
        :param top_n: Number of top results to return.
        """
        # Preprocess the query
        query = self.preprocess_text(query)
        
        # Use __calculate_batch_embeddings to calculate the embedding for a single query
        query_embedding = self.__calculate_batch_embeddings([query]).squeeze(0)  # Remove batch dimension

        # Convert embeddings to PyTorch tensor if not already
        if not isinstance(self.embeddings, torch.Tensor):
            self.embeddings = torch.tensor(self.embeddings, dtype=torch.float)

        # Move embeddings to the same device as the query embedding
        device = query_embedding.device
        self.embeddings = self.embeddings.to(device)
        
        # Compute similarities
        similarities = self.__calculate_similarity(query_embedding, self.embeddings)
        
        # Sort indices in descending order of similarity
        top_indices = similarities.argsort()[:top_n]  # Use PyTorch sorting
        
        # Convert indices to a list if working with pandas
        top_indices = top_indices.tolist()  # Convert to numpy array
        
        # Retrieve results
        results = self.dataframe.iloc[top_indices]
        return [(row['title'], similarities[i].item()) for i, (_, row) in zip(top_indices, results.iterrows())]



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
