import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import unicodedata
from string import punctuation
import pandas as pd

class TfidfSearch:
    def __init__(self, dataframe, text_column='search_content', cache_file='tfidf_cache_new.pkl'):
        """
        Initialize the TF-IDF Search engine.
        :param dataframe: Input pandas DataFrame containing preprocessed text data.
        :param text_column: Column name with the text (already preprocessed) to use.
        :param cache_file: Path to the cache file for saving/loading precomputed data.
        """
        self.df = dataframe.copy()
        self.text_column = text_column
        self.cache_file = cache_file

        # Setup NLTK for query preprocessing only.
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.doc_freq = Counter()
        self.vocabulary = {}

        # Load or build the TF-IDF representation.
        if os.path.exists(cache_file):
            self.__load_cache()
        else:
            self.__build()    # Build vocabulary & compute document frequencies
            self.__fit()      # Compute TF-IDF vectors for all documents
            self.__save_cache()

    def get_df(self):
        return self.df

    def _preprocess_text(self, text):
        """
        Preprocess incoming query text.
        Since the document text is assumed preprocessed, this function is used only for queries.
        """
        tokens = word_tokenize(text)  # Tokenize query text
        tokens = [word.lower() for word in tokens]  # Lowercase
        tokens = [word for word in tokens if word not in self.stop_words]  # Remove stopwords
        # Normalize characters
        tokens = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in tokens]
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # Remove punctuation tokens
        tokens = [token for token in tokens if token not in punctuation]
        return " ".join(tokens)
    
    def __build(self):
        """
        Build the vocabulary from the preprocessed documents and compute document frequencies.
        Assumes each document in self.df[self.text_column] is already preprocessed.
        """
        for doc in self.df[self.text_column]:
            # Split on whitespace since text is preprocessed.
            tokens = doc.split()
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freq[term] += 1
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
        
        # Compute IDF with smoothing.
        num_docs = len(self.df)
        self.idf = {
            term: np.log((1 + num_docs) / (1 + freq)) + 1
            for term, freq in self.doc_freq.items()
        }

    def _compute_tfidf_vector(self, doc):
        """
        Compute the TF-IDF vector for a single document.
        Since the document is preprocessed, we simply split on whitespace.
        """
        tokens = doc.split()
        term_counts = Counter(tokens)
        doc_length = len(tokens)
        indices = []
        values = []
        for term, count in term_counts.items():
            if term in self.vocabulary:
                tf = count / doc_length
                indices.append(self.vocabulary[term])
                values.append(tf * self.idf[term])
        # Create a sparse vector with shape (1, vocabulary_size)
        return csr_matrix((values, (np.zeros(len(indices)), indices)), shape=(1, len(self.vocabulary)))
    
    def __fit(self):
        """
        Compute TF-IDF sparse matrix for the entire dataset.
        """
        rows = [self._compute_tfidf_vector(doc) for doc in self.df[self.text_column]]
        self.tfidf_matrix = vstack(rows).tocsr()

    def __save_cache(self):
        """
        Save precomputed vocabulary, IDF, and TF-IDF matrix to the cache file.
        """
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'idf': self.idf,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    def __load_cache(self):
        """
        Load precomputed vocabulary, IDF, and TF-IDF matrix from the cache file.
        """
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
            self.vocabulary = cache['vocabulary']
            self.idf = cache['idf']
            self.tfidf_matrix = cache['tfidf_matrix']
    
    def transform_query(self, query):
        """
        Preprocess the incoming query and compute its TF-IDF vector.
        """
        # Preprocess the query text
        processed_query = self._preprocess_text(query)
        query_terms = processed_query.split()
        total_terms = len(query_terms)
        if total_terms == 0:
            return csr_matrix((1, len(self.vocabulary)))  # Return empty vector if no terms
        
        # Count term frequencies using Counter
        query_tf = Counter(query_terms)
        # Normalize term frequencies
        for term in query_tf:
            query_tf[term] /= total_terms
        
        # Build the query vector using the precomputed IDF values
        query_tfidf = np.zeros(len(self.vocabulary))
        for term in query_terms:
            if term in self.vocabulary:
                term_index = self.vocabulary[term]
                query_tfidf[term_index] = query_tf[term] * self.idf.get(term, 0)
        
        return csr_matrix(query_tfidf)
    
    def search(self, query, top_n=5):
        """
        Search for the top N most similar documents based on cosine similarity.
        Returns a list of document IDs corresponding to the best matches.
        """
        query_vector_sparse = self.transform_query(query)
        similarity_scores = cosine_similarity(self.tfidf_matrix, query_vector_sparse).flatten()
        
        # Get indices of top N documents (highest similarity scores)
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
        
        # Use pandas iloc for safe indexing; assumes 'id' column exists in dataframe.
        return self.df.iloc[top_indices]['id'].tolist()
