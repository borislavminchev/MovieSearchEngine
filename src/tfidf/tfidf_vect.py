from collections import Counter
from scipy.sparse import csr_matrix, vstack
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import unicodedata

class TfidfSearch:
    def __init__(self, dataframe, text_column='search_content'):
        # Initialize with the dataframe and the column that contains text data
        self.df = dataframe
        self.text_column = text_column
        
        # Initialize variables for vocabulary, document frequencies, and IDF
        self.vocabulary = {}
        self.doc_freq = Counter()
        self.doc_count = len(self.df)
        self.idf = {}
        
        # NLTK setup
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Preprocess data and build the vocabulary
        self._build()

    def _preprocess_text(self, text):
        # Tokenize, lowercase, remove stopwords, normalize characters, and lemmatize
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word.lower() for word in tokens]  # Lowercasing
        tokens = [word for word in tokens if word not in self.stop_words]  # Stopword removal
        tokens = [
            unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for word in tokens  # Character Normalization
        ]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        tokens = [token for token in tokens if token not in punctuation]
        return " ".join(tokens)
    
    def _build(self):
        # Build vocabulary and calculate document frequencies
        for doc in self.df[self.text_column]:
            terms = set(doc.split())  # Get unique terms in document
            for term in terms:
                self.doc_freq[term] += 1
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
        
        # Compute IDF (Inverse Document Frequency)
        self.idf = {
            term: np.log((1 + self.doc_count) / (1 + freq)) + 1  # Add 1 for smoothing
            for term, freq in self.doc_freq.items()
        }

    def _compute_tfidf_vector(self, doc):
        term_counts = Counter(doc.split())
        doc_length = len(doc.split())
        indices = []
        values = []
        for term, count in term_counts.items():
            if term in self.vocabulary:
                tf = count / doc_length
                indices.append(self.vocabulary[term])
                values.append(tf * self.idf[term])
        return csr_matrix((values, (np.zeros(len(indices)), indices)), shape=(1, len(self.vocabulary)))
    
    def fit(self):
        # Compute TF-IDF sparse matrix for the entire dataset
        rows = [self._compute_tfidf_vector(doc) for doc in self.df[self.text_column]]
        self.tfidf_matrix = csr_matrix(vstack(rows))
    
    def transform_query(self, query):
        # Preprocess the query and compute its TF-IDF vector
        query_terms = self._preprocess_text(query).split()
        query_tf = {}
        total_terms = len(query_terms)
        for term in query_terms:
            query_tf[term] = query_tf.get(term, 0) + 1
        
        # Normalize TF
        for term in query_tf:
            query_tf[term] /= total_terms
        
        # Create a query vector using precomputed IDF
        query_tfidf = np.zeros(len(self.vocabulary))
        for term in query_terms:
            if term in self.vocabulary:
                term_index = self.vocabulary[term]
                query_tfidf[term_index] = query_tf[term] * self.idf.get(term, 0)  # Use global IDF
        
        # Convert query vector to sparse format
        return csr_matrix(query_tfidf)
    
    def get_top_similar_documents(self, query, top_n=5):
        # Get top N similar documents based on cosine similarity
        query_vector_sparse = self.transform_query(query)
        similarity_scores = cosine_similarity(self.tfidf_matrix, query_vector_sparse)
        
        # Get top N similar document indices
        top_indices = similarity_scores.flatten().argsort()[-top_n:][::-1]
        top_documents = [(self.df['id'][index], similarity_scores[index][0]) for index in top_indices]
        
        return top_documents
