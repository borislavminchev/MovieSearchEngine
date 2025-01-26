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

class TfidfSearch:
    def __init__(self, dataframe, text_column='search_content', cache_file='tfidf_cache.pkl'):
        """
        Initialize the TF-IDF Search engine.
        :param dataframe: Input pandas DataFrame containing text data.
        :param text_column: Column name with the text to process.
        :param cache_file: Path to the cache file for saving/loading precomputed data.
        """
        self.df = dataframe
        self.text_column = text_column
        self.cache_file = cache_file

        # NLTK setup
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.doc_freq = Counter()
        self.vocabulary = {}

        # Load or build data
        if os.path.exists(cache_file):
            self.__load_cache()
        else:
            self.__build()
            self.__fit()
            self.__save_cache()

    def _preprocess_text(self, text):
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
    
    def __build(self):
        """
        Build the vocabulary and calculate document frequencies.
        """
        for doc in self.df[self.text_column]:
            terms = set(doc.split())  # Get unique terms in document
            for term in terms:
                self.doc_freq[term] += 1
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
        
        # Compute IDF (Inverse Document Frequency)
        self.idf = {
            term: np.log((1 + len(self.df)) / (1 + freq)) + 1  # Add 1 for smoothing
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
    
    def __fit(self):
        """
        Compute TF-IDF sparse matrix for the entire dataset.
        """
        rows = [self._compute_tfidf_vector(doc) for doc in self.df[self.text_column]]
        self.tfidf_matrix = csr_matrix(vstack(rows))

    def __save_cache(self):
        """
        Save precomputed TF-IDF data to the cache file.
        """
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'idf': self.idf,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    def __load_cache(self):
        """
        Load precomputed TF-IDF data from the cache file.
        """
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
            self.vocabulary = cache['vocabulary']
            self.idf = cache['idf']
            self.tfidf_matrix = cache['tfidf_matrix']
    
    def transform_query(self, query):
        """
        Preprocess the query and compute its TF-IDF vector.
        """
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
    
    def search(self, query, top_n=5):
        """
        Search for top N similar documents based on cosine similarity.
        """
        query_vector_sparse = self.transform_query(query)
        similarity_scores = cosine_similarity(self.tfidf_matrix, query_vector_sparse)
        
        # Get top N similar document indices
        top_indices = similarity_scores.flatten().argsort()[-top_n:][::-1]
        
        return [self.df['id'][index] for index in top_indices]
