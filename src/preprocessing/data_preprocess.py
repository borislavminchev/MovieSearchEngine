import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import unicodedata
import nltk

class MovieDataPreprocessor:
    def __init__(self, input_file, output_file):
        # File paths for input and output data
        self.input_file = input_file
        self.output_file = output_file
        
        # Columns to keep
        self.columns_to_keep = [
            "id", "title", "overview", "genres", 
            "cast", "director"
        ]
        
        # Initialize NLTK resources and variables
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        # Load the dataset
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.columns_to_keep]  # Select relevant columns
        
        
    def preprocess_text(self, text):
        # Preprocess text by tokenizing, normalizing, and lemmatizing
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word.lower() for word in tokens]  # Lowercasing
        tokens = [word for word in tokens if word not in self.stop_words]  # Stopword removal
        tokens = [
            unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for word in tokens  # Character Normalization
        ]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        tokens = [token for token in tokens if token not in punctuation]  # Remove punctuation
        return " ".join(tokens)
    
    def clean_special_characters(self, text):
        # Remove special characters from text
        if not isinstance(text, str):
            return ""
        text = re.sub(r'["]{2,}', '', text)  # Replace multiple quotes
        text = re.sub(r'^["]|["]$', '', text)  # Remove leading/trailing quotes
        text = re.sub(r'[^\w\s\.,!?\'"]', '', text)  # Remove unwanted special characters
        return text
    
    def apply_preprocessing(self):
        # Apply preprocessing steps
        text_columns = ["title", "overview", "genres", "cast", "director"]

        self.df[text_columns] = self.df[text_columns].fillna("").astype(str)  # Fill missing text fields
        
        # Clean and preprocess the text columns
        for col in text_columns[1:]:
            self.df[col] = self.df[col].apply(self.clean_special_characters).apply(self.preprocess_text)
        
        # Process 'title' column
        self.df["title"] = self.df["title"].apply(lambda x: x.lower())
        
        # Combine the text columns to create search content
        self.df["search_content"] = self.df[text_columns].apply(lambda row: " ".join(row), axis=1)
        
        # Replace 'nan' with 'Nan' and remove empty or null entries
        self.df["title"].replace("nan", "Nan", inplace=True)
        self.df = self.df.dropna()  # Remove any rows with NaN values
        self.df = self.df[(self.df != "").all(axis=1)]  # Remove empty rows
    
    def save_preprocessed_data(self):
        # Save the preprocessed data to CSV
        self.df.to_csv(self.output_file, index=False)
    
    def preprocess(self):
        # Encapsulate the entire process
        self.load_data()
        self.apply_preprocessing()
        self.save_preprocessed_data()

# Example usage:
if __name__ == "__main__":
    input_file = "./TMDB_all_movies.csv"
    output_file = "./preprocessed_movies_clean.csv"
    
    preprocessor = MovieDataPreprocessor(input_file, output_file)
    preprocessor.preprocess()




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