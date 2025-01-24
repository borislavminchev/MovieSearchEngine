import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import unicodedata
import nltk
from elasticsearch import Elasticsearch, helpers

class GamesPreprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        
        self.columns_to_keep = ["ID", "Name", "Subtitle", "Description", "Genres"]
        
        # Initialize NLTK resources and variables
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.columns_to_keep]
        
    def preprocess_text(self, text):
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
    
    def clean_special_characters(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'["]{1,}', '', text)
        text = re.sub(r'^["]|["]$', '', text)
        # text = re.sub(r'[^\w\s\.,!?\'\\"]', '', text)
        return text
    
    def apply_preprocessing(self):
        # Fill missing values in self.df
        self.df = self.df.fillna("")
        self.df = self.df[(self.df != "").all(axis=1)]

        # Create a separate DataFrame for text preprocessing
        temp_df = self.df.copy()
        text_columns = ["Name", "Subtitle", "Description", "Genres"]

        # Preprocess only for search_content
        for col in text_columns:
            temp_df[col] = temp_df[col].apply(self.clean_special_characters).apply(self.preprocess_text)
        
        # Generate search_content column in self.df using processed temp_df
        self.df["search_content"] = temp_df[text_columns].apply(lambda row: " ".join(row), axis=1)

    
    def save_preprocessed_data(self):
        self.df.to_csv(self.output_file, index=False)
    
    def preprocess(self):
        self.load_data()
        self.apply_preprocessing()
        self.save_preprocessed_data()



if __name__ == "__main__":
    input_file = "./src/homework/appstore_games.csv"
    output_file = "./src/homework/games_clean.csv"
    
    preprocessor = GamesPreprocessor(input_file, output_file)
    preprocessor.preprocess()

    es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
    basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
    )


    df = pd.read_csv("./src/homework/games_clean.csv")


    actions = [
        {
            "_index": "games",
            "_id": doc["ID"],
            "_source": doc
        } 
        for doc in df.to_dict(orient="records")
    ]

    helpers.bulk(es, actions)
    
    search = input("Search: ")
    while search != "<end>": 
        query = {
            "query": {
                "match": {
                    "search_content": search
                }
            }
        }

        response = es.search(index="games", body=query)
        print("\n", f"Results for{search}:\n")
        for hit in response["hits"]["hits"][:5]:
            title = hit["_source"].get("Name", "No Title Available")
            print(title)
        
        search = input("\n\nSearch: ")
