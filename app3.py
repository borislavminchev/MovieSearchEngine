from src.semantic.moviesearch import MovieSearchEngine
from src.indexing.elastic_search import ElasticsearchEngine
import pandas as pd
from src.tfidf.tfidf_vect import TfidfSearch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Metrics:
    def __init__(self, search_engine, elasticsearch):
        self.search_engine = search_engine
        self.elasticsearch = elasticsearch

    

    def __get_metrics(self, result, truth):
        """
        Calculate metrics for search engine evaluation.
        
        Parameters:
            result (set or list): The IDs of documents retrieved by the search engine.
            truth (set or list): The IDs of documents in the ground truth.
        
        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1-score.
        """
        # Convert inputs to sets for comparison
        result_set = set(result)
        truth_set = set(truth)
        
        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
        tp = len(result_set & truth_set)  # Intersection of result and truth
        fp = len(result_set - truth_set)  # Result items not in truth
        fn = len(truth_set - result_set)  # Truth items not in result
        tn = 0  # Optional for search engines; typically not applicable
        
        # Calculate precision, recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy (optional, as TN is not typically computed in search scenarios)
        total_items = len(result_set | truth_set)  # Union of result and truth
        accuracy = tp / total_items if total_items > 0 else 0.0
        
        # Return metrics as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }


    def measure(self, query):
        result = self.search_engine.search(query, top_n=30)
        truth = self.elasticsearch.search(query, top_n=30)
        print(result, '\n', truth)
        return self.__get_metrics(result, truth)
    


# Example usage
if __name__ == "__main__":
    # Path to the preprocessed CSV
    csv_file = "./raw_movies_clean.csv"
    text_column = "overview"

    # Initialize the search engine
    search_engine = MovieSearchEngine(csv_file, text_column, embeddings_file='./embeddings_ov.pkl')

    # Search for a query
    query = "lightsaber jedi"
    # top_results = search_engine.search(query, top_n=5)

    # print(top_results)
    elastic = ElasticsearchEngine()
    # res = elastic.search("star")

    df = pd.read_csv("./preprocessed_movies_clean.csv")

    # Initialize the TfidfSearch object
    tfidf_search = TfidfSearch(df)


    metrics = Metrics(search_engine, elastic)
    print(metrics.measure(query))
    metrics = Metrics(tfidf_search, elastic)
    print(metrics.measure(query))

