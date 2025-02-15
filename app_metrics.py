from src.semantic import MovieSearchEngineV2, CombinedMovieSearchEngine  # Assuming we want to test V2
from src.indexing.elastic_search import ElasticsearchEngine
import pandas as pd
from src.tfidf.tfidf_vect import TfidfSearch

class Metrics:
    def __init__(self, search_engine, elasticsearch):
        self.search_engine = search_engine
        self.elasticsearch = elasticsearch

    def __get_metrics(self, result, truth):
        """
        Calculate metrics for search engine evaluation based on set overlap.
        Note: This discards ranking order. For ranking-sensitive evaluation,
        consider using metrics like precision@k, MAP, or nDCG.
        """
        result_set = set(result)
        truth_set = set(truth)
        
        # Compute True Positives, False Positives, and False Negatives.
        tp = len(result_set & truth_set)
        fp = len(result_set - truth_set)
        fn = len(truth_set - result_set)
        
        # Compute precision, recall, and F1-score.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy based on union of result and truth (not typical in IR evaluation).
        total_items = len(result_set | truth_set)
        accuracy = tp / total_items if total_items > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def measure(self, query):
        # Retrieve top 30 documents from both search engines.
        result = self.search_engine.search(query, top_n=30)
        truth = self.elasticsearch.search(query, top_n=30)
        return self.__get_metrics(result, truth)

# Example usage:
if __name__ == "__main__":
    csv_file = "./raw_movies_clean.csv"
    text_column = "overview"  # Using the "overview" column for semantic search.
    
    # Initialize the semantic search engine (MovieSearchEngineV2)
    # search_engine = MovieSearchEngineV2(
    #     csv_file=csv_file,
    #     text_column=text_column,
    #     embeddings_file='./raw_embeddings_v2.pkl',
    #     use_mean_pooling=True
    # )

    # csv_file = "./raw_movies_clean.csv"
    title_emb_file = "./raw_embeddings_v2.pkl"         # Embeddings computed on the "title" column
    overview_emb_file = "./raw_embeddings_ov_v2.pkl"     # Embeddings computed on the "overview" column
    
    search_engine = CombinedMovieSearchEngine(
        csv_file=csv_file,
        title_embeddings_file=title_emb_file,
        overview_embeddings_file=overview_emb_file,
        title_weight=0.25,
        overview_weight=0.75
    )
    
    # Initialize the Elastic Search engine (ground truth).
    elastic = ElasticsearchEngine()
    
    # Load the preprocessed dataframe for TF-IDF search.
    df = pd.read_csv("./preprocessed_movies_clean.csv")
    tfidf_search = TfidfSearch(df)  # Ensure df has the column expected by TfidfSearch.
    
    query = "boxing"
    
    # Measure metrics for the semantic search engine.
    metrics_semantic = Metrics(search_engine, elastic)
    print("Semantic Search Metrics:", metrics_semantic.measure(query))
    
    # Measure metrics for the TF-IDF search engine.
    metrics_tfidf = Metrics(tfidf_search, elastic)
    print("TF-IDF Search Metrics:", metrics_tfidf.measure(query))
