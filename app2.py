import pandas as pd
from src.tfidf.tfidf_vect import TfidfSearch
from src.indexing.elastic_search import ElasticsearchEngine

# Example usage:
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("./preprocessed_movies_clean.csv")

    # Initialize the TfidfSearch object
    tfidf_search = TfidfSearch(df)

    # Query
    query = "star war"

    # Get top 5 similar documents
    top_documents = tfidf_search.search(query, top_n=5)

    # Print the results
    for doc_id in top_documents:
        print(f"Document ID: {doc_id}")

    print('______________________________________________')

    query = "science fiction adventure"

    # Get top 5 similar documents
    top_documents = tfidf_search.search(query, top_n=5)

    # Print the results
    for doc_id in top_documents:
        print(f"Document ID: {doc_id}")

    print('______________________________________________')

    elastic = ElasticsearchEngine()
    res = elastic.search("star")

    for doc_id in res:
        print(f"Document ID: {doc_id}")