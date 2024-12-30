import pandas as pd
from src.tfidf.tfidf_vect import TfidfSearch

# Example usage:
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("./preprocessed_movies_clean.csv")

    # Initialize the TfidfSearch object
    tfidf_search = TfidfSearch(df)

    # Fit the model (build the TF-IDF matrix)
    tfidf_search.fit()

    # Query
    query = "star wars"

    # Get top 5 similar documents
    top_documents = tfidf_search.get_top_similar_documents(query, top_n=5)

    # Print the results
    for doc_id, score in top_documents:
        print(f"Document ID: {doc_id}, Similarity: {score:.4f}")

    print('______________________________________________')

    query = "science fiction adventure"

    # Get top 5 similar documents
    top_documents = tfidf_search.get_top_similar_documents(query, top_n=5)

    # Print the results
    for doc_id, score in top_documents:
        print(f"Document ID: {doc_id}, Similarity: {score:.4f}")
