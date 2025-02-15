from src.semantic import CombinedMovieSearchEngine
from src.tfidf.tfidf_vect import TfidfSearch
import pandas as pd

if __name__ == "__main__":
    csv_file = "./raw_movies_clean.csv"
    title_emb_file = "./embeddings.pkl"
    overview_emb_file = "./embeddings_ov.pkl"
    
    combined_engine = CombinedMovieSearchEngine(
        csv_file=csv_file,
        title_embeddings_file=title_emb_file,
        overview_embeddings_file=overview_emb_file,
        title_weight=0.75,
        overview_weight=0.25
    )

    # df = pd.read_csv("./preprocessed_movies_clean.csv")
    # combined_engine = TfidfSearch(df) 
    
    for i in range(5):
        query = input("Search: ")
        results = combined_engine.search(query, top_n=5)
        print("Top movie IDs:", results)

        movie_titles = []
        for movie_id in results:
            # Assuming the dataframe contains a 'title' column
            title = combined_engine.get_df().loc[combined_engine.get_df()['id'] == movie_id, 'title'].values
            if len(title) > 0:
                movie_titles.append(title[0])
            else:
                movie_titles.append("Unknown Title")
        print("Top movie titles:", movie_titles)
