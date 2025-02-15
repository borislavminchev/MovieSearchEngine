from src.indexing import ElasticsearchEngine


import pandas as pd

# Example usage:
if __name__ == "__main__":
    csv_file = "./raw_movies_clean.csv"

    df = pd.read_csv("./raw_movies_clean.csv")
    engine = ElasticsearchEngine() 
    
    for i in range(5):
        query = input("Search: ")
        results = engine.search(query, top_n=15)
        print("Top movie IDs:", results)

        movie_titles = []
        for movie_id in results:

            title = df.loc[df['id'] == movie_id, 'title'].values
            if len(title) > 0:
                movie_titles.append(title[0])
            else:
                movie_titles.append("Unknown Title")
        print("Top movie titles:", movie_titles)
