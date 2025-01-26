from elasticsearch import Elasticsearch

class ElasticsearchEngine:
    def __init__(self):
        self.es = Elasticsearch(
            "https://localhost:9200",
            ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
            basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
        )

    def search(self, search_string, top_n):
        query = {
            "query": {
                "match": {
                    "search_content": search_string
                }
            }
        }

        response = self.es.search(index="movies", body=query, size=top_n)
        return [hit["_source"]['id'] for hit in response["hits"]["hits"][:top_n]]
  