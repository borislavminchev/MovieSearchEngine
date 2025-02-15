## This is  Movie Search Engine. From given search query the sytem suggests list of simmilar movies to the search.

### Prerequsites:
1. Download initial dataset https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates and place the CSV in the root of project.
2. After that, run once the following commands:
```console
python -m venv .venv
```

```console
./.venv/Scripts/activate
```

```console
pip install -r ./requirements.txt
```

```console
python ./src/preprocessing/data_preprocess.py
```
3. For Elasttic search to work plase use this mapping
```json
{
  "mappings": {
    "properties": {
      "cast": {
        "type": "text"
      },
      "director": {
        "type": "text"
      },
      "genres": {
        "type": "keyword"
      },
      "id": {
        "type": "integer"
      },
      "overview": {
        "type": "text"
      },
      "search_content": {
        "type": "text",
        "fields": {
            "raw": {
                "type": "keyword"
            }
        }
      },
      "title": {
          "type": "text"
      }
    }
  }
}
```
