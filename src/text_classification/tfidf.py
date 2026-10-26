from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(train_texts, val_texts, test_texts, **vectorizer_params):

    params = {
        "analyzer": "word",
        "tokenizer": str.split,
        "token_pattern": None,
        "lowercase": False,
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 10000,
        "norm": "l2",
        "sublinear_tf": True,
    }
    params.update(vectorizer_params)
    vectorizer = TfidfVectorizer(**params)

    features = {
        "X_train": vectorizer.fit_transform(train_texts),
        "X_val": vectorizer.transform(val_texts),
        "X_test": vectorizer.transform(test_texts),
    }
    return vectorizer, features
