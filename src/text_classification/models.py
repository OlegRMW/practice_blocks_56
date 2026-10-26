from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_logistic_regression(X_train, y_train, seed, **model_params):

    params = {"max_iter": 1000, "random_state": seed}
    params.update(model_params)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train, y_train, seed, svd_components=100, **model_params
):
    n_components = min(svd_components, X_train.shape[1], X_train.shape[0] - 1)
    params = {"random_state": seed}
    params.update(model_params)
    model = Pipeline([
        ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
        ("classifier", GradientBoostingClassifier(**params)),
    ])
    model.fit(X_train, y_train)
    return model
