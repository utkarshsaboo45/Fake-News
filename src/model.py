"""
Author: Utkarsh

Reads processed csv data from path, applies column transformations,
trains models and saves the trained model to disk

Usage: src/model.py --raw_data_path=<raw_data_path>  --processed_data_path=<processed_data_path>

Options:
--raw_data_path=<raw_data_path>                 path to the raw data
--processed_data_path=<processed_data_path>     path to the processed data

Example:
python src/model.py --raw_data_path=data/processed/train.csv  --processed_data_path=data/processed/train.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import altair as alt

from docopt import docopt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import (
    cross_validate,
    train_test_split,
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from wordcloud import WordCloud

alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


def get_feature_types(X_train):
    text_feature_title = [
        "title"
    ]

    text_feature_text = [
        "text"
    ]

    binary_features = [
        "is_title_null",
        "is_text_empty",
        "is_author_null",
        "is_multiple_authors",
        "author_contains_domain",
        "title_contains_famous_journal"
    ]

    numeric_features = [
        "no_of_chars_title",
        "no_of_chars_title",
        "title_word_count",
        "text_word_count",
        "count_noun_title",
        "count_verb_title",
        "count_adj_title",
        "count_noun_text",
        "count_verb_text",
        "count_adj_text",
        "title_text_sim"
    ]

    passthrough_features = []

    drop_features = [
        "id",
        "author",
        "updated_author"
    ]

    assert (
        len(text_feature_title) +
        len(text_feature_text) +
        len(binary_features) +
        len(numeric_features) +
        len(passthrough_features) +
        len(drop_features)
    ) == len(X_train.columns)

    return {
        "text_feature_title": text_feature_title,
        "text_feature_text": text_feature_text,
        "binary_features": binary_features,
        "numeric_features": numeric_features,
        "passthrough_features": passthrough_features,
        "drop_features": drop_features
    }


def column_transformer(X_train):
    function_transformer = FunctionTransformer(
        np.reshape, kw_args={"newshape": -1}
    )

    enc1 = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        function_transformer,
        CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    )

    enc2 = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        function_transformer,
        CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    )

    standard_scalar = make_pipeline(StandardScaler())

    onehotencoder = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse=False, drop="if_binary"))

    fature_types_dict = get_feature_types(X_train)

    column_transformer = make_column_transformer(
        (enc1, fature_types_dict["text_feature_title"]),
        (enc2, fature_types_dict["text_feature_text"]),
        (standard_scalar, fature_types_dict["numeric_features"]),
        (onehotencoder, fature_types_dict["binary_features"]),
        ("passthrough", fature_types_dict["passthrough_features"]),
        ("drop", fature_types_dict["drop_features"])
    )

    column_transformer.fit(X_train)

def get_new_column_names(column_transformer, fature_types_dict):
    return (
        column_transformer.named_transformers_["pipeline-1"].named_steps["countvectorizer"].get_feature_names_out().tolist() + 
        column_transformer.named_transformers_["pipeline-2"].named_steps["countvectorizer"].get_feature_names_out().tolist() +
        fature_types_dict["numeric_features"] +
        fature_types_dict["binary_features"]
    )


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)



def train_base_models(column_transformer, X, y):
    pipe_lr = make_pipeline(column_transformer, LogisticRegression(max_iter=100000))
    pipe_dt = make_pipeline(column_transformer, DecisionTreeClassifier())
    pipe_nb = make_pipeline(
        column_transformer,
        FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
        GaussianNB()
    )
    pipe_svc = make_pipeline(column_transformer, SVC())
    pipe_rf = make_pipeline(column_transformer, RandomForestClassifier())
    pipe_catboost = make_pipeline(column_transformer, CatBoostClassifier(verbose=False))
    pipe_lasso = make_pipeline(column_transformer, Lasso())
    pipe_ridge = make_pipeline(column_transformer, Ridge())

    models = {
        "Logistic Regression": pipe_lr,
        "Lasso": pipe_lasso,
        "Ridge": pipe_ridge,
        "Decision Tree": pipe_dt,
        "NB": pipe_nb,
        "SVC": pipe_svc,
        "Random Forest": pipe_rf,
        "Cat boost": pipe_catboost,
    }

    results = {}

    for name, value in models.items():
        print(f"Start training {name}!")
        results[name] = mean_std_cross_val_scores(
            value, X, y, cv=10, return_train_score=True
        )
        print(f"{name} done!")



def read_data(filepath):
    if os.path.exists(filepath):
        train_df = pd.read_csv(filepath)
        return train_df
    return None


def main(processed_data_in_path):
    train_df = read_data(processed_data_in_path)

    if train_df is None:
        print("Error reading data from given path, returning...")
        return

    train_df_small, val_df = train_test_split(train_df, test_size=0.2, random_state=123)

    X_train, y_train = train_df_small.drop(columns=["label"]), train_df_small["label"]