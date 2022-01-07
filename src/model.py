"""
Author: Utkarsh

Reads processed csv data from path, applies column transformations,
trains models and saves the trained model to disk

Usage: src/model.py --processed_data_in_path=<processed_data_in_path>  --column_transformer_out_path=<column_transformer_out_path> --save_results_path=<save_results_path>

Options:
--processed_data_in_path=<processed_data_in_path>               path to the processed data
--column_transformer_out_path=<column_transformer_out_path>     path to the column transformer pickle file
--save_results_path=<save_results_path>                         path to the results csv

Example:
python src/model.py --processed_data_in_path=data/processed/train.csv  --column_transformer_out_path=models/column_transformer.pkl --save_results_path=results/results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
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


def read_data(filepath):
    if os.path.exists(filepath):
        train_df = pd.read_csv(filepath)
        return train_df
    return None


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
        "no_of_chars_text",
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


def get_column_transformer(X_train, column_transformer_path):
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

    feature_types_dict = get_feature_types(X_train)

    column_transformer = make_column_transformer(
        (enc1, feature_types_dict["text_feature_title"]),
        (enc2, feature_types_dict["text_feature_text"]),
        (standard_scalar, feature_types_dict["numeric_features"]),
        (onehotencoder, feature_types_dict["binary_features"]),
        # ("passthrough", feature_types_dict["passthrough_features"]),
        ("drop", feature_types_dict["drop_features"])
    )

    column_transformer.fit(X_train)

    if not os.path.exists(os.path.dirname(column_transformer_path)):
        os.makedirs(os.path.dirname(column_transformer_path))
    
    pickle.dump(column_transformer, open(column_transformer_path, "wb"))
    
    print("Column Transformer dumped!")

    return column_transformer


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


def get_base_models(column_transformer):
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

    return {
        "Logistic Regression": pipe_lr,
        "Lasso": pipe_lasso,
        "Ridge": pipe_ridge,
        "Decision Tree": pipe_dt,
        "NB": pipe_nb,
        "SVC": pipe_svc,
        "Random Forest": pipe_rf,
        "Cat boost": pipe_catboost
    }


def train_base_models(column_transformer, X, y):
    models = get_base_models(column_transformer)

    results = {}

    for name, value in models.items():
        print(f"Start training {name}!")
        results[name] = mean_std_cross_val_scores(
            value, X, y, cv=5, return_train_score=True, n_jobs=-1
        )
        print(f"{name} done!")
    
    return results


def save_results(results, save_results_path):
    if not os.path.exists(os.path.dirname(save_results_path)):
        os.makedirs(os.path.dirname(save_results_path))

    results.to_csv(save_results_path)


def main(processed_data_in_path, column_transformer_path, save_results_path):
    print("Reading processed data...")
    train_df = read_data(processed_data_in_path)

    if train_df is None:
        print("Error reading data from given path, returning...")
        return

    train_df_small, val_df = train_test_split(train_df, test_size=0.2, random_state=123)

    X_train, y_train = train_df_small.drop(columns=["label"]), train_df_small["label"]

    if os.path.exists(column_transformer_path):
        print("Loading column transformer...")
        column_transformer = pickle.load(open(column_transformer_path, "rb"))
    else:
        print("Building column transformer...")
        column_transformer = get_column_transformer(X_train, column_transformer_path)
    
    print("Training models...")
    results = train_base_models(column_transformer, X_train, y_train)

    print("Saving results...")
    save_results(pd.DataFrame.from_dict(results), save_results_path)
    print("Results saved!")


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(
        opt["--processed_data_in_path"],
        opt["--column_transformer_out_path"],
        opt["--save_results_path"]
    )
