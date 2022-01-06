# Author: Utkarsh
"""
Reads train csv data from path, preprocess the data, build a preprocessor pipeline and output the preprocessor
Usage: preprocessor.py --train_set_path=<train_set_path>  --column_transformer_out_path=<column_transformer_out_path> 
Options:
--train_set_path=<train_set_path>                               path to the train set
--column_transformer_out_path=<column_transformer_out_path>     path to column transformer
Example:
python src/preprocessor.py --train_set_path=data/processed/train_df.csv  --column_transformer_out_path=models
"""


import pandas as pd
import numpy as np
import os
from os import path
import re
import string
import altair as alt
import matplotlib.pyplot as plt

from collections import Counter

import nltk

# nltk.download('words')
# nltk.download("cmudict")
# nltk.download("vader_lexicon")
# nltk.download("punkt")
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import cmudict, stopwords


alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


def read_data():
    if os.path.exists( "../data/processed/train.csv"):
        train_df = pd.read_csv("../data/processed/train.csv")
        print("Read processed data")
    else:
        train_df = pd.read_csv("../data/raw/train.csv")
        print("Read raw data")


def remove_punctuation(word):
    punctuations = string.punctuation
    punctuations += "“”–\n"

    for element in word:
        if element in punctuations:
            word = word.replace(element, "")
    return word


def clean_data(text):
    text = str(text).lower()
    text = str(text).strip()
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = remove_punctuation(text)
    return text


# Create a feature "author_is_null"
def author_is_null(x):
    if x["author"] != x["author"]:
        return 0
    return 1


# Change author null to "Unknown"
def get_author_unknown():
    unknown_authors_ids = train_df.query("author.isnull()")["id"]
    train_df['updated_author_column_name'] = np.where(~train_df['id'].isin(unknown_authors_ids), train_df['author'], 'Unknown')


# Others category if value_counts of an author is less than 5
def define_other_authors():
    less_frequent = train_df['author'].value_counts()[train_df['author'].value_counts() <= 5].index.tolist()
    train_df['author'] = np.where(train_df['author'].isin(less_frequent), 'Other', train_df['author'])


# Create if "is_multiple_authors"
def is_multiple_authors(data):

    data["is_multiple_authors"] = [
        1 if " and " in str(author) else 0 for author in data["author"]
    ]

    return data


# Check if author name contains a domain suffix
def author_contains_domain(data):

    data["author_contains_domain"] = [
        1 if re.search(r"\.[a-zA-Z]{3}", str(author)) else 0 for author in train_df["author"]
    ]

    return data


# Check if title is null
def is_title_null(data):

    data["is_title_null"] = [
        0 if title == title
        else 1 for title in train_df["title"]
    ]

    return data


# Check if title ends with a famous journal name
def title_contains_famous_journal(data):

    data["title_contains_famous_journal"] = [
        1 if
        str(title).endswith("The New York Times") or
        str(title).endswith("Breitbart")
        else 0 for title in train_df["title"]
    ]

    return data


# Get number of words
def no_of_words(data):

    data["no_of_words"] = [
        len(str(title).split(" ")) for title in train_df["title"]
    ]

    return data


# Get number of characters
def no_of_chars(data):

    data["no_of_chars"] = [
        len(str(title)) for title in train_df["title"]
    ]

    return data

# Get number of words in a text
def get_text_length(text):
    """
    Returns the number of words in a text without punctuations. 
    Counts clitics as separate words.

    Parameters
    ----------
    text : str
        A text from which we find the number of words

    Returns
    -------
    An int which represents the number of words in the text
    """
    non_punc = []
    for word in word_tokenize(str(text)):
        if word not in string.punctuation:
            non_punc.append(word)
    return len(non_punc)


# Get counts of Nouns, Verbs and Adjs
def get_pos_count(text):
    """
    Counts the number of nouns, verbs and adjectives in a text.

    Parameters
    ----------
    text : str
        A text for which we find the number of nouns, verbs
        and adjectives

    Returns
    -------
    A tuple of (noun_count: int, verb_count: int, adj_count: int)
    which represents the number of nouns, verbs adjectives in the text
    respectively
    """
    noun_count = 0
    verb_count = 0
    adj_count = 0

    if len(str(text)) == 0:
        return 0, 0, 0

    for word, pos in pos_tag(word_tokenize(str(text))):
        if(pos[0] == 'N'):
            noun_count += 1
        if(pos[0] == 'V'):
            verb_count += 1
        if(pos == 'JJ'):
            adj_count += 1
    return noun_count, verb_count, adj_count


# Check if text is empty
def is_text_empty(data):

    data["is_text_empty"] = [
        1 if text == " " or
        not text == text
        else 0 for text in train_df["text"]
    ]

    return data


# Get Jaccard Similarity between two strings
def get_jaccard_sim(str1, str2):
    str1_set = set(str1.split())
    str2_set = set(str2.split())
    intersection = str1_set.intersection(str2_set)
    return float(len(intersection)) / (len(str1_set) + len(str2_set) - len(intersection))


# Get similarity between title and text fields
def get_title_text_similarity(data):
    data["title_text_sim"] = data.apply(lambda x: get_jaccard_sim(str(x["title"]), str(x["text"])), axis=1)
    return data







# Combine preprocessing steps
def preprocess():
    train_df["title"] = train_df["title"].apply(clean_data)
    train_df["title"] = train_df["title"].apply(clean_data)

    train_df["author_is_null"] = train_df.apply(lambda x: author_is_null(x), axis=1)

    train_df = is_multiple_authors(train_df)
    train_df.query("is_multiple_authors == 1")["label"].value_counts()

    train_df = author_contains_domain(train_df)
    train_df.query("author_contains_domain == 1")["label"].value_counts()

    train_df = is_title_null(train_df)
    train_df.query("is_title_null == 1")["label"].value_counts()

    train_df = title_contains_famous_journal(train_df)
    train_df.query("title_contains_famous_journal == 1")["label"].value_counts()

    train_df = no_of_words(train_df)

    train_df = no_of_chars(train_df)

    train_df = train_df.assign(title_len=train_df["title"].apply(get_text_length))
    
    train_df["text_len"] = train_df["text"].apply(get_text_length)

    pos_title = train_df["title"].apply(get_pos_count)
    train_df["count_noun_title"], train_df["count_verb_title"], train_df["count_adj_title"] = (
        pos_title.str[0],
        pos_title.str[1],
        pos_title.str[2]
    )

    pos_text = train_df["text"].apply(get_pos_count)
    train_df["count_noun_text"], train_df["count_verb_text"], train_df["count_adj_text"] = (
        pos_text.str[0],
        pos_text.str[1],
        pos_text.str[2]
    )

    train_df = is_text_empty(train_df)
    train_df.query("text == ' '")["label"].value_counts()

    train_df = get_title_text_similarity(train_df)

    train_df["text"] = train_df["text"].values.astype("U")