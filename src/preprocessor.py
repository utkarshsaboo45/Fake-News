"""
Author: Utkarsh

Reads train csv data from path, preprocess the data, and saves it as a csv file

Usage: src/preprocessor.py --raw_data_path=<raw_data_path>  --processed_data_path=<processed_data_path>

Options:
--raw_data_path=<raw_data_path>                 path to the raw data
--processed_data_path=<processed_data_path>     path to the processed data

Example:
python src/preprocessor.py --raw_data_path=data/raw/train.csv  --processed_data_path=data/processed/train.csv
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import string
import altair as alt

from docopt import docopt

import nltk

# nltk.download('words')
# nltk.download("cmudict")
# nltk.download("vader_lexicon")
# nltk.download("punkt")
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

from nltk import word_tokenize
from nltk import pos_tag

alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


def read_data(filepath):
    if os.path.exists(filepath):
        train_df = pd.read_csv(filepath)
        print("Read raw data")
        return train_df
    return None


# Check if title is null
def is_title_null(data):

    data["is_title_null"] = [
        0 if title == title
        else 1 for title in data["title"]
    ]

    return data


# Check if text is empty
def is_text_empty(data):

    data["is_text_empty"] = [
        1 if text == " " or
        text == "" or
        not text == text
        else 0 for text in data["text"]
    ]

    return data


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


# Create a feature "is_author_null"
def is_author_null(data):
    if data["author"] != data["author"]:
        return 1
    return 0


# Change author null to "Unknown"
def get_author_unknown(data):
    unknown_authors_ids = data.query("author.isnull()")["id"]
    data["updated_author"] = np.where(~data["id"].isin(unknown_authors_ids), data["author"], "Unknown")

    return data


# Others category if value_counts of an author is less than 5
def define_other_authors(data, threshold=5):
    less_frequent = data["updated_author"].value_counts()[data["updated_author"].value_counts() <= threshold].index.tolist()
    data["updated_author"] = np.where(data["updated_author"].isin(less_frequent), "Other", data["updated_author"])

    return data


# Create if "is_multiple_authors"
def is_multiple_authors(data):

    data["is_multiple_authors"] = [
        1 if " and " in str(author) else 0 for author in data["author"]
    ]

    return data


# Check if author name contains a domain suffix
def author_contains_domain(data):

    data["author_contains_domain"] = [
        1 if re.search(r"\.[a-zA-Z]{3}", str(author)) else 0 for author in data["author"]
    ]

    return data


# Check if title ends with a famous journal name
def title_contains_famous_journal(data):

    data["title_contains_famous_journal"] = [
        1 if
        str(title).endswith("the new york times") or
        str(title).endswith("breitbart")
        else 0 for title in data["title"]
    ]

    return data


# Get number of characters
def char_count(data):

    data["no_of_chars_title"] = [
        len(str(title)) for title in data["title"]
    ]

    data["no_of_chars_text"] = [
        len(str(text)) for text in data["text"]
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

    for _, pos in pos_tag(word_tokenize(str(text))):
        if(pos[0] == 'N'):
            noun_count += 1
        if(pos[0] == 'V'):
            verb_count += 1
        if(pos == 'JJ'):
            adj_count += 1
    return noun_count, verb_count, adj_count


# Get Jaccard Similarity between two strings
def get_jaccard_sim(str1, str2):
    str1_set = set(str1.split())
    str2_set = set(str2.split())
    intersection = str1_set.intersection(str2_set)
    return round(float(len(intersection)) / (len(str1_set) + len(str2_set) - len(intersection)), 4)


# Get similarity between title and text fields
def get_title_text_similarity(data):
    data["title_text_sim"] = data.apply(lambda x: get_jaccard_sim(str(x["title"]), str(x["text"])), axis=1)
    return data


# Combine preprocessing steps
def preprocess(train_df):

    print("Begin Preprocessing...")

    train_df = is_title_null(train_df)

    train_df = is_text_empty(train_df)

    train_df["title"] = train_df["title"].apply(clean_data)
    train_df["text"] = train_df["text"].apply(clean_data)

    train_df["is_author_null"] = train_df.apply(lambda x: is_author_null(x), axis=1)

    train_df = get_author_unknown(train_df)

    train_df = define_other_authors(train_df)

    train_df = is_multiple_authors(train_df)

    train_df = author_contains_domain(train_df)

    train_df = title_contains_famous_journal(train_df)

    train_df = char_count(train_df)

    train_df = train_df.assign(title_word_count=train_df["title"].apply(get_text_length))

    train_df = train_df.assign(text_word_count=train_df["text"].apply(get_text_length))
    
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

    train_df = get_title_text_similarity(train_df)

    train_df["text"] = train_df["text"].values.astype("U")

    print("Preprocessing Done!")

    return train_df


def save_csv(data, data_out_path):
    print("Now Saving Data...")

    try:
        if not os.path.exists(os.path.dirname(data_out_path)):
            os.makedirs(os.path.dirname(data_out_path))

        data.to_csv(data_out_path, encoding="utf-8", index=False)
        print("Saved Preprocessed Data!")
    except:
        print("Save Preprocessed Data Failed!")


def main(raw_data_path, processed_data_path):
    if os.path.exists(processed_data_path):
        print("Data already processed. Returning...")
        return
    
    train_df = read_data(raw_data_path)

    if not train_df is None:
        train_df = preprocess(train_df)
        save_csv(train_df, processed_data_path)
    else:
        print("Invalid training data path")


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--raw_data_path"], opt["--processed_data_path"])