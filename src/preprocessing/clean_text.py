import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle


nlp = spacy.load("en_core_web_sm")
custom_stopwords = set()  # TODO:Add custom stopwords here

input_path = "data/raw" 
output_path = "data/processed" #TODO: complete path to the output directory

def preprocess(docs, verbose=True, custom_stopwords=set()):
    """
    Preprocess text data and return the cleaned data in the same document format.
    Args:
        docs: list of strings
        verbose: bool
    Returns:    
        cleaned_docs: list of strings
    """

    nlp.Defaults.stop_words |= custom_stopwords
    
    if verbose:
        print("Preprocessing text data...")

    cleaned_docs = []
    
    for doc in tqdm(docs):
        doc = nlp(doc)
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        cleaned_docs.append(' '.join(tokens))

    if verbose:
        print("Preprocessing complete.")
    
    return cleaned_docs

 

def preprocess_file(input_file= input_path, output_file= output_path, verbose=True, text_col="text", custom_stopwords=custom_stopwords):
    """
    Preprocess text data and save the cleaned data in the same document format.
    Args:
        input_file: str
        output_file: str
        verbose: bool
    Returns:
        None
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found.")
    
    if verbose:
        print("Reading data...")

    df = pd.read_csv(input_file) # NOTE: This might need to be changed if the input file is not a csv file
    
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in the input file.")
    
    # warning if custom_stopwords is empty
    if len(custom_stopwords) == 0:
        print("Warning: No custom stopwords provided.")
   
    preprocessed_text = preprocess(df[text_col].values, verbose=verbose, custom_stopwords=custom_stopwords)
    df[text_col] = preprocessed_text

    if verbose:
        print("Saving preprocessed data...")
    
    df.to_csv(output_file, index=False) # NOTE: This might need to be changed if the input file is not a csv file



        

 