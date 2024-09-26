import os
import pandas as pd
import spacy
import re
import dask.dataframe as dd
from utils.package_handler import install_spacy_model
install_spacy_model()

# Words to exclude from the vocabulary
stop_words = ['ah', 'ahh', 'alright', 'ay', 'aye', 'bridge', 'chorus', 'cmon', 
    'da', 'dee', 'deh', 'dem', 'doh', 'doo', 'duh', 'eh', 'ha', 'hey', 'hmm', 
    'ho', 'hook', 'ill', 'intro', 'instrumental', 'ive', 'jah', 'la', 'laa', 'mm', 'mmm', 
    'na', 'nah', 'oh', 'ohh', 'ok', 'okay', 'ooh', 'oooh', 'outro', 'repeat', 
    'solo', 'them', 'tho', 'uh', 'uh-huh', 'uhh', 'universe', 'verse', 'was', 
    'whew', 'whoa', 'whoo', 'wo', 'woah', 'woo', 'yah', 'ye', 'yea', 'yeah', 
    'yeh', 'yo']


# Setting up the preprocessing 
nlp = spacy.load('en_core_web_sm')

spacy_stopwords = nlp.Defaults.stop_words

CUSTOM_STOPWORDS = stop_words.extend(spacy_stopwords)

def clean_text(text, stopwords=CUSTOM_STOPWORDS):
    '''
    Cleans the input text by removing special characters, whitespaces and stopword

    Params:
        path_to_text (str): Path to the .txt to clean.
        stopwords (list): List of stopwords to remove from the text. Default is None.
    
    Returns:
        str: Cleaned text.
    '''

    # Convert to lowercase
    text = text.lower()

    # Remove special characters, whitespaces and numbers of strings that have numbers in them
    text = ' '.join(text.split())
    text = ''.join(e for e in text if e.isalnum() or e.isspace() or e.isnumeric())

    # Remove stopwords
    if stopwords:
        text = ' '.join([word for word in text.split() if word not in stopwords])
    
    return text


def extract_corpus_and_labels_from_songs_csv(csv_input_path, output_path='data/input/', frac=1):
    '''
    Extracts corpus and labels from a CSV file and saves them as two .txt files: 'corpus.txt' and 'labels.txt'.
    
    Params:
        csv_input_path (str): Path to the CSV file that contains the data.
        output_path (str): Directory where the 'corpus.txt' and 'labels.txt' will be saved. Default is 'data/input/'.
        frac (float): Fraction of the dataset to sample (e.g., frac=0.5 means 50% of the data will be used).
    
    Returns:
        None
    '''
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Load the CSV file in parallel using Dask, drop useless column, and rename Lyrics to lyrics
    ddf = dd.read_csv(csv_input_path).drop(columns=['Unnamed: 0']).rename(columns={'Lyric': 'lyrics'})
    
    # Check if the required columns exist
    if 'lyrics' not in ddf.columns or 'genre' not in ddf.columns:
        raise ValueError("CSV file must contain 'lyrics' and 'genre' columns.")
    
    # Sample a fraction of the dataset if frac < 1
    if frac < 1:
        ddf = ddf.sample(frac=frac)
    
    # Compute the DataFrame to load it into memory
    df = ddf.compute()

    # Extract the 'lyrics' (corpus) and 'genre' (labels) columns
    corpus = df['lyrics'].tolist()
    labels = df['genre'].tolist()

    # Save corpus and labels in batch mode
    corpus_file = os.path.join(output_path, 'corpus.txt')
    labels_file = os.path.join(output_path, 'labels.txt')

    # Save corpus in batch mode
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(corpus))
    print(f"Corpus has been saved to {corpus_file}")

    # Save labels in batch mode
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(labels))
    print(f"Labels have been saved to {labels_file}")