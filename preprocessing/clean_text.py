import os
import pandas as pd
import spacy
import re
import zipfile 
from utils.package_handler import install_spacy_model
install_spacy_model()

# Words to exclude from the vocabulary
stop_words = ['ah', 'ahh', 'alright', 'ay', 'aye', 'bridge', 'chorus', 'cmon', 
    'da', 'dee', 'deh', 'dem', 'doh', 'doo', 'duh', 'eh', 'ha', 'hey', 'hmm', 
    'ho', 'hook', 'ill', 'intro', 'instrumental', 'its','ive', 'jah', 'la', 'laa', 'mm', 'mmm', 'mmmh',
    'na', 'nah', 'oh', 'ohh', 'ok', 'okay', 'ooh', 'oooh', 'outro', 'repeat', 
    'solo', 'them', 'tho', 'uh', 'uh-huh', 'uhh', 'universe', 'verse', 'was', 
    'whew', 'whoa', 'whoo', 'wo', 'woah', 'woo', 'yah', 'ye', 'yea', 'yeah', 
    'yeh', 'yo']


# Setting up the preprocessing 
nlp = spacy.load('en_core_web_sm')

spacy_stopwords = nlp.Defaults.stop_words
stop_words.extend(spacy_stopwords)
CUSTOM_STOPWORDS = stop_words


# def clean_text(text, stopwords=CUSTOM_STOPWORDS):
#     '''
#     Cleans the input text by removing special characters, whitespaces and stopword

#     Params:
#         path_to_text (str): Path to the .txt to clean.
#         stopwords (list): List of stopwords to remove from the text. Default is None.
    
#     Returns:
#         str: Cleaned text.
#     '''

#     # Convert to lowercase
#     text = text.lower()

#     # Remove special characters, whitespaces and numbers of strings that have numbers in them
#     text = ' '.join(text.split())
#     text = ''.join(e for e in text if e.isalnum() or e.isspace() or e.isnumeric())

#     # Remove stopwords
#     if stopwords:
#         text = ' '.join([word for word in text.split() if word not in stopwords])
    
#     return text


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
    
    # Ensure the output directory exists or create it
    os.makedirs(output_path, exist_ok=True)

    # Check that frac is a float between 0 and 1
    if not 0 <= frac <= 1:
        raise ValueError("frac must be a float between 0 and 1.")


    # Load the CSV file using pandas, drop useless column, and rename Lyrics to lyrics
    # (We assume to be working with the 500k song dataset)
    df = pd.read_csv(csv_input_path).drop(columns=['Unnamed: 0']).rename(columns={'Lyric': 'lyrics'})
    
    # Check if the required columns exist
    if 'lyrics' not in df.columns or 'genre' not in df.columns:
        raise ValueError("CSV file must contain 'lyrics' and 'genre' columns.")
    
    # Sample a fraction of the dataset if frac < 1
    if frac < 1:
        df = df.sample(frac=frac)
    
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


def filter_songs_by_genre(corpus_path, labels_path, genre):
    output_path = os.path.join(os.path.dirname(corpus_path), f'{genre}_songs.txt')
    
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file, open(labels_path, 'r', encoding='utf-8') as labels_file:
        lyrics = corpus_file.readlines()
        genres = labels_file.readlines()
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for lyric, song_genre in zip(lyrics, genres):
            if song_genre.strip().lower() == genre.lower():
                output_file.write(lyric)
    
    print(f'Filtered lyrics saved to {output_path}')
    return output_path


def extract_corpus_and_labels_from_directory(base_path, output_path='data/input/'):
    """
    Extracts corpus and labels from text files in a directory structure, removes empty lines from text, 
    and saves them as 'corpus.txt' and 'labels.txt'.
    
    Params:
        base_path (str): Path to the base directory containing category subdirectories with .txt files.
        output_path (str): Directory where 'corpus.txt' and 'labels.txt' will be saved.
    
    Returns:
        None
    """
    # Ensure the output directory exists or create it
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize corpus and labels lists
    corpus = []
    labels = []
    
    # Walk through the directory structure
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):  # Only process directories
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.endswith('.txt'):  # Only process .txt files
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # Read lines, strip whitespace, and remove empty lines
                            content = " ".join(line.strip() for line in file if line.strip())
                            
                            if content:  # Only add non-empty content
                                corpus.append(content)
                                labels.append(folder_name)  # Use folder name as label
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
    
    # Define file paths for corpus and labels
    corpus_file = os.path.join(output_path, 'corpus.txt')
    labels_file = os.path.join(output_path, 'labels.txt')
    
    # Save corpus and labels in batch mode
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(corpus))
    print(f"Corpus has been saved to {corpus_file}")
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(labels))
    print(f"Labels have been saved to {labels_file}")

