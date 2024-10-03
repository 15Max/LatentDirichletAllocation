import os
import pandas as pd
import spacy
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
    
    # Load the CSV file, drop useless column, and rename Lyrics to lyrics
    df = pd.read_csv(csv_input_path)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={'Lyric': 'lyrics'})
    
    # Check if the required columns exist
    if 'lyrics' not in df.columns or 'genre' not in df.columns:
        raise ValueError("CSV file must contain 'lyrics' and 'genre' columns.")
    
    # Sample a fraction of the dataset if frac < 1
    df = df.sample(frac=frac).reset_index(drop=True)
    
    # Extract the 'lyrics' (corpus) and 'genre' (labels) columns
    corpus = df['lyrics'].tolist()
    labels = df['genre'].tolist()

    # Save corpus to 'corpus.txt'
    corpus_file = os.path.join(output_path, 'corpus.txt')
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(f"{line}\n")
    print(f"Corpus has been saved to {corpus_file}")

    # Save labels to 'labels.txt'
    labels_file = os.path.join(output_path, 'labels.txt')
    with open(labels_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels have been saved to {labels_file}")

