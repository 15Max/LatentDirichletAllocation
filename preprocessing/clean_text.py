import os
import pandas as pd
import spacy
import re
import zipfile 
from utils.package_handler import install_spacy_model
install_spacy_model()

# Words to exclude from the vocabulary
stop_words = ['year', 'month', 'day']

# Setting up the preprocessing 
nlp = spacy.load('en_core_web_sm')

spacy_stopwords = nlp.Defaults.stop_words
stop_words.extend(spacy_stopwords)
CUSTOM_STOPWORDS = stop_words


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

