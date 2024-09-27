from preprocessing.clean_text import clean_text
from gensim.models import KeyedVectors
from tqdm import tqdm
import pickle

def load_pickle(file):
    '''
    Loads a pickle file.
    
    Params:
        file (str): Path to the pickle file.
    
    Returns:
        obj: Object loaded from the pickle file.
    '''
    
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    
    return obj

def save_pickle(obj, file):
    '''
    Saves an object to a pickle file.
    
    Params:
        obj (obj): Object to save.
        file (str): Path to save the pickle file.
    
    Returns:
        None
    '''
    
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_vectors(file_path):
    '''
    Loads vectors from a fastText model file.
    
    Params:
        file (str): Path to the fastText model file.
    
    Returns:
        dict: Dictionary containing the vectors.
    '''
    
    return KeyedVectors.load_word2vec_format('model/cc.en.300.vec/cc.en.300.vec', binary=False)


import os
import pickle
from tqdm import tqdm  # Assuming tqdm is being used for progress

def extract_relevant_embeddings(vocab_path, model_path, output_dir, verbose=True):
    '''
    Given a vocabulary file path, extracts the relevant embeddings from a fastText model file
    and saves the relevant embeddings as a pickle file in the specified output directory.
    
    Params:
        vocab_path (str): Path to a file containing the vocabulary (one word per line).
        model_path (str): Path to the fastText model file.
        output_dir (str): Path to the directory where the pickle file will be saved.
        verbose (bool): Whether to print debugging information.
    
    Returns:
        list: List of strings containing relevant embeddings in the format "word vector".
    '''

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary from file
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {line.strip() for line in f}  # create a set of words from the file

    if verbose:
        print(f"Loaded {len(vocab)} words from {vocab_path}")

    # Load fastText embeddings from file (assuming load_vectors is defined elsewhere)
    embeddings = load_vectors(model_path)

    if verbose:
        print(f"Loaded embeddings from {model_path}")
    
    relevant_embeddings = []

    for word in tqdm(vocab):
        if word in embeddings:
            vector_str = ' '.join(map(str, embeddings[word]))
            relevant_embeddings.append(f"{word} {vector_str}")

    if verbose:
        print(f"Found {len(relevant_embeddings)} relevant embeddings")

    # Save relevant embeddings to a pickle file
    pickle_file_path = os.path.join(output_dir, 'embeddings.pkl')

    if verbose:
        print(f"Saving relevant embeddings to {pickle_file_path}")

    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(relevant_embeddings, pickle_file)

    if verbose:
        print(f"Saved relevant embeddings to {pickle_file_path}")

    return relevant_embeddings
