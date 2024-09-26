from preprocessing.clean_text import clean_text
import fasttext
from tqdm import tqdm
import pickle
import io

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

def compute_embeddings(path_to_text, save_path, dim = 100):
    '''
    Computes FastText embeddings for a given text file and saves them to a .bin file.
    
    Params:
        path_to_text (str): Path to the .txt file to compute embeddings.
        save_path (str): Path to save the embeddings .bin file.
    
    Returns:
        None
    '''
    
    # Load the text file
    with open(path_to_text, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    cleaned_text = clean_text(text)

    # Save the cleaned text temporarily to pass it to FastText
    with open('temp_cleaned_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Train FastText model
    model = fasttext.train_unsupervised('temp_cleaned_corpus.txt', model='skipgram', dim = dim)


    # Create a list of strings. Each string contains a word and its corresponding vector

    embeddings = []
    for word in model.words:
        vector = model.get_word_vector(word)
        embeddings.append(f"{word} {' '.join(map(str, vector))}")

    # Save the embeddings to a .pkl file 
    save_pickle(embeddings, save_path)

def load_vectors(fname):
    '''
    Loads FastText embeddings from a .bin file.
    
    Params:
        fname (str): Path to the .bin file.
        
    Returns:
        dict: Dictionary containing the embeddings.
    '''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin, total=n, desc="Loading vectors"):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
