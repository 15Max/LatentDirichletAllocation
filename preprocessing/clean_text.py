import os
import pandas as pd
import string
from collections import Counter
import string
import spacy
import tqdm
from utils.package_handler import install_spacy_model
install_spacy_model()
from octis.preprocessing.preprocessing import Preprocessing


def convert_to_tsv(csv_input_path, tsv_output_path, frac = 1): #TODO: there can also be a colum with the dataset partition (train, test, val), in the ex. it was the second column

    df = pd.read_csv(csv_input_path)
    df = df.drop(columns = ['Unnamed: 0'])
    df = df.rename(columns = {'Lyric':'lyrics'})
    df = df.sample(frac=frac).reset_index(drop=True) # Uncomment this line to sample a fraction of the dataset
    print('The dataset contains {} songs'.format(len(df)))
    # Save it in the required TSV format
    df.to_csv(tsv_output_path, sep='\t', index=False) 
    print(f"CSV file has been successfully converted and saved as TSV at {tsv_output_path}.")
    return df

# Words to exclude from the vocabulary
custom_stopwords = ['ahh','bridge','chorus','cmon','deh', 'dem', 'doh', 'hey', 'hmm','hook', 'ill','instrumental',' intro','ive', 'jah','ooh', 
                    'ohh', 'oooh','outro' ,'repeat','solo','them','universe','verse', 'was', 'whoa', 'woah',  'yah', 'yeah', 'yeh']
    

vocab_output_path = 'data/input/vocabulary.txt'

def preprocess_text(text, custom_stopwords = custom_stopwords):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.lower().split()
        words = [word for word in words if word not in custom_stopwords and len(word) > 2]
        return words


def build_vocabulary(df, custom_stopwords = custom_stopwords, min_freq=2, output_path=vocab_output_path):
    word_counts = Counter()
    for lyrics in tqdm.tqdm(df['lyrics'], desc="Processing lyrics"):  
        processed_words = preprocess_text(lyrics, custom_stopwords)
        word_counts.update(processed_words)

    vocabulary = [word for word, freq in word_counts.items() if freq >= min_freq]
    with open(output_path, 'w') as f:
        for word in tqdm.tqdm(vocabulary, desc="Saving vocabulary"):
            f.write(f"{word}\n")
    print(f"Vocabulary saved to {output_path}.")
    return vocabulary



# # Initialize preprocessing
# preprocessor = Preprocessing(vocabulary=vocab_output_path, max_features=None,
#                              remove_punctuation=True, punctuation=string.punctuation,
#                              lemmatize=True, stopword_list='english',
#                              min_chars=3, min_words_docs=2)
# # preprocess
# dataset = preprocessor.preprocess_dataset(documents_path=tsv_output_path, labels_path=tsv_output_path)

# # save the preprocessed dataset
# dataset.save('processed')



        
