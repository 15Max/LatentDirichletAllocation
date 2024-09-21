import os
import pandas as pd
import string
from collections import Counter
import string
import spacy
from octis.preprocessing.preprocessing import Preprocessing
os.chdir(os.path.pardir)

nlp = spacy.load('en_core_web_sm')

# Load the CSV file
csv_input_path = 'data/raw/cleaned_train_lyrics.csv'  
df = pd.read_csv(csv_input_path)
df = df.drop(columns = ['Unnamed: 0'])
df = df.rename(columns = {'Lyric':'lyrics'})
df = df.sample(frac=0.001).reset_index(drop=True) 
print(len(df))
# Save it in the required TSV format
tsv_output_path ='data/input/train_lyrics.tsv'  
df.to_csv(tsv_output_path, sep='\t', index=False) 
print(f"CSV file has been successfully converted and saved as TSV at {tsv_output_path}.")

# Words to exclude from the vocabulary
custom_stopwords = ['ahh','bridge','chorus','cmon','deh', 'dem', 'doh', 'hey', 'hmm','hook', 'ill','instrumental',' intro','ive', 'jah','ooh', 
                    'ohh', 'oooh','outro' ,'repeat','solo','them','universe','verse', 'was', 'whoa', 'woah',  'yah', 'yeah', 'yeh']
    

word_counts = Counter()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    words = [word for word in words if word not in custom_stopwords and len(word) > 2]
    return words

for lyrics in df['lyrics']:  
    processed_words = preprocess_text(lyrics)
    word_counts.update(processed_words)

min_freq = 2 # Minimum frequency threshold for a word to be included in the vocabulary
vocabulary = [word for word, freq in word_counts.items() if freq >= min_freq]


vocab_output_path = 'data/input/vocabulary.txt'  # Destination path for vocabulary file
with open(vocab_output_path, 'w') as f:
    for word in vocabulary:
        f.write(f"{word}\n")

print(f"Vocabulary saved to {vocab_output_path}.")

# with open(vocab_output_path, 'r') as f:
    #vocab_list = f.read().splitlines()

# Initialize preprocessing
preprocessor = Preprocessing(vocabulary=vocab_output_path, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=3, min_words_docs=2)
# preprocess
dataset = preprocessor.preprocess_dataset(documents_path=tsv_output_path, labels_path=tsv_output_path)

# save the preprocessed dataset
dataset.save('processed')



        
