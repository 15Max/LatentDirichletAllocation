import spacy
import subprocess
import sys

def install_spacy_model():
    try:
        # Try to load the model
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        # If the model is not found, install it
        print("Model 'en_core_web_sm' not found. Downloading the model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        # Load the model after installation
        nlp = spacy.load('en_core_web_sm')
    return nlp