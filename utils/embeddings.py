
import torch
import pickle
import numpy as np
import os 
from transformers import BertTokenizer, BertModel




#### Bert embeddings functions ####

# Function to save embeddings using pickle
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Function to load embeddings using pickle
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
# Function to get embeddings for a batch of texts, ensuring inputs are on the correct device
def get_embeddings_batch(texts, tokenizer, model,device):

    # Tokenize and move inputs to the GPU
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        # Perform forward pass on the GPU
        outputs = model(**inputs)
    # Move the output back to CPU for further processing or storage
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# Create embeddings function (ensure model and inputs are on the correct device)
def create_embeddings(df, embeddings_file, force_creation=False, batch_size=32):
    if not force_creation and os.path.exists(embeddings_file):
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = load_embeddings(embeddings_file)
        print(f"Embeddings loaded with shape: {embeddings.shape}")
        return embeddings
    else:
        print("Embeddings not found, generating...")

    # Initialize an empty list to store the embeddings
    embeddings = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(next(model.parameters()).device)

    end = len(df['lyrics'])
    for i in range(0, end, batch_size):
        batch_texts = df['lyrics'][i:i+batch_size].tolist()  # Get batch of texts
        embeddings_batch = get_embeddings_batch(batch_texts, tokenizer, model,device)
        embeddings.append(embeddings_batch)

    # Convert list of batches to numpy array
    embeddings = np.vstack(embeddings)  # Stack the batches into a single array

    # Save the embeddings to a pickle file
    save_embeddings(embeddings, embeddings_file)
    print(f"Embeddings saved to {embeddings_file}")

    return embeddings