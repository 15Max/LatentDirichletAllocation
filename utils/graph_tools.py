from octis.dataset.dataset import Dataset
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

color_palette = sns.color_palette("husl", 5)
color_map = {
    'tech': color_palette[0],
    'entertainment': color_palette[1],
    'business': color_palette[2],
    'sport': color_palette[3],
    'politics': color_palette[4]
}


# Plot histograms for the top 10 words per genre using seaborn
def word_frequencies_barplot_dataset(dataset):
    """
    Plot the barplots of word frequencies in the dataset using seaborn.
    :param dataset: Dataset object
    """
    labels = dataset.get_labels()
    docs = dataset.get_corpus()

    df = pd.DataFrame({'document': docs, 'genre': labels})

    genres = df['genre'].unique()
    for genre in genres:
        all_words = [word for doc in df[df['genre'] == genre]['document'] for word in doc]
        word_counts = Counter(all_words).most_common(10)
        words, counts = zip(*word_counts)
        
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=list(words), y=list(counts), palette=[color_map.get(genre, 'gray')], edgecolor='black')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title(f'Top 10 Frequent Words in {genre} Articles')
        plt.xticks(rotation=45)
        
        for bar in ax.patches:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)
        
        plt.show()

def word_frequencies_barplot_dict(results: dict):
    """
    Plot the barplot of word frequencies in each topic using Seaborn color palettes.
    
    :param results: Dictionary containing 'topic-word-matrix' and 'topics'.
    """
    topic_word_matrix = results['topic-word-matrix']  # The probability of each word for each topic
    words = results['topics']  # The top words for each topic
    
    # Number of words to display 
    top_n_words = 10
    
    # Use Seaborn color palette instead of a manual color list
    palette = sns.color_palette("husl", len(topic_word_matrix))  # Creates distinct colors for each topic
    
    # Display the top n words for each topic in a bar plot
    for i, topic in enumerate(topic_word_matrix):
        # Get the top N words and their probabilities
        top_n_words_indices = np.argsort(topic)[::-1][:top_n_words]
        top_n_words_values = [topic[i] for i in top_n_words_indices]
        top_n_words_words = words[i]
        
        # Create horizontal bar plot with Seaborn
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_n_words_values, y=top_n_words_words, color=palette[i], edgecolor='black')
        plt.xlabel("Probability")
        plt.ylabel("Word")
        plt.title(f"Topic {i}")
        
        # Invert y-axis for readability
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()


# Compute cosine similarity between genres
def compute_genre_similarity(dataset: Dataset) -> pd.DataFrame:
    '''
    Compute the cosine similarity between genres based on the TF-IDF representation of the documents
    :param dataset: Dataset object
    :return: DataFrame containing the cosine similarity between genres
    '''

    
    labels = dataset.get_labels()
    docs = dataset.get_corpus()

    df = pd.DataFrame({'document': docs, 'genre': labels})
    
    df['document'] = df['document'].apply(lambda x: ' '.join(x))  # Convert list of words to a single string
    genre_texts = df.groupby('genre')['document'].apply(lambda x: ' '.join(x)).to_dict()
    
    # Now we are working with a dictionary where the keys are genres and the values are the concatenated texts of all documents in that genre

    genres = list(genre_texts.keys())


    # Text into numerical representations based on word frequency.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([genre_texts[genre] for genre in genres])


    # TF = Term Frequency = (Number of times term t appears in a document) / (Total number of terms in the document)
    # IDF = Inverse Document Frequency = log_e(Total number of documents / Number of documents with term t in it)
    # TF-IDF = TF * IDF

    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    similarity_df = pd.DataFrame(similarity_matrix, index=genres, columns=genres)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap='Spectral', fmt='.2f')
    plt.title('Cosine Similarity Heatmap')
    plt.show()
    
    return similarity_df


def word_occurrences(dataset):
    """
    Plot the barplot of the top 10 word occurrences in the BBC news dataset
    :param dataset: Dataset object
    """
    vocab = dataset.get_vocabulary()
    documents = dataset.get_corpus()
    word_occurrences = defaultdict(int)

    for doc in documents:
        for word in vocab:
            word_occurrences[word] += doc.count(word)

    word_count_df = pd.DataFrame.from_dict(word_occurrences, orient='index', columns=['count'])
    word_count_df = word_count_df.reset_index().rename(columns={'index': 'word'})
    
    # Sort and keep only the top 10 most frequent words
    word_count_df = word_count_df.sort_values(by='count', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='word', y='count', data=word_count_df, palette='deep', edgecolor = 'black')
    plt.xlabel('Words')
    plt.ylabel('Occurrences')
    plt.title('Top 10 Word Occurrences in Dataset')
    plt.xticks(rotation=45)
    plt.show()

