## Latent Diriclet Allocation (LDA)
This project explores Latent Dirichlet Allocation (LDA), applied to the [BBC News dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive).
This dataset contains 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. The categories are: business, entertainment, politics, sport, and tech.
The choice of this datasset is also related to the limited computational resources at our disposal. in fact both the data pre-processing and model training were carried out on a NVIDIA GeForce RTX 4060 Laptop GPU.

The main goal of our project is to identify the topics in these news articles.
To carry out this task, also known as topic modeling, we will use LDA, a classical techinque in the field of Natural Language Processing (NLP) and ProdLDA, a more advanced version of LDA that leverages the product of experts to improve the interpretability of the topics identified. 
After computing the topic distributions, they will be visualized and interpreted to understand the main themes present in the dataset.

## Project structure

```bash
TopicModelComparison/
│
├── data/
│   ├── input/                  # Data ready for preprocessing
│   ├── raw/                    # Raw, unprocessed data (optional)
│   └── processed/              # Processed and cleaned data ready for modeling
│
├── images/                     # Images used in the README
│   
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis (EDA) notebook
│   ├── LDA_BBC.ipynb           # Jupyter notebook for LDA experiments and testing
│   └── PROD_BBC.ipynb          # Jupyter notebook for ProdLDA experiments and testing
│
├── papers/
│   ├──OCTIS.pdf                # Main paper for  OCTIS library (OCTIS)
│   ├──LDA.pdf                  # Main paper for Latent Dirichlet Allocation (LDA)
│   ├──ProdLDA.pdf              # Main paper for Product of experts LDA (ProdLDA)
│   └──TM_Survey.pdf            # Paper containing a survey for topic model techniques
│
├── preprocessing/
│   ├── __init__.py             # Initialize the preprocessing package
│   └── clean_text.py           # Text cleaning, tokenization, stopword removal, etc.
│
├── requirements/
│   ├── environment.yml 
│   └── requirements.txt
│
├── results/
│   ├── test_LDA/               # Results of LDA model 
│   └── test_ProdLDA/           # Results of ProdLDA model
├── utils/
│   ├── __init__.py             # Initialize the utils package
│   ├── graph_tools.py          # Helper functions to visualize results
│   └── package_handler.py      # Helper functions to manage packages
│
├── .gitignore                  # Files and directories to be ignored by git
│
└── README.md                   # Project documentation
```

## [OCTIS](references/OCTIS.pdf)
This project was mainly developed using the OCTIS library, a python library thought to facilitate the comparison of topic modelling techniques.
There are adequate tools to preprocess the data, train the models, evaluate them and visualize the results.
Both LDA and ProdLDA were pre-implemented in the octis library, with refernce to the original papers.

## Pre-processing
We used the [SpaCy library](https://spacy.io/usage/linguistic-features) to remove stopwords, numeric chars and punctuation, along with a custom function to pre-treat sentences.
We also added some custom stopwords that were not present in the SpaCy library, such as ....
These words, while normally present in news articles, are irrelevant for distinguishing topics from one another.
After these steps, we leveraged a custom function to build the corpus and the labels in the format required by the [OCTIS library](https://github.com/MIND-Lab/OCTIS/tree/master/octis).
Finally we leveraged the OCTIS preprocessing tool for lemmatization and tokenization, while keeping only words with more than three characters, minimum ten appearances in the dataset and maximum frequency of 0.85.

## [LDA applied to BBC news articles](notebooks/LDA_BBC.ipynb)
Latent Dirichlet Allocation (LDA) is a generative probabilistic model that discovers hidden topics in a collection of documents by assuming each document is a mixture of topics, and each topic is a distribution over words.
The model is based on the bag of words assumption and the De Finetti exchangeability theorem. 
Its latent variables are the topic-word distribution and the document-topic distribution.
The following is the representation of the model as a pgm:

![alt text](images/LDA.png)


In our case we used the [OCTIS implementation of LDA](https://github.com/MIND-Lab/OCTIS/blob/master/octis/models/LDA.py) that is based on its corresponding implementation in gensim.
The model parameters are:
- num_topics: the number of topics to identify in the data
- alpha: the prior on the document topic distribution
- passes: the number of passes through the data
- iterations: the number of iterations to run the algorithm
- update_every: the number of documents to process before updating the model parameters
- chunksize: the number of documents to use in each iteration

### Metrics
The metrics we chose to evaluate this model and also the ProdLDA model are topic coherence and topic diversity.
Topic coherence is a measure of how interpretable the topics are, it is a value between 0 and 1 where 1 is the best value, while topic diversity is a measure of how different the topics are from one another and it is also on a scale from 0 to 1.
Specifically it's defined as the ratio of the number of unique words in the topics to the total number of words in the topics.
There is a trade-off between these two metrics, as increasing the number of topics will increase the diversity but decrease the coherence of the topics.

Given the nature of this task, human judgement is also important to evaluate the topics identified by the model which is why we plotted the word clouds and the top 10 word distibution for each topic.

##Add mathemathical formulation?


#TODO: Discusss hyperparamenter tuning, choice of number of topics





#TODO: add topic distributions and word clouds, try to intepret



## [ProdLDA](references/ProdLDA.pdf) 





## Conclusions




## Conda environment setup
There is a [.yml file](environment.yml) containing all the neccesary packages to run the code we developed.
Some useful commands:
* Exporting environment:
  ```
  conda env export --from-history > environment.yml
  ```
 
* Recreating environment:
  ```
  conda env create -f environment.yml
  ```
* Updating environment:
  ```
  conda env update -f environment.yml --prune
  ```
  
## References
- [Latent Dirichlet Allocation, Blei et. alt. (2003);](/References/Main_paper.pdf)
- [Latent Dirichlet allocation (LDA) and topic modeling: models, applications, a survey, Jelodar et alt. (2018);](/References/LDA_survey.pdf)
- [Autoencoding Variational Inference for topic models, Srivastava enad Sutton (2017)](references/ProdLDA.pdf)


## Authors
- [Nicola Cortinovis](https://github.com/NicolaCortinovis)
- [Edoardo Cortolezzis](https://github.com/EdoardoCortolezzis)
- [Marta Lucas](https://github.com/15Max)

Checkout our brief [presentation](presentation.pdf) for a quick overview of the project. #TODO: add corect link

