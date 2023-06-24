# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


def main():
    '''
    Use generated document embeddings to train BERTopic model(s)
    '''
    logger = logging.getLogger(__name__)
    logger.info('performing topic modelling using BERTopic')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # load training data SimCSE document embeddings
    train_simcse_embeddings_path = project_dir.joinpath('data/processed/train_simcse_embeddings.pkl')
    with open(train_simcse_embeddings_path, "rb") as embeddings_input:
      saved_embeddings = pickle.load(embeddings_input)
      train_simcse_embeddings = saved_embeddings['train_simcse_embeddings']

    # get documents
    docs = train['Concatenated']

    # load our fine-tuned Longformer-SimCSE model
    model_simcse_path = project_dir.joinpath('models/longformer-simcse')
    model_simcse = SentenceTransformer(model_simcse_path)

    # parameters for dimensionality reduction (UMAP)
    # the defaults, except random_state is set to 1 for reproducibility
    umap_model = UMAP(n_neighbors = 15, n_components = 5, min_dist = 0.0, 
                      metric = 'cosine', random_state = 1)

    # vectoriser; default in BERTopic except we remove some basic stopwords
    # note that stopword removal takes place after embeddings are generated
    vectoriser = CountVectorizer(stop_words = 'english')

    # fine-tuning of topics by using c-TF-IDF (increases topic quality)
    representation_model = KeyBERTInspired(random_state = 1)

    # BERTopic c-TF-IDF modelling
    topic_model = BERTopic(embedding_model = model_simcse,
                           umap_model = umap_model,
                           vectorizer_model = vectoriser,
                           representation_model = representation_model)
    fitted_topic_model = topic_model.fit(docs, train_simcse_embeddings)

    # save fitted BERTopic model
    save_path = str(project_dir.joinpath(f'models/longformer-simcse-bertopic'))
    fitted_topic_model.save(save_path, serialization = 'pytorch', save_ctfidf = True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()