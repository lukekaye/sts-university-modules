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
    Use generated document embeddings to train BERTopic models
    All fine-tuned model embeddings are required for this script to work
    output saved to ../models
    '''
    logger = logging.getLogger(__name__)
    logger.info('performing topic modelling using BERTopic')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # load training data document embeddings
    train_embeddings_path = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_path, "rb") as embeddings_input:
      saved_embeddings = pickle.load(embeddings_input)
      train_simcse_embeddings = saved_embeddings['train_simcse_embeddings']
      train_ct_embeddings = saved_embeddings['train_ct_embeddings']

    # get documents
    docs = train['Concatenated']

    # load our fine-tuned models
    model_simcse_path = project_dir.joinpath('models/longformer-simcse')
    model_simcse = SentenceTransformer(model_simcse_path)
    model_ct_path = project_dir.joinpath('models/longformer-ct')
    model_ct = SentenceTransformer(model_ct_path)

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
    # SimCSE, CT with in-batch negatives
    topic_model_simcse = BERTopic(embedding_model = model_simcse,
                                  umap_model = umap_model,
                                  vectorizer_model = vectoriser,
                                  representation_model = representation_model)
    topic_model_ct = BERTopic(embedding_model = model_ct,
                              umap_model = umap_model,
                              vectorizer_model = vectoriser,
                              representation_model = representation_model)
    fitted_topic_model_simcse = topic_model_simcse.fit(docs, train_simcse_embeddings)
    fitted_topic_model_ct = topic_model_ct.fit(docs, train_ct_embeddings)

    # save fitted BERTopic models
    save_path_simcse = str(project_dir.joinpath(f'models/longformer-simcse-bertopic'))
    save_path_ct = str(project_dir.joinpath(f'models/longformer-ct-bertopic'))
    fitted_topic_model_simcse.save(save_path_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_ct.save(save_path_ct, serialization = 'pytorch', save_ctfidf = True)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()