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
      train_longformer_simcse_embeddings = saved_embeddings['train_longformer_simcse_embeddings']
      train_longformer_ct_embeddings = saved_embeddings['train_longformer_ct_embeddings']
      train_bigbird_tsdae_embeddings = saved_embeddings['train_bigbird_tsdae_embeddings']

    # get documents
    docs = train['Concatenated']

    # load our fine-tuned models
    model_longformer_simcse_path = project_dir.joinpath('models/longformer-simcse')
    model_longformer_simcse = SentenceTransformer(model_longformer_simcse_path)
    model_longformer_ct_path = project_dir.joinpath('models/longformer-ct')
    model_longformer_ct = SentenceTransformer(model_longformer_ct_path)
    model_bigbird_tsdae_path = project_dir.joinpath('models/bigbird-tsdae')
    model_bigbird_tsdae = SentenceTransformer(model_bigbird_tsdae_path)

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
    # Longformer-SimCSE, Longformer-CT with in-batch negatives, BigBird-TSDAE
    topic_model_longformer_simcse = BERTopic(embedding_model = model_longformer_simcse,
                                             umap_model = umap_model,
                                             vectorizer_model = vectoriser,
                                             representation_model = representation_model)
    topic_model_longformer_ct = BERTopic(embedding_model = model_longformer_ct,
                                         umap_model = umap_model,
                                         vectorizer_model = vectoriser,
                                         representation_model = representation_model)
    topic_model_bigbird_tsdae = BERTopic(embedding_model = model_bigbird_tsdae,
                                         umap_model = umap_model,
                                         vectorizer_model = vectoriser,
                                         representation_model = representation_model)
    fitted_topic_model_longformer_simcse = topic_model_longformer_simcse.fit(docs, train_longformer_simcse_embeddings)
    fitted_topic_model_longformer_ct = topic_model_longformer_ct.fit(docs, train_longformer_ct_embeddings)
    fitted_topic_model_bigbird_tsdae = topic_model_bigbird_tsdae.fit(docs, train_bigbird_tsdae_embeddings)

    # save fitted BERTopic models
    save_path_longformer_simcse = str(project_dir.joinpath(f'models/longformer-simcse-bertopic'))
    save_path_longformer_ct = str(project_dir.joinpath(f'models/longformer-ct-bertopic'))
    save_path_bigbird_tsdae = str(project_dir.joinpath(f'models/bigbird-tsdae-bertopic'))
    fitted_topic_model_longformer_simcse.save(save_path_longformer_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_longformer_ct.save(save_path_longformer_ct, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_bigbird_tsdae.save(save_path_bigbird_tsdae, serialization = 'pytorch', save_ctfidf = True)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()