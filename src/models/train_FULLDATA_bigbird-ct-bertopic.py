# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, models
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


def main():
    '''
    Use BigBird-CT full dataset document embeddings to train BigBird-CT-BERTopic
    FULLDATA_BigBird-CT document embeddings are required for this script
    output saved to ../models
    '''
    logger = logging.getLogger(__name__)
    logger.info('generating BigBird-CT-BERTopic topic model')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # load test_unlabelled.pkl
    test_path = project_dir.joinpath('data/interim/test_unlabelled.pkl')
    test = pd.read_pickle(test_path)

    # concatenate train and test data
    fulldata = pd.concat([train, test])

    # load full dataset BigBird-CT document embeddings
    fulldata_bigbird_ct_embeddings_output = project_dir.joinpath('data/processed/fulldata_bigbird_ct_document_embeddings.pkl')
    with open(fulldata_bigbird_ct_embeddings_output, "rb") as embeddings_input:
        saved_embeddings = pickle.load(embeddings_input)
        fulldata_bigbird_ct_embeddings = saved_embeddings['fulldata_bigbird_ct_embeddings']

    # get documents
    docs = fulldata['Concatenated']

    # load FULLDATA_BigBird-CT
    model_fulldata_bigbird_ct_path = project_dir.joinpath('models/FULLDATA_bigbird-ct')
    model_fulldata_bigbird_ct = SentenceTransformer(model_fulldata_bigbird_ct_path)    

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
    # FULLDATA_bigbird-ct
    topic_model_fulldata_bigbird_ct = BERTopic(embedding_model = model_fulldata_bigbird_ct,
                                               umap_model = umap_model,
                                               vectorizer_model = vectoriser,
                                               representation_model = representation_model)


    fitted_topic_model_fulldata_bigbird_ct = topic_model_fulldata_bigbird_ct.fit(docs, fulldata_bigbird_ct_embeddings)

    # save fitted BERTopic model
    save_path_fulldata_bigbird_ct = str(project_dir.joinpath(f'models/FULLDATA_bigbird-ct-bertopic'))

    fitted_topic_model_fulldata_bigbird_ct.save(save_path_fulldata_bigbird_ct, serialization = 'pytorch', save_ctfidf = True)

    logger.info('finished fitting topic model, '
                'output saved to ../models/FULLDATA_bigbird-ct-bertopic')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()