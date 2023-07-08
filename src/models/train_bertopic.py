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
    Use generated document embeddings to train BERTopic models
    All Transformer model embeddings are required for this script to work
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
        train_longformer_embeddings = saved_embeddings['train_longformer_embeddings']
        train_bigbird_embeddings = saved_embeddings['train_bigbird_embeddings']
        train_distilroberta_embeddings = saved_embeddings['train_distilroberta_embeddings']
        train_all_distilroberta_embeddings = saved_embeddings['train_all_distilroberta_embeddings']
        train_longformer_simcse_embeddings = saved_embeddings['train_longformer_simcse_embeddings']
        train_longformer_ct_embeddings = saved_embeddings['train_longformer_ct_embeddings']
        train_bigbird_simcse_embeddings = saved_embeddings['train_bigbird_simcse_embeddings']
        train_bigbird_ct_embeddings = saved_embeddings['train_bigbird_ct_embeddings']
        train_bigbird_tsdae_embeddings = saved_embeddings['train_bigbird_tsdae_embeddings']
        train_distilroberta_simcse_embeddings = saved_embeddings['train_distilroberta_simcse_embeddings']
        train_distilroberta_ct_embeddings = saved_embeddings['train_distilroberta_ct_embeddings']
        train_distilroberta_tsdae_embeddings = saved_embeddings['train_distilroberta_tsdae_embeddings']
        train_all_distilroberta_simcse_embeddings = saved_embeddings['train_all_distilroberta_simcse_embeddings']
        train_all_distilroberta_ct_embeddings = saved_embeddings['train_all_distilroberta_ct_embeddings']
        train_all_distilroberta_tsdae_embeddings = saved_embeddings['train_all_distilroberta_tsdae_embeddings']

    # get documents
    docs = train['Concatenated']


    # base models

    # load longformer
    longformer_name = 'allenai/longformer-base-4096'
    word_embedding_model = models.Transformer(longformer_name,
                                              model_args = {'gradient_checkpointing': True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model_longformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # load bigbird
    bigbird_name = 'google/bigbird-roberta-base'
    word_embedding_model = models.Transformer(bigbird_name,
                                              model_args = {'gradient_checkpointing': True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model_bigbird = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # load distilroberta-base
    distilroberta_name = 'distilroberta-base'
    word_embedding_model = models.Transformer(distilroberta_name,
                                              model_args = {'gradient_checkpointing': True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model_distilroberta = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # load all-distilroberta-v1 (externally fine-tuned version)
    all_distilroberta_name = 'sentence-transformers/all-distilroberta-v1'
    word_embedding_model = models.Transformer(all_distilroberta_name,
                                              model_args = {'gradient_checkpointing': True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model_all_distilroberta = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # fine-tuned models

    # load our fine-tuned Longformer-SimCSE model
    model_longformer_simcse_path = project_dir.joinpath('models/longformer-simcse')
    model_longformer_simcse = SentenceTransformer(model_longformer_simcse_path)
    # load our fine-tuned Longformer-CT with in-batch negatives model
    model_longformer_ct_path = project_dir.joinpath('models/longformer-ct')
    model_longformer_ct = SentenceTransformer(model_longformer_ct_path)

    # load our fine-tuned BigBird-SimCSE model
    model_bigbird_simcse_path = project_dir.joinpath('models/bigbird-simcse')
    model_bigbird_simcse = SentenceTransformer(model_bigbird_simcse_path)
    # load our fine-tuned BigBird-CT with in-batch negatives model
    model_bigbird_ct_path = project_dir.joinpath('models/bigbird-ct')
    model_bigbird_ct = SentenceTransformer(model_bigbird_ct_path)
    # load our fine-tuned BigBird-TSDAE model
    model_bigbird_tsdae_path = project_dir.joinpath('models/bigbird-tsdae')
    model_bigbird_tsdae = SentenceTransformer(model_bigbird_tsdae_path)

    # load our fine-tuned DistilRoBERTa-SimCSE model
    model_distilroberta_simcse_path = project_dir.joinpath('models/distilroberta-simcse')
    model_distilroberta_simcse = SentenceTransformer(model_distilroberta_simcse_path)
    # load our fine-tuned DistilRoBERTa-CT with in-batch negatives model
    model_distilroberta_ct_path = project_dir.joinpath('models/distilroberta-ct')
    model_distilroberta_ct = SentenceTransformer(model_distilroberta_ct_path)
    # load our fine-tuned DistilRoBERTa-TSDAE model
    model_distilroberta_tsdae_path = project_dir.joinpath('models/distilroberta-tsdae')
    model_distilroberta_tsdae = SentenceTransformer(model_distilroberta_tsdae_path)

    # load our further fine-tuned all_DistilRoBERTa-SimCSE model
    model_all_distilroberta_simcse_path = project_dir.joinpath('models/all_distilroberta-simcse')
    model_all_distilroberta_simcse = SentenceTransformer(model_all_distilroberta_simcse_path)
    # load our further fine-tuned all_DistilRoBERTa-CT with in-batch negatives model
    model_all_distilroberta_ct_path = project_dir.joinpath('models/all_distilroberta-ct')
    model_all_distilroberta_ct = SentenceTransformer(model_all_distilroberta_ct_path)
    # load our further fine-tuned all_DistilRoBERTa-TSDAE model
    model_all_distilroberta_tsdae_path = project_dir.joinpath('models/all_distilroberta-tsdae')
    model_all_distilroberta_tsdae = SentenceTransformer(model_all_distilroberta_tsdae_path)


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

    # longformer
    topic_model_longformer = BERTopic(embedding_model = model_longformer,
                                      umap_model = umap_model,
                                      vectorizer_model = vectoriser,
                                      representation_model = representation_model)
    # bigbird
    topic_model_bigbird = BERTopic(embedding_model = model_bigbird,
                                   umap_model = umap_model,
                                   vectorizer_model = vectoriser,
                                   representation_model = representation_model)
    # distilroberta
    topic_model_distilroberta = BERTopic(embedding_model = model_distilroberta,
                                         umap_model = umap_model,
                                         vectorizer_model = vectoriser,
                                         representation_model = representation_model)
    # all_distilroberta
    topic_model_all_distilroberta = BERTopic(embedding_model = model_all_distilroberta,
                                             umap_model = umap_model,
                                             vectorizer_model = vectoriser,
                                             representation_model = representation_model)

    # longformer-simcse
    topic_model_longformer_simcse = BERTopic(embedding_model = model_longformer_simcse,
                                             umap_model = umap_model,
                                             vectorizer_model = vectoriser,
                                             representation_model = representation_model)
    # longformer-ct
    topic_model_longformer_ct = BERTopic(embedding_model = model_longformer_ct,
                                         umap_model = umap_model,
                                         vectorizer_model = vectoriser,
                                         representation_model = representation_model)

    # bigbird-simcse
    topic_model_bigbird_simcse = BERTopic(embedding_model = model_bigbird_simcse,
                                          umap_model = umap_model,
                                          vectorizer_model = vectoriser,
                                          representation_model = representation_model)
    # bigbird-ct
    topic_model_bigbird_ct = BERTopic(embedding_model = model_bigbird_ct,
                                      umap_model = umap_model,
                                      vectorizer_model = vectoriser,
                                      representation_model = representation_model)
    # bigbird-tsdae
    topic_model_bigbird_tsdae = BERTopic(embedding_model = model_bigbird_tsdae,
                                         umap_model = umap_model,
                                         vectorizer_model = vectoriser,
                                         representation_model = representation_model)

    # distilroberta-simcse
    topic_model_distilroberta_simcse = BERTopic(embedding_model = model_distilroberta_simcse,
                                                umap_model = umap_model,
                                                vectorizer_model = vectoriser,
                                                representation_model = representation_model)
    # distilroberta-ct
    topic_model_distilroberta_ct = BERTopic(embedding_model = model_distilroberta_ct,
                                            umap_model = umap_model,
                                            vectorizer_model = vectoriser,
                                            representation_model = representation_model)
    # distilroberta-tsdae
    topic_model_distilroberta_tsdae = BERTopic(embedding_model = model_distilroberta_tsdae,
                                               umap_model = umap_model,
                                               vectorizer_model = vectoriser,
                                               representation_model = representation_model)

    # all_distilroberta-simcse
    topic_model_all_distilroberta_simcse = BERTopic(embedding_model = model_all_distilroberta_simcse,
                                                    umap_model = umap_model,
                                                    vectorizer_model = vectoriser,
                                                    representation_model = representation_model)
    # all_distilroberta-ct
    topic_model_all_distilroberta_ct = BERTopic(embedding_model = model_all_distilroberta_ct,
                                                umap_model = umap_model,
                                                vectorizer_model = vectoriser,
                                                representation_model = representation_model)
    # all_distilroberta-tsdae
    topic_model_all_distilroberta_tsdae = BERTopic(embedding_model = model_all_distilroberta_tsdae,
                                                   umap_model = umap_model,
                                                   vectorizer_model = vectoriser,
                                                   representation_model = representation_model)


    fitted_topic_model_longformer = topic_model_longformer.fit(docs, train_longformer_embeddings)
    fitted_topic_model_bigbird = topic_model_bigbird.fit(docs, train_bigbird_embeddings)
    fitted_topic_model_distilroberta = topic_model_distilroberta.fit(docs, train_distilroberta_embeddings)
    fitted_topic_model_all_distilroberta = topic_model_all_distilroberta.fit(docs, train_all_distilroberta_embeddings)
    fitted_topic_model_longformer_simcse = topic_model_longformer_simcse.fit(docs, train_longformer_simcse_embeddings)
    fitted_topic_model_longformer_ct = topic_model_longformer_ct.fit(docs, train_longformer_ct_embeddings)
    fitted_topic_model_bigbird_simcse = topic_model_bigbird_simcse.fit(docs, train_bigbird_simcse_embeddings)
    fitted_topic_model_bigbird_ct = topic_model_bigbird_ct.fit(docs, train_bigbird_ct_embeddings)
    fitted_topic_model_bigbird_tsdae = topic_model_bigbird_tsdae.fit(docs, train_bigbird_tsdae_embeddings)
    fitted_topic_model_distilroberta_simcse = topic_model_distilroberta_simcse.fit(docs, train_distilroberta_simcse_embeddings)
    fitted_topic_model_distilroberta_ct = topic_model_distilroberta_ct.fit(docs, train_distilroberta_ct_embeddings)
    fitted_topic_model_distilroberta_tsdae = topic_model_distilroberta_tsdae.fit(docs, train_distilroberta_tsdae_embeddings)
    fitted_topic_model_all_distilroberta_simcse = topic_model_all_distilroberta_simcse.fit(docs, train_all_distilroberta_simcse_embeddings)
    fitted_topic_model_all_distilroberta_ct = topic_model_all_distilroberta_ct.fit(docs, train_all_distilroberta_ct_embeddings)
    fitted_topic_model_all_distilroberta_tsdae = topic_model_all_distilroberta_tsdae.fit(docs, train_all_distilroberta_tsdae_embeddings)

    # save fitted BERTopic models
    save_path_longformer = str(project_dir.joinpath(f'models/longformer-bertopic'))
    save_path_bigbird = str(project_dir.joinpath(f'models/bigbird-bertopic'))
    save_path_distilroberta = str(project_dir.joinpath(f'models/distilroberta-bertopic'))
    save_path_all_distilroberta = str(project_dir.joinpath(f'models/all_distilroberta-bertopic'))
    save_path_longformer_simcse = str(project_dir.joinpath(f'models/longformer-simcse-bertopic'))
    save_path_longformer_ct = str(project_dir.joinpath(f'models/longformer-ct-bertopic'))
    save_path_bigbird_simcse = str(project_dir.joinpath(f'models/bigbird-simcse-bertopic'))
    save_path_bigbird_ct = str(project_dir.joinpath(f'models/bigbird-ct-bertopic'))
    save_path_bigbird_tsdae = str(project_dir.joinpath(f'models/bigbird-tsdae-bertopic'))
    save_path_distilroberta_simcse = str(project_dir.joinpath(f'models/distilroberta-simcse-bertopic'))
    save_path_distilroberta_ct = str(project_dir.joinpath(f'models/distilroberta-ct-bertopic'))
    save_path_distilroberta_tsdae = str(project_dir.joinpath(f'models/distilroberta-tsdae-bertopic'))
    save_path_all_distilroberta_simcse = str(project_dir.joinpath(f'models/all_distilroberta-simcse-bertopic'))
    save_path_all_distilroberta_ct = str(project_dir.joinpath(f'models/all_distilroberta-ct-bertopic'))
    save_path_all_distilroberta_tsdae = str(project_dir.joinpath(f'models/all_distilroberta-tsdae-bertopic'))

    fitted_topic_model_longformer.save(save_path_longformer, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_bigbird.save(save_path_bigbird, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_distilroberta.save(save_path_distilroberta, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_all_distilroberta.save(save_path_all_distilroberta, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_longformer_simcse.save(save_path_longformer_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_longformer_ct.save(save_path_longformer_ct, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_bigbird_simcse.save(save_path_bigbird_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_bigbird_ct.save(save_path_bigbird_ct, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_bigbird_tsdae.save(save_path_bigbird_tsdae, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_distilroberta_simcse.save(save_path_distilroberta_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_distilroberta_ct.save(save_path_distilroberta_ct, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_distilroberta_tsdae.save(save_path_distilroberta_tsdae, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_all_distilroberta_simcse.save(save_path_all_distilroberta_simcse, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_all_distilroberta_ct.save(save_path_all_distilroberta_ct, serialization = 'pytorch', save_ctfidf = True)
    fitted_topic_model_all_distilroberta_tsdae.save(save_path_all_distilroberta_tsdae, serialization = 'pytorch', save_ctfidf = True)

    logger.info('finished fitting topic models, '
                'output saved to ../models/ under various names and formats')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()