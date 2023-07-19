# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, models
import torch
import pickle

def main():
    '''
    Use the pre-trained and fine-tuned models to generate document embeddings
    All fine-tuned models are required for this script to work, including FULLDATA_bigbird-ct
    output saved to ../data/processed as train_document_embeddings.pkl and fulldata_bigbird-ct-document-embeddings.pkl
    '''
    logger = logging.getLogger(__name__)
    logger.info('generating embeddings corresponding to fitted models')

    # set pytorch seed for reproducibility (might be unnecessary)
    torch.manual_seed(1)

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # load test_unlabelled.pkl
    test_path = project_dir.joinpath('data/interim/test_unlabelled.pkl')
    test = pd.read_pickle(test_path)

    # concatenate train and test data
    fulldata = pd.concat([train, test])

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

    # load FULLDATA_BigBird-CT
    model_fulldata_bigbird_ct_path = project_dir.joinpath('models/FULLDATA_bigbird-ct')
    model_fulldata_bigbird_ct = SentenceTransformer(model_fulldata_bigbird_ct_path)    

    # get document embeddings

    # generate Longformer document embeddings
    train_longformer_embeddings = model_longformer.encode(sentences = train['Concatenated'].to_numpy(),
                                                          batch_size = 16,
                                                          show_progress_bar = True)
    # generate Bigbird document embeddings
    train_bigbird_embeddings = model_bigbird.encode(sentences = train['Concatenated'].to_numpy(),
                                                    batch_size = 16,
                                                    show_progress_bar = True)
    # generate DistilRoBERTa document embeddings
    train_distilroberta_embeddings = model_distilroberta.encode(sentences = train['Concatenated'].to_numpy(),
                                                                batch_size = 16,
                                                                show_progress_bar = True)         
    # generate all_DistilRoBERTa document embeddings
    train_all_distilroberta_embeddings = model_all_distilroberta.encode(sentences = train['Concatenated'].to_numpy(),
                                                                        batch_size = 16,
                                                                        show_progress_bar = True) 

    # generate Longformer-SimCSE document embeddings
    train_longformer_simcse_embeddings = model_longformer_simcse.encode(sentences = train['Concatenated'].to_numpy(),
                                                                        batch_size = 16,
                                                                        show_progress_bar = True)
    # generate Longformer-CT document embeddings
    train_longformer_ct_embeddings = model_longformer_ct.encode(sentences = train['Concatenated'].to_numpy(),
                                                                batch_size = 16,
                                                                show_progress_bar = True)

    # generate BigBird-SimCSE document embeddings
    train_bigbird_simcse_embeddings = model_bigbird_simcse.encode(sentences = train['Concatenated'].to_numpy(),
                                                                  batch_size = 16,
                                                                  show_progress_bar = True)
    # generate BigBird-CT document embeddings
    train_bigbird_ct_embeddings = model_bigbird_ct.encode(sentences = train['Concatenated'].to_numpy(),
                                                          batch_size = 16,
                                                          show_progress_bar = True)
    # generate Bigbird-TSDAE document embeddings
    train_bigbird_tsdae_embeddings = model_bigbird_tsdae.encode(sentences = train['Concatenated'].to_numpy(),
                                                                batch_size = 16,
                                                                show_progress_bar = True)

    # generate DistilRoBERTa-SimCSE document embeddings
    train_distilroberta_simcse_embeddings = model_distilroberta_simcse.encode(sentences = train['Concatenated'].to_numpy(),
                                                                              batch_size = 16,
                                                                              show_progress_bar = True)
    # generate DistilRoBERTa-CT document embeddings
    train_distilroberta_ct_embeddings = model_distilroberta_ct.encode(sentences = train['Concatenated'].to_numpy(),
                                                                      batch_size = 16,
                                                                      show_progress_bar = True)
    # generate DistilRoBERTa-TSDAE document embeddings
    train_distilroberta_tsdae_embeddings = model_distilroberta_tsdae.encode(sentences = train['Concatenated'].to_numpy(),
                                                                            batch_size = 16,
                                                                            show_progress_bar = True)

    # generate all_DistilRoBERTa-SimCSE document embeddings
    train_all_distilroberta_simcse_embeddings = model_all_distilroberta_simcse.encode(sentences = train['Concatenated'].to_numpy(),
                                                                                      batch_size = 16,
                                                                                      show_progress_bar = True)
    # generate all_DistilRoBERTa-CT document embeddings
    train_all_distilroberta_ct_embeddings = model_all_distilroberta_ct.encode(sentences = train['Concatenated'].to_numpy(),
                                                                              batch_size = 16,
                                                                              show_progress_bar = True)
    # generate all_DistilRoBERTa-TSDAE document embeddings
    train_all_distilroberta_tsdae_embeddings = model_all_distilroberta_tsdae.encode(sentences = train['Concatenated'].to_numpy(),
                                                                                    batch_size = 16,
                                                                                    show_progress_bar = True)

    # generate FULLDATA_BigBird-CT document embeddings
    fulldata_bigbird_ct_embeddings = model_fulldata_bigbird_ct.encode(sentences = fulldata['Concatenated'].to_numpy(),
                                                                      batch_size = 16,
                                                                      show_progress_bar = True)

    # save document embeddings
    train_embeddings_output = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_output, "wb") as output:
        pickle.dump({'train_longformer_embeddings': train_longformer_embeddings,
                    'train_bigbird_embeddings': train_bigbird_embeddings,
                    'train_distilroberta_embeddings': train_distilroberta_embeddings,
                    'train_all_distilroberta_embeddings': train_all_distilroberta_embeddings,
                    'train_longformer_simcse_embeddings': train_longformer_simcse_embeddings,
                    'train_longformer_ct_embeddings': train_longformer_ct_embeddings,
                    'train_bigbird_simcse_embeddings': train_bigbird_simcse_embeddings,
                    'train_bigbird_ct_embeddings': train_bigbird_ct_embeddings,
                    'train_bigbird_tsdae_embeddings': train_bigbird_tsdae_embeddings,
                    'train_distilroberta_simcse_embeddings': train_distilroberta_simcse_embeddings,
                    'train_distilroberta_ct_embeddings': train_distilroberta_ct_embeddings,
                    'train_distilroberta_tsdae_embeddings': train_distilroberta_tsdae_embeddings,
                    'train_all_distilroberta_simcse_embeddings': train_all_distilroberta_simcse_embeddings,
                    'train_all_distilroberta_ct_embeddings': train_all_distilroberta_ct_embeddings,
                    'train_all_distilroberta_tsdae_embeddings': train_all_distilroberta_tsdae_embeddings}, 
                    output, 
                    protocol = pickle.HIGHEST_PROTOCOL)

    fulldata_bigbird_ct_embeddings_output = project_dir.joinpath('data/processed/fulldata_bigbird_ct_document_embeddings.pkl')
    with open(fulldata_bigbird_ct_embeddings_output, "wb") as output:
        pickle.dump({'fulldata_bigbird_ct_embeddings': fulldata_bigbird_ct_embeddings}, 
                    output, 
                    protocol = pickle.HIGHEST_PROTOCOL)

    logger.info('finished generating document embeddings, '
                'output saved to ../data/processed/ as train_document_embeddings.pkl and fulldata_bigbird-ct-document-embeddings.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()