# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import pickle

def main():
    '''
    Use the trained model to generate document embeddings for each model
    Embeddings are generated for both training and testing data
    All fine-tuned models are required for this script to work
    output saved to ../data/processed as train_document_embeddings.pkl
    '''


    # todo: implement embedding generation for testing data


    logger = logging.getLogger(__name__)
    logger.info('generating embeddings for training and testing data in ../data/processed')

    # set pytorch seed for reproducibility (might be unnecessary)
    torch.manual_seed(1)

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # load our fine-tuned Longformer-SimCSE model
    model_longformer_simcse_path = project_dir.joinpath('models/longformer-simcse')
    model_longformer_simcse = SentenceTransformer(model_longformer_simcse_path)
    # load our fine-tuned Longformer-CT with in-batch negatives model
    model_longformer_ct_path = project_dir.joinpath('models/longformer-ct')
    model_longformer_ct = SentenceTransformer(model_longformer_ct_path)
    # load our fine-tuned BigBird-TSDAE model
    model_bigbird_tsdae_path = project_dir.joinpath('models/bigbird-tsdae')
    model_bigbird_tsdae = SentenceTransformer(model_bigbird_tsdae_path)

    # generate Longformer-SimCSE document embeddings
    train_longformer_simcse_embeddings = model_longformer_simcse.encode(sentences = train['Concatenated'].to_numpy(),
                                                                        batch_size = 16,
                                                                        show_progress_bar = True)
    # generate Longformer-CT document embeddings
    train_longformer_ct_embeddings = model_longformer_ct.encode(sentences = train['Concatenated'].to_numpy(),
                                                                batch_size = 16,
                                                                show_progress_bar = True)
    # generate Longformer-CT document embeddings
    train_bigbird_tsdae_embeddings = model_bigbird_tsdae.encode(sentences = train['Concatenated'].to_numpy(),
                                                                batch_size = 16,
                                                                show_progress_bar = True)

    # save document embeddings
    train_embeddings_output = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_output, "wb") as output:
      pickle.dump({'train_longformer_simcse_embeddings': train_longformer_simcse_embeddings,
                   'train_longformer_ct_embeddings': train_longformer_ct_embeddings,
                   'train_bigbird_tsdae_embeddings': train_bigbird_tsdae_embeddings}, 
                  output, 
                  protocol = pickle.HIGHEST_PROTOCOL)

    logger.info('finished generating document embeddings, '
                'output saved to ../data/processed/ as train_document_embeddings.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()