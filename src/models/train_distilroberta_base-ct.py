# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
import torch
from torch.utils.data import DataLoader


def main():
    '''
    Trains CT with in-batch negatives model backed by DistilRoBERTa and saves to ../models
    Using in-batch negatives slightly improves performance of CT
    '''
    logger = logging.getLogger(__name__)
    logger.info('training DistilRoBERTa-CT with in-batch negatives model from training data ../data/processed/train.pkl')

    # set pytorch seed for reproducibility
    torch.manual_seed(1)

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # model parameters
    data = train['Concatenated']
    model_name = 'distilroberta-base'
    batch_size = 16
    num_epochs = 1

    # define pre-trained model
    # gradient checkpointing reduces memory usage
    word_embedding_model = models.Transformer(model_name,
                                              model_args = {'gradient_checkpointing': True})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # batch the data
    train_data = [InputExample(texts=[s, s]) for s in data]
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # CT with in-batch negatives loss
    train_loss = losses.ContrastiveTensionLossInBatchNegatives(model)

    # fit CT with in-batch negatives model to training data
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        use_amp = True #16-bit training; reduces memory usage
    )

    # save fitted model
    save_path = str(project_dir.joinpath(f'models/distilroberta-ct'))
    model.save(save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()