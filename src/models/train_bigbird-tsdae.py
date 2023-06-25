# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses, datasets
import torch
from torch.utils.data import DataLoader


def main():
    '''
    Trains BigBird-TSDAE model and saves to ../models
    '''
    logger = logging.getLogger(__name__)
    logger.info('training BigBird-TSDAE model from training data ../data/processed/train.pkl')

    # set pytorch seed for reproducibility
    torch.manual_seed(1)

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # model parameters
    data = train['Concatenated']
    model_name = 'google/bigbird-roberta-base'
    batch_size = 1 # effective batch_size is 16 due to gradient accumulation
    num_epochs = 1

    # define pre-trained model
    # gradient checkpointing and gradient accumulation reduces memory usage
    word_embedding_model = models.Transformer(model_name,
                                              model_args = {'gradient_checkpointing': True,
                                                            'gradient_accumulation_steps': 16})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # create denoising dataset and batch the data
    train_sentences = data.to_numpy()
    train_data = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # TSDAE loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path = model_name, tie_encoder_decoder=True)

    # fine-tune pre-trained BigBird model using TSDAE approach
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        use_amp = True #16-bit training; reduces memory usage
    )

    # save model
    save_path = str(project_dir.joinpath(f'models/bigbird-tsdae'))
    model.save(save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()