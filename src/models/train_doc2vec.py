# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import gensim



def main():
    '''
    Trains doc2vec model and saves to ../models
    '''
    logger = logging.getLogger(__name__)
    logger.info('training doc2vec model from training data ../data/processed/train.pkl')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # tokenise training data
    train['Tokenised'] = train['Concatenated'].apply(gensim.utils.simple_preprocess)

    # tag training data with indices of records
    train_tagged = []
    for index, record in enumerate(train['Tokenised']):
        train_tagged.append(gensim.models.doc2vec.TaggedDocument(record, [index]))

    # doc2vec model (dm = 1 is Distributed Memory approach as used in Seidel et al., 2020, set seed for reproducibility)
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size = 256,
                                                  dm = 1,
                                                  min_count = 2,
                                                  epochs = 40,
                                                  seed = 1,
                                                  workers = 1)

    # build vocabulary and train model
    doc2vec_model.build_vocab(train_tagged)
    doc2vec_model.train(train_tagged, total_examples = doc2vec_model.corpus_count, epochs = doc2vec_model.epochs)

    # save model
    save_path = str(project_dir.joinpath(f'models/doc2vec.model'))
    doc2vec_model.save(save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()