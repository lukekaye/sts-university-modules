# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    '''
    Creates training and testing partitions and builds features for modelling from cleaned data text.pkl in ../interim
    Output saved as train.pkl in ../processed and test_unlabelled.csv in ../interim.
    '''
    logger = logging.getLogger(__name__)
    logger.info('building features from ../data/interim/text.pkl')

    # load text.pkl
    text_path = project_dir.joinpath('data/interim/text.pkl')
    text = pd.read_pickle(text_path)

    # create concatenated feature representation; missing values are converted to a blank string for the concatenation
    text['Concatenated'] = text.Aims.fillna('') + ' ' + \
                           text.OutlineOfSyllabus.fillna('') + ' ' + \
                           text.IntendedKnowledgeOutcomes.fillna('') + ' ' + \
                           text.IntendedSkillOutcomes.fillna('')

    # partition data into training and testing sets
    train, test = train_test_split(text, test_size=0.2, random_state=1, shuffle=True)

    test_pairs = []
    # create testing pairs: ModuleCode indices 0 and 1 are paired, 2 and 3 are paired, etc.
    for index in range(0, len(test), 2):
        pair = [test.ModuleCode.iloc[index], test.ModuleCode.iloc[index + 1]]
        test_pairs.append(pair)
    test_pairs_series = pd.Series(test_pairs)

    # save train and test tables to ../data/processed and ../data/interim as train.pkl and test_unlabelled.csv, respectively
    train_output = project_dir.joinpath('data/processed/train.pkl')
    test_output = project_dir.joinpath('data/interim/test_unlabelled.csv')
    train.to_pickle(train_output)
    test_pairs_series.to_csv(test_output, sep = ';', index = False)

    logger.info('finished constructing feature partitions from ../data/interim/text.pkl, '
                'output saved to ../data/processed/ as train.pkl and to ../data/interim as test_unlabelled.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()