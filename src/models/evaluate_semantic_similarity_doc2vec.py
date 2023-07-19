# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr


def evaluate_doc2vec_model(model_name, data_with_labels, logger):
    '''
    Evaluate the Spearman Rank Correlation of the given doc2vec model
    The given score is averaged over ten iterations of the testing approach
    This is done to reduce variance of the score as infer_vector is stochastic
    '''
    # load doc2vec models
    model_doc2vec_path = str(project_dir.joinpath(f'models/{model_name}'))
    model_doc2vec = Doc2Vec.load(model_doc2vec_path)

    # due to stochasticity of infer_vector, get the Spearman Rank Correlation 10 times and average it
    spearman_list = []
    for i in range(10):
        # get doc2vec document embeddings for testing documents
        data_with_labels['Text_1_doc2vec'] = data_with_labels['Text_1_Tokens'].apply(model_doc2vec.infer_vector)
        data_with_labels['Text_2_doc2vec'] = data_with_labels['Text_2_Tokens'].apply(model_doc2vec.infer_vector)

        # get cosine scores of doc2vec embeddings
        cosine_scores_doc2vec = 1 - (paired_cosine_distances(data_with_labels['Text_1_doc2vec'].tolist(), 
                                                             data_with_labels['Text_2_doc2vec'].tolist()))
        
        # evaluate Spearman Rank Correlation of cosine similarity on doc2vec with labels
        spearman_cosine_doc2vec, _ = spearmanr(data_with_labels['Similarity'].tolist(), cosine_scores_doc2vec)
        
        spearman_list.append(spearman_cosine_doc2vec)
    
    # average the Spearman Rank Correlations
    spearman_cosine_doc2vec_avg = sum(spearman_list) / len(spearman_list)

    logger.info(f'Spearman Rank Correlation Coefficient for {model_name} (average of 10 iterations): {spearman_cosine_doc2vec_avg}')

    return spearman_cosine_doc2vec_avg


def main():
    '''
    Evaluate the Spearman Rank Correlation Coefficient of Doc2Vec embedding models, using the labelled testing data
    '''
    logger = logging.getLogger(__name__)
    logger.info('evaluating embedding models (Spearman Rank Correlation Coefficient)')

    # load labelled testing data pairs
    test_pairs_path = project_dir.joinpath('data/raw/test_pairs_labelled.txt')
    test_pairs = pd.read_csv(test_pairs_path, sep=';', header=None, names=['ModuleCodes', 'Similarity'])

    # load full testing data
    test_path = project_dir.joinpath('data/interim/test_unlabelled.pkl')
    test = pd.read_pickle(test_path)

    # convert test.ModuleCode from lists to strings
    test['ModuleCode'] = test['ModuleCode'].astype(str)

    # remove first [ and last ] characters from records in test_pairs.ModuleCodes
    test_pairs['ModuleCodes'] = test_pairs['ModuleCodes'].str[1:-1]

    # split ModuleCodes from test_pairs into constituent elements
    test_pairs_separate = test_pairs['ModuleCodes'].str.extractall(r'(\[.*?\])').unstack()
    test_pairs[['ModuleCode_1', 'ModuleCode_2']] = test_pairs_separate

    # get passages of text corresponding to ModuleCodes via lookup
    test_merged_partial = pd.merge(test_pairs,
                                   test[['ModuleCode', 'Concatenated']],
                                   how = 'inner',
                                   left_on = 'ModuleCode_1',
                                   right_on = 'ModuleCode')
    test_merged_partial.drop(columns = 'ModuleCode', inplace = True)
    test_merged_partial.rename(columns = {'Concatenated': 'Text_1'}, inplace = True)

    test_merged = pd.merge(test_merged_partial,
                           test[['ModuleCode', 'Concatenated']],
                           how = 'inner',
                           left_on = 'ModuleCode_2',
                           right_on = 'ModuleCode')
    test_merged.drop(columns = 'ModuleCode', inplace = True)
    test_merged.rename(columns = {'Concatenated': 'Text_2'}, inplace = True)

    # normalise similarities to [0, 1]
    test_merged['Similarity'] = test_merged['Similarity'].div(5)

    # tokenise text passages in testing data
    test_merged['Text_1_Tokens'] = test_merged['Text_1'].apply(gensim.utils.simple_preprocess)
    test_merged['Text_2_Tokens'] = test_merged['Text_2'].apply(gensim.utils.simple_preprocess)

    # find the averaged Spearman rank correlation for each fitted doc2vec model
    doc2vec_scores_list = []
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_1_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_5_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_10_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_25_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_50_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_100_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_200_epochs.model', test_merged, logger))
    doc2vec_scores_list.append(evaluate_doc2vec_model('doc2vec_500_epochs.model', test_merged, logger))

    # save calculated Spearman Rank Correlation for doc2vec, to file
    doc2vec_output_path = project_dir.joinpath('reports/scores/similarity_evaluation_results_doc2vec.txt')
    with open(doc2vec_output_path, 'w') as file:
        file.write('Spearman Rank Correlation Coefficient for doc2vec (average of 10 iterations),\n')
        file.write('for doc2vec models with training epochs 1, 5, 10, 25, 50, 100, 200, 500:\n')
        for score in doc2vec_scores_list:
            file.write(str(score)+'\n')

    logger.info('finished evaluating Doc2Vec embedding models, '
                'output saved to ../reports/scores/similarity_evaluation_results_doc2vec.txt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()