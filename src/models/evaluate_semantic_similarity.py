# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models

def main():
    '''
    Evaluate the Spearman Rank Correlation Coefficient of all embedding models, using the labelled testing data
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

    # define semantic textual similarity evaluator
    evaluator = EmbeddingSimilarityEvaluator(test_merged['Text_1'].tolist(),
                                             test_merged['Text_2'].tolist(),
                                             test_merged['Similarity'].tolist(),
                                             batch_size = 16,
                                             show_progress_bar = True)


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


    # evaluate embedding models by Spearman Rank Correlation (and others)
    output_path = project_dir.joinpath('reports/scores')
    
    evaluator(model_longformer, output_path = output_path)
    evaluator(model_bigbird, output_path = output_path)
    evaluator(model_distilroberta, output_path = output_path)
    evaluator(model_all_distilroberta, output_path = output_path)

    evaluator(model_longformer_simcse, output_path = output_path)
    evaluator(model_longformer_ct, output_path = output_path)

    evaluator(model_bigbird_simcse, output_path = output_path)
    evaluator(model_bigbird_ct, output_path = output_path)
    evaluator(model_bigbird_tsdae, output_path = output_path)

    evaluator(model_distilroberta_simcse, output_path = output_path)
    evaluator(model_distilroberta_ct, output_path = output_path)
    evaluator(model_distilroberta_tsdae, output_path = output_path)

    evaluator(model_all_distilroberta_simcse, output_path = output_path)
    evaluator(model_all_distilroberta_ct, output_path = output_path)
    evaluator(model_all_distilroberta_tsdae, output_path = output_path)

    logger.info('finished evaluating embedding models, '
                'output saved to ../reports/scores/similarity_evaluation_results.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()