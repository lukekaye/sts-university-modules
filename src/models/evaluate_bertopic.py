# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from bertopic import BERTopic
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from nltk.tokenize import RegexpTokenizer


def evaluate_topic_model(model_name, documents, tokenised_corpus, in_path, embeddings):
    '''
    Evaluate the given fitted BERTopic model, specifically its Topic Diversity and Topic Coherence
    Topic Coherence is given by Normalised Pointwise Mutual Information (NPMI)
    '''
    # load BERTopic model
    model_path = project_dir.joinpath(in_path)
    model = BERTopic.load(model_path)

    # transform training data by fitted SimCSE BERTopic model
    topics, probs = model.transform(documents, embeddings)

    # get top-10 most significant words per topic
    significant_words = model.get_topic_info()['Representation']

    # drop outlier topic (topic -1, first row)
    significant_words = significant_words.iloc[1:]

    # convert significant_words to list of lists and make the value of a dictionary
    significant_words = significant_words.tolist()
    significant_words = {'topics': significant_words}

    # find Topic Diversity
    topic_diversity = TopicDiversity(topk = 10)
    topic_diversity_score = round(topic_diversity.score(significant_words), 3)

    # find Topic Coherence
    logging.getLogger('gensim.topic_coherence.text_analysis').setLevel(logging.WARNING)
    topic_coherence = Coherence(texts = tokenised_corpus, topk = 10, measure = 'c_npmi')
    topic_coherence_score = round(topic_coherence.score(significant_words), 3)
    logging.getLogger('gensim.topic_coherence.text_analysis').setLevel(logging.INFO)

    return [model_name, topic_diversity_score, topic_coherence_score]


def main():
    '''
    Evaluate the Topic Diversity and Topic Coherence of all BERTopic models
    '''
    logger = logging.getLogger(__name__)
    logger.info('evaluating BERTopic models (Topic Diversity & Topic Coherence)')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # get the tokenised corpus in list of lists form, only retaining alphanumeric characters
    # only retaining alphanumeric characters should make the NPMI score more accurate
    tokenised_corpus = train['Concatenated'].apply(RegexpTokenizer(r'\w+').tokenize).tolist()

    # load training data document embeddings
    train_embeddings_path = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_path, "rb") as embeddings_input:
        saved_embeddings = pickle.load(embeddings_input)
        train_simcse_embeddings = saved_embeddings['train_simcse_embeddings']
        train_ct_embeddings = saved_embeddings['train_ct_embeddings']

    # create DataFrame to store evaluation metric scores
    topic_metrics = pd.DataFrame(columns = ['Model', 'Topic Diversity', 'Topic Coherence (NPMI)'])
    topic_scores = []

    # evaluate topic models
    # Longformer-SimCSE
    topic_scores.append(evaluate_topic_model('Longformer-SimCSE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/longformer-simcse-bertopic',
                                             train_simcse_embeddings))
    # Longformer-CT
    topic_scores.append(evaluate_topic_model('Longformer-CT-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/longformer-ct-bertopic',
                                             train_ct_embeddings))

    # append topic model scores to DataFrame
    for scores in topic_scores:
        topic_metrics.loc[len(topic_metrics)] = scores

    # save evaluation metric scores to file
    topic_metrics_path = project_dir.joinpath('reports/scores/topic_evaluation_scores.csv')
    topic_metrics.to_csv(topic_metrics_path, sep=';', index=False)

    logger.info('finished evaluating topic models, '
                'output saved to ../reports/scores/ as topic_evaluation_scores.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()