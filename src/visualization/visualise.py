# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from bertopic import BERTopic
import numpy as np


def main():
    '''
    Graphical visualisation of the BERTopic topic modelling
    '''
    logger = logging.getLogger(__name__)
    logger.info('visualising BERTopic output')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # get documents and modulecodes
    docs = train['Concatenated']
    titles = train['ModuleCode'].apply(str) # modulecodes now strings instead of lists
    classes = train['SchoolCode']

    # load training data SimCSE document embeddings
    train_simcse_embeddings_path = project_dir.joinpath('data/processed/train_simcse_embeddings.pkl')
    with open(train_simcse_embeddings_path, "rb") as embeddings_input:
      saved_embeddings = pickle.load(embeddings_input)
      train_simcse_embeddings = saved_embeddings['train_simcse_embeddings']

    # load BERTopic model
    simcse_bertopic_path = project_dir.joinpath('models/longformer-simcse-bertopic')
    simcse_bertopic = BERTopic.load(simcse_bertopic_path)

    # transform training data by fitted BERTopic model
    topics, probs = simcse_bertopic.transform(docs, train_simcse_embeddings)

    # get hierachical representation of topics (for hierarchical visualisation)
    hierarchical_topics = simcse_bertopic.hierarchical_topics(docs)

    # get topics per SchoolCode (for stratified visualisation)
    topics_per_class = simcse_bertopic.topics_per_class(docs, classes = classes)

    # set output directory for visualisations
    output = project_dir.joinpath('reports/figures/bertopic_simcse')

    # get number of generated topics (excluding the outlier topic)
    num_topics = len(np.unique(topics)) - 1
    

    # VISUALISATION: 2D representation of topics
    output_topics = output.joinpath('visualise_topics_simcse.html')
    simcse_bertopic.visualize_topics().write_html(output_topics)

    # VISUALISATION: 2D representation of documents
    output_documents = output.joinpath('visualise_documents_simcse.html')
    simcse_bertopic.visualize_documents(titles.to_numpy(), embeddings = train_simcse_embeddings).write_html(output_documents)

    # VISUALISATION: hierarchical structure of topics
    output_hierarchy = output.joinpath('visualise_hierarchical_topics_simcse.html')
    simcse_bertopic.visualize_hierarchy(hierarchical_topics = hierarchical_topics).write_html(output_hierarchy)

    # VISUALISATION: hierarchical structure of documents
    output_hierarchy_documents = output.joinpath('visualise_hierarchical_documents_simcse.html')
    simcse_bertopic.visualize_hierarchical_documents(titles.to_numpy(), 
                                                     hierarchical_topics, 
                                                     embeddings = train_simcse_embeddings,
                                                     hide_document_hover = False).write_html(output_hierarchy_documents)

    # VISUALISATION: terms representative of topics, per topic
    output_representative_terms = output.joinpath('visualise_representative_terms_simcse.html')
    simcse_bertopic.visualize_barchart(top_n_topics = num_topics).write_html(output_representative_terms)

    # VISUALISATION: topic similarity matrix
    # generate multiple matrices, each with i = 1, 2, ..., 10 similarity clusters
    for i in range(1, 11):
      output_similarity_matrix = output.joinpath(f'visualise_similarity_matrix_{i}_clusters_simcse.html')
      simcse_bertopic.visualize_heatmap(top_n_topics = num_topics, n_clusters = i).write_html(output_similarity_matrix)

    # VISUALISATION: term score decline; the importance of terms, per topic
    output_term_score = output.joinpath('visualise_term_score_simcse.html')
    simcse_bertopic.visualize_term_rank().write_html(output_term_score)

    # VISUALISATION: topics per university school (SchoolCode)
    output_topics_per_school = output.joinpath('visualise_topics_per_school_simcse.html')
    simcse_bertopic.visualize_topics_per_class(topics_per_class, 
                                               top_n_topics = num_topics).write_html(output_topics_per_school)
                                          
    logger.info('finished visualising BERTopic output, '
        'output saved to ../reports/figures/bertopic_simcse as 17 .html files')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()