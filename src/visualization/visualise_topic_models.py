# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from bertopic import BERTopic
import numpy as np


def generate_visualisations(name, docs, titles, classes, in_path, out_path, embeddings):
    '''
    Visualise the given fitted BERTopic model
    '''
    # load BERTopic model
    model_path = project_dir.joinpath(in_path)
    model = BERTopic.load(model_path)

    # transform training data by fitted BERTopic model
    topics, probs = model.transform(docs, embeddings)

    # get hierachical representation of topics (for hierarchical visualisation)
    hierarchical_topics = model.hierarchical_topics(docs)

    # get topics per SchoolCode (for stratified visualisation)
    topics_per_class = model.topics_per_class(docs, classes = classes)

    # set output directory for visualisations
    output = project_dir.joinpath(out_path)

    # get number of generated topics (excluding the outlier topic)
    num_topics = len(np.unique(topics)) - 1
    

    # VISUALISATION: 2D representation of topics
    output_topics = output.joinpath(f'visualise_topics_{name}.html')
    model.visualize_topics(title = f'<b>Intertopic Distance Map {name}</b>').write_html(output_topics)

    # VISUALISATION: 2D representation of documents
    output_documents = output.joinpath(f'visualise_documents_{name}.html')
    model.visualize_documents(titles.to_numpy(),
                              embeddings = embeddings,
                              title = f'<b>Documents and Topics {name}</b>').write_html(output_documents)

    # VISUALISATION: hierarchical structure of topics
    output_hierarchy = output.joinpath(f'visualise_hierarchical_topics_{name}.html')
    model.visualize_hierarchy(hierarchical_topics = hierarchical_topics,
                              title = f'<b>Hierarchical Clustering {name}</b>').write_html(output_hierarchy)

    # VISUALISATION: hierarchical structure of documents
    output_hierarchy_documents = output.joinpath(f'visualise_hierarchical_documents_{name}.html')
    model.visualize_hierarchical_documents(titles.to_numpy(), 
                                           hierarchical_topics, 
                                           embeddings = embeddings,
                                           hide_document_hover = False,
                                           title = f'<b>Hierarchical Documents and Topics {name}</b>').write_html(output_hierarchy_documents)

    # VISUALISATION: terms representative of topics, per topic
    output_representative_terms = output.joinpath(f'visualise_representative_terms_{name}.html')
    model.visualize_barchart(top_n_topics = num_topics,
                             title = f'Topic Word Scores {name}').write_html(output_representative_terms)

    # VISUALISATION: topic similarity matrix
    # generate multiple matrices, each with i = 1, 2, ..., 10 similarity clusters
    for i in range(1, 11):
      output_similarity_matrix = output.joinpath(f'visualise_similarity_matrix_{i}_clusters_{name}.html')
      model.visualize_heatmap(top_n_topics = num_topics,
                              n_clusters = i,
                              title = f'<b>Similarity Matrix {i} clusters {name}</b>').write_html(output_similarity_matrix)

    # VISUALISATION: term score decline; the importance of terms, per topic
    output_term_score = output.joinpath(f'visualise_term_score_{name}.html')
    model.visualize_term_rank(title = f'<b>Term score decline per Topic {name}</b>').write_html(output_term_score)

    # VISUALISATION: topics per university school (SchoolCode)
    output_topics_per_school = output.joinpath(f'visualise_topics_per_school_{name}.html')
    model.visualize_topics_per_class(topics_per_class, 
                                     top_n_topics = num_topics,
                                     title = f'<b>Topics per Class {name}</b>').write_html(output_topics_per_school)

def main():
    '''
    Graphical visualisation of the BERTopic topic modelling
    visualisations saved to ../reports/figures/
    '''
    logger = logging.getLogger(__name__)
    logger.info('visualising BERTopic output')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)

    # get documents, ModuleCodes and SchoolCodes
    docs = train['Concatenated']
    titles = train['ModuleCode'].apply(str) # modulecodes now strings instead of lists
    classes = train['SchoolCode']

    # load training data document embeddings
    train_embeddings_path = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_path, "rb") as embeddings_input:
      saved_embeddings = pickle.load(embeddings_input)
      train_longformer_simcse_embeddings = saved_embeddings['train_longformer_simcse_embeddings']
      train_longformer_ct_embeddings = saved_embeddings['train_longformer_ct_embeddings']
      train_bigbird_tsdae_embeddings = saved_embeddings['train_bigbird_tsdae_embeddings']

    # generate visualisations per fitted model
    # Longformer-SimCSE-BERTopic
    generate_visualisations('longformer_simcse_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/longformer-simcse-bertopic',
                            'reports/figures/longformer_simcse_bertopic',
                            train_longformer_simcse_embeddings)
    # Longformer-CT (with in-batch negatives) - BERTopic
    generate_visualisations('longformer_ct_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/longformer-ct-bertopic',
                            'reports/figures/longformer_ct_bertopic',
                            train_longformer_ct_embeddings)
    # BigBird-TSDAE-BERTopic
    generate_visualisations('bigbird_tsdae_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/bigbird-tsdae-bertopic',
                            'reports/figures/bigbird_tsdae_bertopic',
                            train_bigbird_tsdae_embeddings)
                                          
    logger.info('finished visualising BERTopic output, '
                'output saved to ../reports/figures under multiple directories, corresponding to each topic model')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()