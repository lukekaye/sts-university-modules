# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from bertopic import BERTopic
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.lda_model


def generate_visualisations_lda(name, vectorised_corpus, vectoriser, in_path, out_path):
    '''
    Visualise the given fitted LDA model
    Uses Jensen-Shannon Divergence & Principal Coordinate Analysis to represent the embedding-space
    '''
    # load LDA model
    model_path = project_dir.joinpath(in_path)
    with open(model_path, "rb") as model_input:
        saved_lda_models = pickle.load(model_input)
        lda = saved_lda_models['best_lda_model']

    # prepare lda model for visualisation
    visualisation = pyLDAvis.lda_model.prepare(lda, vectorised_corpus, vectoriser)

    # set output directory for visualisation
    output = project_dir.joinpath(out_path)

    # save visualisation to file
    output_pyldavis = output.joinpath(f'visualise_{name}.html')
    pyLDAvis.save_html(visualisation, str(output_pyldavis))


def generate_visualisations(name, docs, titles, classes, in_path, out_path, embeddings):
    '''
    Visualise the given fitted BERTopic model
    Uses Uniform Manifold Approximation Projection (UMAP) to represent the embedding-space
    All visualisations are saved to file
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
    # skip this visualisation for Longformer-BERTopic (broken)
    # possibly broken for Longformer-BERTopic due to the low number of topics found
    if not name == 'longformer_bertopic':
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
    for i in range(1, min(num_topics, 11)):
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
    Graphical visualisation of the topic models
    visualisations saved to ../reports/figures/
    '''
    logger = logging.getLogger(__name__)
    logger.info('visualising topic modelling output')

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)
    train_list = train['Concatenated'].tolist()

    # get documents, ModuleCodes and SchoolCodes
    docs = train['Concatenated']
    titles = train['ModuleCode'].apply(str) # modulecodes now strings instead of lists
    classes = train['SchoolCode']

    # load and fit bag-of-words vectoriser
    vectoriser = CountVectorizer(min_df = 2, stop_words = 'english')
    train_vectorised = vectoriser.fit_transform(train_list)

    # load training data document embeddings
    train_embeddings_path = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_path, "rb") as embeddings_input:
        saved_embeddings = pickle.load(embeddings_input)
        train_longformer_embeddings = saved_embeddings['train_longformer_embeddings']
        train_bigbird_embeddings = saved_embeddings['train_bigbird_embeddings']
        train_distilroberta_embeddings = saved_embeddings['train_distilroberta_embeddings']
        train_all_distilroberta_embeddings = saved_embeddings['train_all_distilroberta_embeddings']
        train_longformer_simcse_embeddings = saved_embeddings['train_longformer_simcse_embeddings']
        train_longformer_ct_embeddings = saved_embeddings['train_longformer_ct_embeddings']
        train_bigbird_simcse_embeddings = saved_embeddings['train_bigbird_simcse_embeddings']
        train_bigbird_ct_embeddings = saved_embeddings['train_bigbird_ct_embeddings']
        train_bigbird_tsdae_embeddings = saved_embeddings['train_bigbird_tsdae_embeddings']
        train_distilroberta_simcse_embeddings = saved_embeddings['train_distilroberta_simcse_embeddings']
        train_distilroberta_ct_embeddings = saved_embeddings['train_distilroberta_ct_embeddings']
        train_distilroberta_tsdae_embeddings = saved_embeddings['train_distilroberta_tsdae_embeddings']
        train_all_distilroberta_simcse_embeddings = saved_embeddings['train_all_distilroberta_simcse_embeddings']
        train_all_distilroberta_ct_embeddings = saved_embeddings['train_all_distilroberta_ct_embeddings']
        train_all_distilroberta_tsdae_embeddings = saved_embeddings['train_all_distilroberta_tsdae_embeddings']


    # generate visualisations per fitted model

    # LDA
    generate_visualisations_lda('LDA_(45_Topics)',
                                train_vectorised,
                                vectoriser,
                                'models/lda_models.pkl',
                                'reports/figures/lda_45_topics')

    # Longformer-BERTopic
    generate_visualisations('longformer_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/longformer-bertopic',
                            'reports/figures/longformer_bertopic',
                            train_longformer_embeddings)
    # BigBird-BERTopic
    generate_visualisations('bigbird_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/bigbird-bertopic',
                            'reports/figures/bigbird_bertopic',
                            train_bigbird_embeddings)
    # DistilRoBERTa-BERTopic
    generate_visualisations('distilroberta_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/distilroberta-bertopic',
                            'reports/figures/distilroberta_bertopic',
                            train_distilroberta_embeddings)
    # all_DistilRoBERTa-BERTopic
    generate_visualisations('all_distilroberta_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/all_distilroberta-bertopic',
                            'reports/figures/all_distilroberta_bertopic',
                            train_all_distilroberta_embeddings)

    # Longformer-SimCSE-BERTopic
    generate_visualisations('longformer_simcse_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/longformer-simcse-bertopic',
                            'reports/figures/longformer_simcse_bertopic',
                            train_longformer_simcse_embeddings)
    # Longformer-CT-BERTopic
    generate_visualisations('longformer_ct_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/longformer-ct-bertopic',
                            'reports/figures/longformer_ct_bertopic',
                            train_longformer_ct_embeddings)

    # BigBird-SimCSE-BERTopic
    generate_visualisations('bigbird_simcse_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/bigbird-simcse-bertopic',
                            'reports/figures/bigbird_simcse_bertopic',
                            train_bigbird_simcse_embeddings)
    # BigBird-CT-BERTopic
    generate_visualisations('bigbird_ct_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/bigbird-ct-bertopic',
                            'reports/figures/bigbird_ct_bertopic',
                            train_bigbird_ct_embeddings)
    # BigBird-TSDAE-BERTopic
    generate_visualisations('bigbird_tsdae_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/bigbird-tsdae-bertopic',
                            'reports/figures/bigbird_tsdae_bertopic',
                            train_bigbird_tsdae_embeddings)

    # DistilRoBERTa-SimCSE-BERTopic
    generate_visualisations('distilroberta_simcse_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/distilroberta-simcse-bertopic',
                            'reports/figures/distilroberta_simcse_bertopic',
                            train_distilroberta_simcse_embeddings)
    # DistilRoBERTa-CT-BERTopic
    generate_visualisations('distilroberta_ct_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/distilroberta-ct-bertopic',
                            'reports/figures/distilroberta_ct_bertopic',
                            train_distilroberta_ct_embeddings)
    # DistilRoBERTa-TSDAE-BERTopic
    generate_visualisations('distilroberta_tsdae_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/distilroberta-tsdae-bertopic',
                            'reports/figures/distilroberta_tsdae_bertopic',
                            train_distilroberta_tsdae_embeddings)

    # all_DistilRoBERTa-SimCSE-BERTopic
    generate_visualisations('all_distilroberta_simcse_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/all_distilroberta-simcse-bertopic',
                            'reports/figures/all_distilroberta_simcse_bertopic',
                            train_all_distilroberta_simcse_embeddings)
    # all_DistilRoBERTa-CT-BERTopic
    generate_visualisations('all_distilroberta_ct_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/all_distilroberta-ct-bertopic',
                            'reports/figures/all_distilroberta_ct_bertopic',
                            train_all_distilroberta_ct_embeddings)
    # all_DistilRoBERTa-TSDAE-BERTopic
    generate_visualisations('all_distilroberta_tsdae_bertopic',
                            docs,
                            titles,
                            classes,
                            'models/all_distilroberta-tsdae-bertopic',
                            'reports/figures/all_distilroberta_tsdae_bertopic',
                            train_all_distilroberta_tsdae_embeddings)

                                          
    logger.info('finished visualising topic modelling output, '
                'output saved to ../reports/figures under multiple directories, corresponding to each topic model')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()