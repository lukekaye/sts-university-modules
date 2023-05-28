# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import networkx as nx


def find_similar(corpus_1, corpus_2, corpus_3, corpus_4, module_codes):
    '''
    Uses bag-of-words and cosine similarity
    Group module codes that have cosine similarity >= 0.9 in all given text fields (corpus_x)
    '''
    # drop corresponding entries in corpus_x and module_codes Series where any corpus value is missing
    missing_indices_1 = corpus_1.isna()
    missing_indices_2 = corpus_2.isna()
    missing_indices_3 = corpus_3.isna()
    missing_indices_4 = corpus_4.isna()
    missing_indices_any = missing_indices_1 + missing_indices_2 + missing_indices_3 + missing_indices_4
    module_codes = module_codes[~missing_indices_any].reset_index(drop=True)
    corpus_1 = corpus_1[~missing_indices_any].reset_index(drop=True)
    corpus_2 = corpus_2[~missing_indices_any].reset_index(drop=True)
    corpus_3 = corpus_3[~missing_indices_any].reset_index(drop=True)
    corpus_4 = corpus_4[~missing_indices_any].reset_index(drop=True)
    # transform documents to bag-of-words representations
    vectoriser_1 = CountVectorizer()
    vectoriser_2 = CountVectorizer()
    vectoriser_3 = CountVectorizer()
    vectoriser_4 = CountVectorizer()
    bag_of_words_1 = vectoriser_1.fit_transform(corpus_1)
    bag_of_words_2 = vectoriser_2.fit_transform(corpus_2)
    bag_of_words_3 = vectoriser_3.fit_transform(corpus_3)
    bag_of_words_4 = vectoriser_4.fit_transform(corpus_4)
    # get cosine similarity matrix of documents
    similarity_matrix_1 = cosine_similarity(bag_of_words_1)
    similarity_matrix_2 = cosine_similarity(bag_of_words_2)
    similarity_matrix_3 = cosine_similarity(bag_of_words_3)
    similarity_matrix_4 = cosine_similarity(bag_of_words_4)
    # set diagonal cosine similarities to zero as we are not interested in self-similarities
    np.fill_diagonal(similarity_matrix_1, 0)
    np.fill_diagonal(similarity_matrix_2, 0)
    np.fill_diagonal(similarity_matrix_3, 0)
    np.fill_diagonal(similarity_matrix_4, 0)
    # get indices where similarities >= 0.9 in all corpus
    indices = np.where((similarity_matrix_1 >= 0.9) &
                       (similarity_matrix_2 >= 0.9) &
                       (similarity_matrix_3 >= 0.9) &
                       (similarity_matrix_4 >= 0.9))
    # convert ModuleCode values to singleton lists
    module_codes = module_codes.apply(lambda module_code: [module_code])
    # merge lists of similar ModuleCodes; each merging is given twice, later fixed through connected components
    similar_modules = module_codes.iloc[indices[0]].reset_index(drop=True).add(
        module_codes.iloc[indices[1]].reset_index(drop=True)
    )

    return similar_modules


def clean_text(document):
    '''
    Helper function to clean text within Aims, OutlineOfSyllabus, IntendedKnowledgeOutcomes and IntendedSkillOutcomes
    Cleaning is minimal here due to self-attention used in Transformers
    '''
    if pd.isna(document):
        return document
    document = str(document)
    document = document.lower() # make lower case
    document = document.replace('\n', ' ')  # remove newline characters
    document = document.replace('\r', ' ')  # remove carriage returns
    document = document.replace('\t', ' ')  # remove tab characters
    document = document.replace(r'•', ' ')  # remove bullets
    document = document.replace(r'+', ' ')  # remove pluses
    document = document.replace(r'-', ' ')  # also removes hyphenation between words, which is desirable
    document = re.sub(r'\([^)]?\)', ' ', document) # remove singular characters between brackets, including brackets
    # removes numbered and asterisk bullet points
    document = re.sub(r'(?:^|(?<=\s))\d\.?(?:\d+)?(?=\s)|\*(?=\s)', ' ', document)
    document = re.sub(r'(?<=[ ]).?([\)])', ' ', document) # remove pairings of single chars and right brackets
    document = contractions.fix(document) # replace contractions, e.g. can't -> can not
    document = re.sub(r' +', ' ', document) # remove excess whitespace
    return document


def main():
    '''
    Preprocesses raw modules.csv in ../raw
    Primarily concerned with dropping uninteresting fields, text cleaning and merging duplicate modules
    Output saved as metadata.pkl and text.pkl, to ../interim
    '''
    logger = logging.getLogger(__name__)
    logger.info('preprocessing ../data/raw/modules.csv')

    # this needs to be set else the output .pkl files will save with truncated values in cells
    pd.set_option('display.max_colwidth', None)

    # load modules.csv; despite the data being stored as comma-separated values, the data is separated by semicolons
    modules_path = project_dir.joinpath('data/raw/modules.csv')
    modules_raw = pd.read_csv(modules_path, sep = ';')

    # discard useless fields
    modules = modules_raw.drop(columns = ['ShortTitle',
                                          'StandAloneAvailability',
                                          'IsFtlcApproved',
                                          'DateFtlcApproved',
                                          'IsBosApproved',
                                          'DateBosApproved',
                                          'IsUploadedToSap',
                                          'DateSapUploaded',
                                          'CriticalThinking',
                                          'DataSynthesis',
                                          'ActiveLearning',
                                          'Numeracy',
                                          'Literacy',
                                          'SelfAwarenessAndReflection',
                                          'InnovationAndCreativity',
                                          'Initiative',
                                          'Independence',
                                          'Adaptability',
                                          'ProblemSolving',
                                          'Budgeting',
                                          'Oral',
                                          'ForeignLanguages',
                                          'Interpersonal',
                                          'WrittenOther',
                                          'Collaboration',
                                          'RelationshipBuilding',
                                          'Leadership',
                                          'Negotiation',
                                          'PeerAssessmentReview',
                                          'OccupationalAwareness',
                                          'MarketAwareness',
                                          'GovernanceAwareness',
                                          'FinancialAwareness',
                                          'BusinessPlanning',
                                          'EthicalAwareness',
                                          'SocialCulturalGlobalAwareness',
                                          'LegalAwareness',
                                          'SourceMaterials',
                                          'SynthesiseAndPresentMaterials',
                                          'UseOfComputerApplications',
                                          'GoalSettingAndActionPlanning',
                                          'DecisionMaking',
                                          'TeachingRationaleAndRelationship',
                                          'AssessmentRationaleAndRelationship',
                                          'ExemptFromAssessment',
                                          'ExemptFromAssessmentDate',
                                          'ExemptFromAssessmentComment',
                                          'IsHepatitisAImmunisationOffered',
                                          'IsHepatitisBImmunisationOffered',
                                          'IsTetanusImmunisationOffered',
                                          'IsAllergyScreeningOffered',
                                          'GeneralNotes',
                                          'NonStandardSessionOfOffering_id',
                                          'AcademicYear',
                                          'AcademicYearId',
                                          'Timestamp',
                                          'IsThemedAgeing',
                                          'IsThemedSocialRenewal',
                                          'IsThemedSustainability',
                                          'IsSapUploadDisabled'])

    # remove IsDummy = True records and discard IsDummy field; dummy modules are not useful for STS
    modules = modules.loc[modules['IsDummy'] == False]
    modules = modules.drop(columns = 'IsDummy')

    # standardise missing-type values in PreRequisiteComment and CoRequisiteComment
    requisites_to_replace = ['None', 'none', '-', 'NONE', 'None.', 'no', 'No', 'N/', 'N/A', 'N/A.', 'Non']

    modules.PreRequisiteComment.replace(requisites_to_replace, 'None given', inplace = True)
    modules.CoRequisiteComment.replace(requisites_to_replace, 'None given', inplace = True)
    modules.PreRequisiteComment.fillna('None given', inplace = True)
    modules.CoRequisiteComment.fillna('None given', inplace = True)

    # remove \r \t \n characters from PreRequisiteComment and CoRequisiteComment
    modules.PreRequisiteComment = modules.PreRequisiteComment.str.replace('\t', ' ')
    modules.PreRequisiteComment = modules.PreRequisiteComment.str.replace('\r', '')
    modules.PreRequisiteComment = modules.PreRequisiteComment.str.replace('\n', ' ')
    modules.CoRequisiteComment = modules.CoRequisiteComment.str.replace('\t', ' ')
    modules.CoRequisiteComment = modules.CoRequisiteComment.str.replace('\r', '')
    modules.CoRequisiteComment = modules.CoRequisiteComment.str.replace('\n', ' ')

    # replace short, meaningless natural language values with missing values
    modules.Aims.replace(['Original Summary:'], pd.NA, inplace = True)
    modules.OutlineOfSyllabus.replace(['TBA',
                                       'See above.',
                                       'Not applicable',
                                       'Compulsory module - please see module aims'], pd.NA, inplace=True)
    modules.IntendedKnowledgeOutcomes.replace('Specific to approved title', pd.NA, inplace=True)
    modules.IntendedSkillOutcomes.replace('No specific skill outcomes for this module.', pd.NA, inplace=True)

    # remove records with all natural language values missing; we cannot use them for STS
    modules.dropna(how = 'all',
                   subset = ['Aims',
                             'OutlineOfSyllabus',
                             'IntendedKnowledgeOutcomes',
                             'IntendedSkillOutcomes'],
                   inplace = True)

    # clean natural language using helper function
    modules['Aims_clean'] = modules.Aims.apply(clean_text)
    modules['OutlineOfSyllabus_clean'] = modules.OutlineOfSyllabus.apply(clean_text)
    modules['IntendedKnowledgeOutcomes_clean'] = modules.IntendedKnowledgeOutcomes.apply(clean_text)
    modules['IntendedSkillOutcomes_clean'] = modules.IntendedSkillOutcomes.apply(clean_text)

    # split modules table into metadata and features tables; ModuleCode and Module_Id are (equivalent) primary keys
    metadata = modules[['ModuleCode', 'SapObjectId', 'Title', 'MaxCapacity',
                        'IsNew', 'Semester1Offered', 'Semester1CreditValue', 'Semester2Offered',
                        'Semester2CreditValue', 'Semester3Offered', 'Semester3CreditValue', 'EctsCreditValue',
                        'FheqLevel', 'Mode', 'Delivery', 'IsOffered',
                        'PreRequisiteComment', 'CoRequisiteComment', 'Availability', 'StudyAbroad',
                        'GraduateSkillsFrameworkApplicable', 'SchoolCode', 'MarkingScale', 'Module_Id',
                        'TeachingLocation']]
    text = modules[['ModuleCode', 'Aims_clean', 'OutlineOfSyllabus_clean', 'IntendedKnowledgeOutcomes_clean',
                    'IntendedSkillOutcomes_clean']]

    # rename text fields in features table
    text = text.rename(columns={'Aims_clean': 'Aims',
                                'OutlineOfSyllabus_clean': 'OutlineOfSyllabus',
                                'IntendedKnowledgeOutcomes_clean': 'IntendedKnowledgeOutcomes',
                                'IntendedSkillOutcomes_clean': 'IntendedSkillOutcomes'})

    # henceforth merge records based on bag-of-words cosine similarity, retaining longest text value per field
    # this procedure takes multiple steps

    # group ModuleCodes that have cosine similarity >= 0.9 in all text fields
    cosine_similar = find_similar(text.Aims,
                                  text.OutlineOfSyllabus,
                                  text.IntendedKnowledgeOutcomes,
                                  text.IntendedSkillOutcomes,
                                  text.ModuleCode)

    # merge the groupings of cosine similar ModuleCodes with the previous set of grouped ModuleCodes
    all_grouped = pd.concat([cosine_similar,
                             text.ModuleCode.apply(lambda module_code: [module_code])],
                            ignore_index=True)

    # graph theoretic approach: group ModuleCodes that share common elements by finding connected components
    # build graph
    graph = nx.from_edgelist(itertools.chain.from_iterable(itertools.pairwise(pair) for pair in sorted(all_grouped)))
    graph.add_nodes_from(set.union(*map(set, all_grouped)))

    # find connected components and sort
    similar_modules = list(nx.connected_components(graph))
    similar_modules = [sorted(list(module_codes)) for module_codes in similar_modules]
    similar_modules = pd.Series(data=sorted(similar_modules))

    # join similar_modules to text fields, retaining longest value per text field per record group
    # create empty dataframe for grouped ModuleCodes with corresponding text
    text_merged = pd.DataFrame(columns = ['ModuleCode', 'Aims', 'OutlineOfSyllabus',
                                          'IntendedKnowledgeOutcomes', 'IntendedSkillOutcomes'])
    # iterate through every group in similar_modules
    for module_group in similar_modules:
        # lists for each text field
        aim_set = []
        oos_set = []
        iko_set = []
        iso_set = []
        # iterate through each ModuleCode in the current group
        for module_code in module_group:
            # get text corresponding to ModuleCode
            text_values = text[text['ModuleCode'] == module_code][['Aims',
                                                                   'OutlineOfSyllabus',
                                                                   'IntendedKnowledgeOutcomes',
                                                                   'IntendedSkillOutcomes']]
            # append text to lists, depending on text field
            aim_set.append(text_values.Aims.to_string(index=False))
            oos_set.append(text_values.OutlineOfSyllabus.to_string(index=False))
            iko_set.append(text_values.IntendedKnowledgeOutcomes.to_string(index=False))
            iso_set.append(text_values.IntendedSkillOutcomes.to_string(index=False))
        # get record to append to text_merged; retains longest string per field to minimise semantic info lost
        record = [module_group,
                  max(aim_set, key = len),
                  max(oos_set, key = len),
                  max(iko_set, key = len),
                  max(iso_set, key = len)]
        # append the record to features_merged
        text_merged.loc[len(text_merged)] = record

    # replace 'NaN' strings with missing values; these have erroneously appeared during this preprocessing
    text_merged = text_merged.replace(['NaN', '<NA>'], pd.NA)

    # save metadata and features_merged tables to ../data/interim as metadata.pkl and features.pkl
    metadata_output = project_dir.joinpath('data/interim/metadata.pkl')
    text_output = project_dir.joinpath('data/interim/text.pkl')
    metadata.to_pickle(metadata_output)
    text_merged.to_pickle(text_output)

    logger.info('finished preprocessing ../data/raw/modules.csv, '
                'output saved to ../data/interim/ as metadata.pkl and text.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
