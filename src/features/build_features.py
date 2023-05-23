# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import itertools
import networkx as nx


def main():
    '''
    Builds features for modelling from cleaned data modules_pp.pkl in ../interim
    Output saved as x, in ../processed.
    '''
    logger = logging.getLogger(__name__)
    logger.info('building features from ../data/interim/modules_pp.pkl')

    # load modules_pp.pkl
    modules_pp_path = project_dir.joinpath('data/interim/modules_pp.pkl')
    modules_pp = pd.read_pickle(modules_pp_path)

    # split modules table into metadata and features tables; ModuleCode and Module_Id are (equivalent) primary keys
    metadata = modules_pp[['ModuleCode', 'SapObjectId', 'Title', 'MaxCapacity',
                           'IsNew', 'Semester1Offered', 'Semester1CreditValue', 'Semester2Offered',
                           'Semester2CreditValue', 'Semester3Offered', 'Semester3CreditValue', 'EctsCreditValue',
                           'FheqLevel', 'Mode', 'Delivery', 'IsOffered',
                           'PreRequisiteComment', 'CoRequisiteComment', 'Availability', 'StudyAbroad',
                           'GraduateSkillsFrameworkApplicable', 'SchoolCode', 'MarkingScale', 'Module_Id',
                           'TeachingLocation']]
    features = modules_pp[['ModuleCode', 'Module_Id', 'Aims_clean', 'OutlineOfSyllabus_clean',
                           'IntendedKnowledgeOutcomes_clean', 'IntendedSkillOutcomes_clean']]

    # rename text fields in features table
    features = features.rename(columns = {'Aims_clean': 'Aims',
                                          'OutlineOfSyllabus_clean': 'OutlineOfSyllabus',
                                          'IntendedKnowledgeOutcomes_clean': 'IntendedKnowledgeOutcomes',
                                          'IntendedSkillOutcomes_clean': 'IntendedSkillOutcomes'})

    # merge records that share at least one duplicate text value, retaining longest text value per field
    # this procedure takes multiple steps

    # todo: delete these options
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.min_rows', 100)

    # get tables of records that have duplications in at least the given field; these tables are not disjoint
    aim_duplicates = features[features.duplicated(subset=['Aims'], keep=False)]
    oos_duplicates = features[features.duplicated(subset=['OutlineOfSyllabus'], keep=False)]
    iko_duplicates = features[features.duplicated(subset=['IntendedKnowledgeOutcomes'], keep=False)]
    iso_duplicates = features[features.duplicated(subset=['IntendedSkillOutcomes'], keep=False)]

    # group ModuleCodes that share values in the given field together, in lists
    aim_grouped = aim_duplicates.groupby(['Aims'
                                          ])['ModuleCode'].apply(list).reset_index(name='RelatedModuleCodes'
                                                                                   )['RelatedModuleCodes']
    oos_grouped = oos_duplicates.groupby(['OutlineOfSyllabus'
                                          ])['ModuleCode'].apply(list).reset_index(name='RelatedModuleCodes'
                                                                                   )['RelatedModuleCodes']
    iko_grouped = iko_duplicates.groupby(['IntendedKnowledgeOutcomes'
                                          ])['ModuleCode'].apply(list).reset_index(name='RelatedModuleCodes'
                                                                                   )['RelatedModuleCodes']
    iso_grouped = iso_duplicates.groupby(['IntendedSkillOutcomes'
                                          ])['ModuleCode'].apply(list).reset_index(name='RelatedModuleCodes'
                                                                                   )['RelatedModuleCodes']

    # merge the tables of grouped duplications with the full set of singleton list modules
    all_grouped = pd.concat([aim_grouped,
                             oos_grouped,
                             iko_grouped,
                             iso_grouped,
                             features['ModuleCode'].rename('RelatedModuleCodes').apply(lambda mod_code: [mod_code])],
                            ignore_index = True)

    # graph theoretic approach: merge ModuleCode lists that share common elements by finding connected components
    # build graph
    graph = nx.from_edgelist(itertools.chain.from_iterable(itertools.pairwise(pair) for pair in sorted(all_grouped)))
    graph.add_nodes_from(set.union(*map(set, all_grouped)))

    # find connected components and sort
    disjoint_modules = list(nx.connected_components(graph))
    disjoint_modules = [sorted(list(module_codes)) for module_codes in disjoint_modules]
    disjoint_modules = pd.Series(data = sorted(disjoint_modules))


    # todo: consider using bag-of-words in conjunction with IsOffered to further remove duplicate modules
    # todo: join RelatedModuleCode subgraphs to text fields, retaining longest value per field per record group


    # create concatenated feature representation; missing values are converted to a blank string for the concatenation
    # features['Concatenated'] = features.Aims.fillna('') + ' ' + \
    #                            features.OutlineOfSyllabus.fillna('') + ' ' + \
    #                            features.IntendedKnowledgeOutcomes.fillna('') + ' ' + \
    #                            features.IntendedSkillOutcomes.fillna('')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()