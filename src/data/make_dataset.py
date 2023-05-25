# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re
import contractions
import itertools
import networkx as nx


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
    Output saved as metadata.pkl and features.pkl, to ../interim
    '''
    logger = logging.getLogger(__name__)
    logger.info('preprocessing ../data/raw/modules.csv')

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
    features = modules[['ModuleCode', 'Aims_clean', 'OutlineOfSyllabus_clean', 'IntendedKnowledgeOutcomes_clean',
                        'IntendedSkillOutcomes_clean']]

    # rename text fields in features table
    features = features.rename(columns={'Aims_clean': 'Aims',
                                        'OutlineOfSyllabus_clean': 'OutlineOfSyllabus',
                                        'IntendedKnowledgeOutcomes_clean': 'IntendedKnowledgeOutcomes',
                                        'IntendedSkillOutcomes_clean': 'IntendedSkillOutcomes'})

    # henceforth merge records that share at least one duplicate text value, retaining longest text value per field
    # this procedure takes multiple steps

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
                            ignore_index=True)

    # graph theoretic approach: group ModuleCodes that share common elements by finding connected components
    # build graph
    graph = nx.from_edgelist(itertools.chain.from_iterable(itertools.pairwise(pair) for pair in sorted(all_grouped)))
    graph.add_nodes_from(set.union(*map(set, all_grouped)))

    # find connected components and sort
    disjoint_modules = list(nx.connected_components(graph))
    disjoint_modules = [sorted(list(module_codes)) for module_codes in disjoint_modules]
    disjoint_modules = pd.Series(data=sorted(disjoint_modules))

    # join RelatedModuleCodes to text fields, retaining longest value per text field per record group
    # create empty dataframe for ModuleCodes grouped by sharing text
    features_merged = pd.DataFrame(columns = ['ModuleCode', 'Aims', 'OutlineOfSyllabus',
                                              'IntendedKnowledgeOutcomes', 'IntendedSkillOutcomes'])
    # iterate through every group in disjoint_modules
    for module_group in disjoint_modules:
        # lists for each text field
        aim_set = []
        oos_set = []
        iko_set = []
        iso_set = []
        # iterate through each ModuleCode in the current group
        for module_code in module_group:
            # get text corresponding to ModuleCode
            text_values = features[features['ModuleCode'] == module_code][['Aims',
                                                                           'OutlineOfSyllabus',
                                                                           'IntendedKnowledgeOutcomes',
                                                                           'IntendedSkillOutcomes']]
            # append text to lists, depending on text field
            aim_set.append(text_values.Aims.to_string(index=False))
            oos_set.append(text_values.OutlineOfSyllabus.to_string(index=False))
            iko_set.append(text_values.IntendedKnowledgeOutcomes.to_string(index=False))
            iso_set.append(text_values.IntendedSkillOutcomes.to_string(index=False))
        # get record to be appended to features_merged; retains longest string per field to minimise semantic info lost
        record = [module_group,
                  max(aim_set, key = len),
                  max(oos_set, key = len),
                  max(iko_set, key = len),
                  max(iso_set, key = len)]
        # append the record to features_merged
        features_merged.loc[len(features_merged)] = record

    # save metadata and features_merged tables to ../data/interim as metadata.pkl and features.pkl
    metadata_output = project_dir.joinpath('data/interim/metadata.pkl')
    features_output = project_dir.joinpath('data/interim/features.pkl')
    metadata.to_pickle(metadata_output)
    features_merged.to_pickle(features_output)

    logger.info('finished preprocessing ../data/raw/modules.csv, '
                'output saved to ../data/interim/ as metadata.pkl and features.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
