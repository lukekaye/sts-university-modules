# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def main():
    """ Preprocesses raw modules.csv in ../raw
    Deal with IsDummy = True records, discard meaningless fields, standardise string fields and clean natural language
    Output saved as modules_interim.csv,in ../interim.
    """
    logger = logging.getLogger(__name__)
    logger.info('preprocessing raw data from modules.csv in ../raw')

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
                                          'IsSapUploadDisabled'
                                          ])

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

    # henceforth preprocessing is concerned solely with:
    # Aims, OutlineOfSyllabus, IntendedKnowledgeOutcomes and IntendedSkillOutcomes

    # replace short, meaningless natural language values with missing values
    modules.Aims.replace('Original Summary:', pd.NA, inplace = True)
    modules.OutlineOfSyllabus.replace(['TBA', 'See above.', 'Not applicable'], pd.NA, inplace=True)
    modules.IntendedKnowledgeOutcomes.replace('Specific to approved title', pd.NA, inplace=True)
    # TODO: there are other short values that should be removed, check the notebook

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
