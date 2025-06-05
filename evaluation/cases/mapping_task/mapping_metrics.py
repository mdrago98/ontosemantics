from random import choice, choices

import plac
from pandas import DataFrame, read_csv

from knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from evaluation.cases.cbr_task.calculate_cdr_metrics import get_db_relations, cal_tp_rate, generate_metrics


def query() -> str:
    """
    A function that returns a query for obtaining relations
    :return:
    """
    return """
    MATCH (doc:Publication)<-[:located_in]-(subject:ChemicalSubstance)-[association]->(obj:Disease)-[:located_in]->(sec_doc:Publication)
    WHERE association.relation <> 'null' and obj.name <> 'null' and subject.name <> 'null' and doc.pmid in {pmids} and association.relation = 'null' and type(association) <> 'could_refer_to' and type(association) <> 'located_in' and sec_doc.pmid = doc.pmid
    RETURN DISTINCT subject.name as subject, subject.synonym as subj_syn, association.relation as predicate, obj.name as obj, obj.synonym as disease_synonym, type(association) as rel_type, doc.pmid as pmid, doc.abstract as abstract
    """


# ['8312983', '24114426', '24190587', '25006961', '11745287', '3125850', '3191389', '11105626', '7967231', '9071336', '15276093', '16083708', '8701013', '2614930', '23846525', '20431083', '17828434', '7248895', '1756784', '10986547']
# ['15265979', '20169779', '17351238', '16880771', '20495512', '10726030', '256433', '7661171', '1449452',
#              '20009434', '24275640', '6893265', '8819482', '3383127', '6585590', '24464946', '7315949', '7315949',
#              '9570197', '689020']
# pmids = ['15265979', '20169779', '17351238', '16880771', '20495512', '10726030', '256433', '7661171', '1449452',
#          '20009434', '24275640', '6893265', '8819482', '3383127', '6585590', '24464946', '7315949', '7315949',
#          '9570197', '689020', '15265979', '20169779', '17351238', '16880771', '20495512', '10726030', '256433',
#          '7661171', '1449452',
#          '20009434', '24275640', '6893265', '8819482', '3383127', '6585590', '24464946', '7315949', '7315949',
#          '9570197', '689020', '8312983', '24114426', '24190587', '25006961', '11745287', '3125850', '3191389',
#          '11105626', '7967231', '9071336', '15276093', '16083708', '8701013', '2614930', '23846525', '20431083',
#          '17828434', '7248895', '1756784', '10986547']
pmids = ['3475563', '1280707', '16083708', '2484903', '10677406', '19293073', '17721298', '24341598', '9889429',
          '20595935', '2383364', '24333387', '7176945', '9128918', '6806735', '23871786', '3990093', '18340638',
          '12059909', '19553912', '24451297', '20959502', '24653743', '24971338', '9759693', '19020118', '3711722',
          '24209900', '3074291', '9323412', '24088636', '3191389', '891494', '6627074', '920167', '25084821', '9570197',
          '2625524', '20431083', '20927253']


def main(cbr_task_csv):
    """
    Main entry point
    """
    task_df: DataFrame = read_csv(cbr_task_csv)
    task_df['pmid'] = task_df.pmid.apply(str)
    results = []
    for i in range(0, 4):
        current_pmids = choices(pmids, k=20)
        associations_df = get_db_relations(KnowledgeGraphConnection(), current_pmids)
        synonyms_disease = {row['obj']: row['disease_synonym'] for index, row in associations_df.iterrows()}
        synonyms_subj = {row['subject']: row['subj_syn'] for index, row in associations_df.iterrows()}
        term_dict = {**synonyms_subj, **synonyms_disease}
        tp_tot = []
        fp_tot = []
        fn_tot = []
        found_pmids = []
        for pmid in set(associations_df['pmid']):
            found_pmids += [pmid]
            filtered_task_df = task_df[task_df.pmid == pmid]
            filtered_associations = associations_df[associations_df.pmid == pmid]
            gold_stand_chem_dis = [(row['chemical'], row['disease']) for index, row in filtered_task_df.iterrows()]
            chemical_dis_rel = [(row['subject'], row['obj']) for index, row in filtered_associations.iterrows()]
            tp, tn, fp, fn = cal_tp_rate(gold_stand_chem_dis, chemical_dis_rel, term_dict)
            tp_tot += tp
            # tn_tot += tn
            fp_tot += fp
            fn_tot += fn
        results.append(generate_metrics(len(set(tp_tot)), len(set(fn_tot)), len(set(fp_tot))))
    DataFrame(results)


if __name__ == '__main__':
    plac.call(main)
