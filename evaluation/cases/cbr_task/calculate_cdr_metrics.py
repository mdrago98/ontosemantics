from os import mkdir
from os.path import join
from random import choice, choices

import plac
from pandas import read_csv, DataFrame, concat

from bioengine.knowledge_engine.connections import Connection
from bioengine.knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection


def query() -> str:
    """
    A function that returns a query for obtaining relations
    :return:
    """
    return """
    MATCH (doc:Publication)<-[:located_in]-(subject)-[association:chemical_to_disease_or_phenotypic_feature_association]->(obj)-[:located_in]->(sec_doc:Publication)
    WHERE association.relation <> 'null' and obj.name <> 'null' and subject.name <> 'null' and doc.pmid in {pmids} and association.relation <> 'null' and type(association) <> 'could_refer_to' and type(association) <> 'located_in' and sec_doc.pmid = doc.pmid
    RETURN DISTINCT subject.name as subject, subject.synonym as subj_syn, association.relation as predicate, obj.name as obj, obj.synonym as disease_synonym, type(association) as rel_type, doc.pmid as pmid, doc.abstract as abstract
    """


def get_db_relations(driver: Connection, pmids, association_query=query):
    return driver.execute_string_query(association_query(), pmids=pmids).to_data_frame()


def cal_tp_rate(gold_stand, results, synonyms: dict):
    results_lower = [(result[0].lower(), result[1].lower()) for result in results]
    gold_stand_lower = [(rel[0].lower(), rel[1].lower()) for rel in gold_stand]
    tp = [result for result in results_lower if result in gold_stand_lower]
    tn = 0
    fp = [result for result in results_lower if result not in gold_stand_lower]
    fn = [stan_rel for stan_rel in gold_stand_lower if stan_rel not in results_lower]
    return tp, tn, fp, fn


def generate_sample(pmids, program_rels, task_rels, output, sample_size=5):
    ran_pmids = choices(pmids, k=sample_size)
    for pmid in ran_pmids:
        base_path = join(output, pmid)
        string_task = task_rels[task_rels.pmid == pmid].to_string()
        filtered_rels = program_rels[program_rels.pmid == pmid]
        cols = filtered_rels.columns.tolist()
        string_prog = filtered_rels[[cols[7], cols[2], cols[6], cols[1]]].to_string()
        abstract = list(filtered_rels['abstract'])[0]
        final_out = f'{abstract} \n Task Result \n {string_task} \n Task Output \n {string_prog}'
        with open(f'{base_path}.txt', 'w') as file:
            file.write(final_out)


def generate_metrics(tp: int, fn: int, fp: int):
    precision: float = tp / (tp + fn) if tp is not 0 and fn is not 0 and fp is not 0 else 0
    recall: float = tp / (tp + fp) if tp is not 0 and fn is not 0 and fp is not 0 else 0
    fscore = ((2 * precision * recall) / (precision + recall)) if tp is not 0 and fn is not 0 and fp is not 0 else 0
    return [precision, recall, fscore]


def main(cbr_task_csv, output='', k=400, iterations=5):
    """
    Main entry point
    """
    task_df: DataFrame = read_csv(cbr_task_csv)
    task_df['pmid'] = task_df.pmid.apply(str)
    pmids = [f'{pmid}' for pmid in set(task_df['pmid'])]
    results = []
    for i in range(0, iterations):
        current_pmids = choices(pmids, k=k)
        associations_df = get_db_relations(KnowledgeGraphConnection(), current_pmids)
        synonyms_disease = {row['obj']: row['disease_synonym'] for index, row in associations_df.iterrows()}
        synonyms_subj = {row['subject']: row['subj_syn'] for index, row in associations_df.iterrows()}
        term_dict = {**synonyms_subj, **synonyms_disease}
        tp_tot = []
        fp_tot = []
        fn_tot = []
        found_pmids = []
        if 'pmid' in associations_df:
            for pmid in set(associations_df['pmid']):
                found_pmids += [pmid]
                filtered_task_df = task_df[task_df.pmid == pmid]
                filtered_associations = associations_df[associations_df.pmid == pmid]
                gold_stand_chem_dis = [(row['chemical'], row['disease']) for index, row in filtered_task_df.iterrows()]
                chemical_dis_rel = [(row['subject'], row['obj']) for index, row in filtered_associations.iterrows()]
                tp, tn, fp, fn = cal_tp_rate(gold_stand_chem_dis, chemical_dis_rel, term_dict)
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn
        results.append(generate_metrics(len(set(tp_tot)), len(set(fn_tot)), len(set(fp_tot))))
    DataFrame(results)
        # generate_sample(found_pmids, associations_df, task_df, output)


if __name__ == '__main__':
    plac.call(main)
