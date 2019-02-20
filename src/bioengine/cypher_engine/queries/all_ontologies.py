def get_go_labels(limit: int = 500):
    return f"MATCH (n:GO) RETURN n.label LIMIT {limit}"


def get_go_labels_by_sub_ontology(sub_onto_name: str, limit: int = 500):
    return f'match (n: GO{{`annotation-has_obo_namespace`: ["{sub_onto_name}" ]}}) return n.label limit {limit}'


def get_doid_labels(limit: int = 500):
    return f"MATCH (n:DOID) RETURN n.label LIMIT {limit}"


def get_chebi_labels(limit: int = 500):
    return f"MATCH (n:ZEBRAFISH_ANATOMICAL_ONTOLOGY) RETURN n.label LIMIT {limit}"
