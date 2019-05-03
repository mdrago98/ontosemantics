from collections import namedtuple

from spacy.tokens import Span, Doc, Token

Relation = namedtuple('Relation', 'effector relation effectee negation')
Part = namedtuple('Part', 'adjectives nouns')


SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl']
OBJECTS = ['dobj', 'dative', 'attr', 'oprd', 'pobj']
ADJECTIVES = ['acomp', 'advcl', 'advmod', 'amod', 'appos', 'nn', 'nmod', 'ccomp', 'complm',
              'hmod', 'infmod', 'xcomp', 'rcmod', 'poss', ' possessive']
COMPOUNDS = ['compound']
PREPOSITIONS = ['prep']
NEGATIONS = {'no', 'not', "n't", 'never', 'none'}


def get_noun_verb_chunks(doc: Doc):
    """
    A function that returns all <noun><verb><noun> chunks
    :param doc: A spacy document representation.
    :return: a list of noun verb chunks
    """
    chunks = []
    for sentence in doc.sents:
        chunks += get_noun_verb_noun_phrases_from_sentence(sentence)
    return chunks


def get_noun_verb_noun_phrases_from_sentence(sentence: Span) -> list:
    """
    A function that returns a list of noun verb noun chunks
    :param sentence: The sentence from which to extract the chunk
    :return: a list of tuples representing the svo chunk
    """
    svos = []
    verbs = [token for token in sentence if token.pos_ == 'VERB' and token.dep_ != 'aux']
    for verb in verbs:
        subjects, is_negation = get_subjects(verb)
        if len(subjects) > 0:
            verb, objects = get_dependencies(verb)
            for subject in subjects:
                potential_subject = get_co_ref(subject)
                if potential_subject is not None:
                    subject = potential_subject
                for obj in objects:
                    negated_object = is_negative(obj)
                    svos.append(
                        Relation(subject, verb, [augment_noun_with_adj(obj)] + generate_sub_compound(obj),
                                 True if is_negation or negated_object else False))
    return svos


def enrich_svo(relation: Relation) -> list:
    """
    A function that finds adps or conjunctions and finds new svo triples using these.
    :param relation: the relation to augment
    :return: a list of augmented relationships
    """
    dependency = relation[2][1]

    pass


def get_subjects_from_conjunctions(subjects: list) -> list:
    """
    A function to retrieve subjects from conjunctions
    :param subjects: the subjects
    :return: a list of enriched subjects
    """
    extended_subjects = []
    for subject in subjects:
        right_deps = {tok.lower_ for tok in subject.rights}
        if 'and' in right_deps:
            extended_subjects += [tok for tok in subject.rights if tok.dep_ in SUBJECTS or tok.pos_ == 'NOUN']
            if len(extended_subjects) > 0:
                extended_subjects += get_subjects_from_conjunctions(extended_subjects)
    return extended_subjects


def find_subjects(token: Token) -> tuple:
    """
    An recursive function to get all subjects in a sentence
    :param token: the root token
    :return: a tuple containing a list of subjects and if they are negated
    """
    head: Token = token.head
    subjects = ([], False)
    while head.pos_ != 'VERB' and head.pos_ != 'NOUN' and head.head != head:
        head = head.head
    if head.pos_ == 'VERB':
        subs = [tok for tok in head.lefts if tok.dep_ == 'SUB']
        if len(subs) > 0:
            verb_negated = is_negative(head)
            subs += get_subjects_from_conjunctions(subs)
            subjects = (subs, verb_negated)
        elif head.head != head:
            return find_subjects(head)
    elif head.pos_ == 'NOUN':
        subjects = ([head], is_negative(token))
    return subjects


def get_subjects(root: Token) -> tuple:
    """
    A method that  retrieves the subjects
    :param root: the root of the sentence
    :return: a tuple containing the subjects and a flag for negation (True IFF negative)
    """
    negation = is_negative(root)
    subjects = [token for token in root.lefts if token.dep_ in SUBJECTS and token.pos_ != 'DET']
    if len(subjects) > 0:
        subjects += get_subjects_from_conjunctions(subjects)
    else:
        found_subjects, negation = find_subjects(root)
        subjects += found_subjects
    return subjects, negation


def get_objs_from_prepositions(tokens: list) -> list:
    """
    A method that tries to resolve the co reference to objects from prepositions.
    :param tokens: a list of prepositions
    :return: a list objects
    """
    objs = []
    for token in tokens:
        if token.pos_ == 'ADP' and token.dep_ == 'prep':
            objs += [tok for tok in token.rights if tok.dep_ in OBJECTS]
    return objs


def get_obj_from_xcomp(tokens: list) -> tuple:
    """
    Returns an object from compositions
    :param tokens: A list of tokens
    :return: a tuple consisting of list of objects and a flag for negation (True IFF negative)
    """
    result = (None, None)
    for token in tokens:
        if token.pos_ == 'VERB' and token.dep_ == 'xcomp':
            verb = token
            rights = list(verb.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs += get_objs_from_prepositions(rights)
            if len(objs) > 0:
                result = verb, objs
    return result


def get_objs_from_conjunctions(objs: list) -> list:
    """
    A recursive function that returns lists of objects from conjunctions
    :param objs: A list of objects
    :return: A list of enriched objects
    """
    more_obj = []
    for obj in objs:
        right_deps = {tok.lower_ for tok in obj.rights}
        if 'and' in right_deps:
            more_obj += [tok for tok in obj.rights if tok.dep_ in OBJECTS or tok.pos_ == 'NOUN']
            if len(more_obj) > 0:
                more_obj += get_objs_from_conjunctions(more_obj)
    return more_obj


def get_dependencies(root) -> tuple:
    """
    A method to get dependencies from the root of a sentence
    :param root: the root
    :return: a tuple containing the root and the dependencies
    """
    rights = list(root.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs += get_objs_from_prepositions(rights)

    potential_new_verb, potential_new_objs = get_obj_from_xcomp(rights)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs += potential_new_objs
        root = potential_new_verb
    if len(objs) > 0:
        objs += get_objs_from_conjunctions(objs)
    return root, objs


def is_negative(root: Token) -> bool:
    """
    A method to check for negations
    :param root: the root to check
    :return: True IFF is negative
    """
    negated = False
    for dep in list(root.lefts) + list(root.rights):
        if dep.lower_ in NEGATIONS:
            negated = True
    return negated


def generate_sub_compound(root) -> list:
    """
    A method that generates sub compounds
    :param root: the root to start from
    :return: the sub compound
    """
    sub_compounds = []
    for token in root.lefts:
        if token.dep_ in COMPOUNDS:
            sub_compounds += generate_sub_compound(token)
    sub_compounds.append(root)
    for token in root.rights:
        if token.dep_ in COMPOUNDS:
            sub_compounds.extend(generate_sub_compound(token))
    return sub_compounds


# TODO: Update the augment with adj code to take a list of nouns
def augment_noun_with_adj(noun: Token) -> tuple:
    """
    A function that gets the descendent adjectives and attaches them to the noun
    :param noun: the noun to augment
    :return: tuple representing the augmented noun
    """
    child_adjectives = list(filter(lambda x: x.pos_ == 'ADJ', noun.children))
    adjective = []
    if len(child_adjectives) != 0:
        adjective = get_full_adj(child_adjectives[0])
    return adjective


def get_co_ref(pronoun: Token) -> Token:
    """
    A method that given a pronoun token returns the referenced noun
    :param pronoun: the pronoun
    """
    ref = []
    # work around neural coref bug, open issue #110
    try:
        if pronoun.pos_ == 'PRON' and pronoun._.in_coref:
            # TODO: get noun coref
            ref = [ref.main for ref in pronoun._.coref_clusters]
    except:
        ref = []
    return ref[0] if len(ref) == 1 and ref[0][0].pos_ != 'PRON' else None


def get_full_adj(adjective: Token, path: list = None):
    """
    A method that traverses the dependency tree to obtain the full adjective
    :param adjective: root adjective
    :param path: A path from the dependency tree of supplementing adjectives
    """
    if path is None:
        path = [adjective]
    possible_adj = list(filter(lambda child: child.pos_ == 'ADJ', adjective.children))
    if len(possible_adj) == 0:
        return path
    for child_adj in possible_adj:
        return get_full_adj(child_adj, path + [child_adj])


def enrich_adp(adp: Token, children: list = None):
    """
    A method that enriches adp tokens with the next representative noun in the tree
    :param children:
    :param adp: a spacy token representing the ADP
    """
    if children is None:
        children = []
    if adp.pos_ == 'ADP':
        new_children = list(adp.children)
        children += new_children if new_children is not None else []
        nouns = list(filter(lambda x: x.pos_ == 'NOUN', children))
        if len(nouns) > 0 and adp.pos_ != 'NOUN':
            return enrich_adp(children[0], children)
    return adp
