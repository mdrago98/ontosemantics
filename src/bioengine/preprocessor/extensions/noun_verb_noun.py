from spacy.tokens import Span, Doc, Token
from grammaregex import find_tokens


def get_noun_verb_chunks(doc: Doc):
    """
    A function that returns all <noun><verb><noun> chunks
    :param doc: A spacy document representation.
    :return: a list of noun verb chunks
    """
    return [get_noun_verb_noun_phrases_from_sentence(sentence) for sentence in doc.sents]


def get_noun_verb_noun_phrases_from_sentence(sentence: Span) -> list:
    """
    A function that returns a list of noun verb noun chunks
    :param sentence: The sentence from which to extract the chunk
    :return: a list of tuples representing the noun verb noun chunk
    """
    verb_chunks = []
    for verb in find_tokens(sentence, 'VERB'):
        verb_chunks += [(get_nouns_from_children(list(verb.lefts)),
                         verb,
                         get_nouns_from_children(list(verb.rights))
                         )]
    return verb_chunks


def get_nouns_from_children(children: list) -> dict:
    """
    A function that returns the a dictionary of nouns from a list of children
    :param children: the children of a root verb
    :return: a noun dictionary
    """
    nouns = list(filter(lambda x: x.pos_ in ('NOUN', 'ADP', 'PROPN'), children))
    if len(nouns) == 0:
        nouns = [get_co_ref(pronoun) for pronoun in children if pronoun.pos_ == 'PRON']
    enriched_tokens = {enrich_adp(noun): enrich_noun(noun) for noun in nouns if
                       type(noun) == Token and noun is not None}
    enriched_spans = {enrich_adp(next(filter(lambda x: x.pos_ == 'NOUN', noun))): enrich_phrase(noun) for noun in nouns
                      if type(noun) == Span and noun is not None}
    return augment_nouns_with_adj({**enriched_spans, **enriched_tokens})


def augment_nouns_with_adj(nouns: dict) -> dict:
    """
    A method that augments a dictionary of nouns with their corresponding adjectives
    :param nouns: the noun dictionary
    :return: an augmented noun dictionary
    """
    return {noun: (augment_noun_with_adj(noun), nouns[noun]) for noun in nouns.keys()}


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


def enrich_phrase(phrase: Span) -> list:
    """
    A method that gets the noun and enriches the noun from a phrase
    :param phrase: a span representing the noun
    """
    phrases = [enrich_noun(token) for token in phrase if token.pos_ == 'NOUN']
    return phrases if len(phrases) > 0 else []


def enrich_noun(noun: Token, path: list = None) -> list:
    """
    A method that enriches nouns by traversing further down the dependency tree and get supplementary nouns supported
    by apposition.
    :param noun: the noun to enrich
    :param path: the dependency path from the root noun to the final noun
    :return:
    """
    if path is None:
        path = []
    next_adp: list = list(filter(lambda child: child.pos_ == 'ADP', noun.children))
    if len(next_adp) == 0:
        return path
    for child_adp in next_adp:
        noun_children: iter = filter(lambda x: x.pos_ == 'NOUN', child_adp.children)
        for noun_child in noun_children:
            # get nouns one step down the dep tree combined by conjunctions
            return enrich_noun(noun_child, path + [noun_child] + get_conjunction_nouns(noun_child))


def get_conjunction_nouns(noun: Token) -> list:
    """
    A helper function that returns nouns in conjunction to a root Noun.
    :param noun: the root noun
    :return: a list [conjunction, noun]
    """
    child_conjunctions = list(filter(lambda x: x.pos_ in ('CONJ', 'CCONJ', 'SCONJ'), noun.children))
    path = []
    if len(child_conjunctions) != 0:
        path = child_conjunctions + list(filter(lambda x: x.pos_ in 'NOUN', noun.children))
    return path


# TODO: build + test method that enriches adp tokens
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
