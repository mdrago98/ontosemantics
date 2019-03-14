from collections import deque
from tokenize import Token

from spacy.tokens import Span, Doc
from grammaregex import find_tokens


def get_noun_verb_noun_phrases(sentence: Span) -> list:
    """
    A function that returns a list of noun verb noun chunks
    :param sentence: The sentence from which to extract the chunk
    :return: a list of tuples representing the noun verb noun chunk
    """
    verb_chunks = []
    for verb in find_tokens(sentence, 'VERB'):
        right_nouns = list(filter(lambda x: x.pos_ == 'NOUN', verb.rights))[0]
        left_nouns = list(filter(lambda x: x.pos_ == 'NOUN', verb.lefts))[0]
        verb_chunks.append((augment_noun_with_adj(left_nouns), verb, augment_noun_with_adj(right_nouns)))
    return verb_chunks


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
    return adjective, noun


def get_full_adj(adjective: Token, path: list = None):
    """
    A method that traverses the dependency tree to obtain the full adjective
    :param adjective: root adjective
    :param path:
    """
    if path is None:
        path = [adjective]
    possible_adj = list(filter(lambda child: child.pos_ == 'ADJ', adjective.children))
    if len(possible_adj) == 0:
        return path
    for child_adj in possible_adj:
        return get_full_adj(child_adj, path + [child_adj])


