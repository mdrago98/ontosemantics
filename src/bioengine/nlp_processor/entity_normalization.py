from nlp_processor.spacy_factory import MedicalSpacyFactory
from re import compile


def normalize_batch(terms: list, nlp=None) -> tuple:
    """
    A function that normalizes a list of terms
    :param terms: a list of terms
    :param nlp: a spacy instance
    :return: a tuple with the term dictionary and all variations from the list of terms
    """
    if nlp is None:
        nlp = MedicalSpacyFactory.factory()
    term_lookup = {}
    variations = []
    for term in terms:
        entity, normalized_ents = normalize(term, nlp)
        variations += [normal_ent for normal_ent in normalized_ents if normal_ent != entity]
        term_lookup.update({entity: normalized_ents})
    return term_lookup, variations


def normalize(entity: str, nlp) -> tuple:
    """
    A function that generates alternate versions of entities
    :param entity: the base entity
    :param nlp: a spacy nlp instance
    :return: a dictionary of alternate words
    """
    if nlp is None:
        nlp = MedicalSpacyFactory.factory()
    return entity, {get_base_word(str(entity), nlp),
                    remove_stop_words(str(entity), nlp),
                    convert_parenthesis(str(entity)),
                    remove_punctuation(str(entity), nlp)}


def get_base_word(word: str, nlp) -> str:
    """
    A method that lematizes a word to get it's base form
    :param word: the word to lemmatize
    :param nlp: a spacy instance
    :return: a str
    """
    doc = nlp(str(word))
    return ' '.join([token.lemma_ for token in doc]).strip()


def remove_stop_words(word: str, nlp) -> str:
    """
    A function that removes stop words from entities
    :param word: the entity
    :param nlp: a space instance
    :return: the pruned entity
    """
    doc = nlp(str(word))
    return ' '.join([token.lemma_ for token in doc if not token.is_stop]).strip()


def convert_parenthesis(word: str) -> str:
    """
    A function that converts words in parenthesis to hyphenated words
    :param word: the word to convert
    :return: the converted word
    """
    hyphen_regex = compile(r'([a-zA-Z]*)(\()([a-zA-Z]*)(\))')
    matches = hyphen_regex.findall(word)
    normalized_entity = word
    if len(matches) is not 0:
        words = [word.strip() for groups in matches for word in groups if word.strip() not in ('(', ')', '')]
        normalized_entity = '-'.join(words)
    return normalized_entity


def remove_punctuation(word, nlp) -> str:
    """
    A function that removes parenthesis from the word
    :param word: the entity
    :param nlp: a spacy instance
    :return: the word, stripped out of punctuation
    """

    pruned_word = word
    if word is not '':
        doc = nlp(word)
        pruned_word = ' '.join([str(token) for token in doc if not token.is_punct]).strip()
    return pruned_word
