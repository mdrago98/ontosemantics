
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span, Token, Doc
from hashlib import md5
from bioengine.nlp_processor import TaggerFactory


class BiologicalNamedEntity:
    """
    A class extending the becas ner to interface with spacy.
    """
    name = 'becas_ner'

    def __init__(self, nlp, label='MISC'):
        nlp.vocab.strings.add('MISC')
        self.label = nlp.vocab.strings[label]
        self.matcher = PhraseMatcher(nlp.vocab)
        Span.set_extension('medical_mapping', default=[])

    # TODO: strip the medical mapping functionality into the own function and put after span merge
    def __call__(self, doc, *args, **kwargs):
        nlp = English()
        tagger = TaggerFactory.factory([str(sentence) for sentence in doc.sents])
        named_ents = tagger.tag_sentences()
        terms = {md5(bytes(term['text'].lower(), 'utf-8')).hexdigest(): term['terms'] for sentence in named_ents
                 for term in sentence['terms']}
        patterns = [nlp(term) for term in terms.keys()]
        for pattern in patterns:
            [nlp.vocab.strings.add(token.text) for token in pattern]
        matches = [self.get_spans(sentence, doc) for sentence in named_ents]
        spans = []
        for sentence in matches:
            for term in sentence:
                if term is not None:
                    term_hash = md5(bytes(term.text.lower(), 'utf-8')).hexdigest()
                    if term_hash in terms.keys():
                        term._.set('medical_mapping', terms[term_hash])
                        spans.append(term)
                        doc.ents = list(doc.ents) + [term]
        # merge ent spans after not to mess with the ordering
        for span in spans:
            span.merge()
        return doc

    @staticmethod
    def get_spans(terms: dict, doc: Doc, label: str = 'MISC') -> list:
        """
        A helper function that returns a list of spacy spans from becas tagged entities.
        :param terms: A dictionary of becas named entities
        :param doc: the document object
        :param label: the label to assign the entity
        :return:
        """
        # doc.char_span(78, 96, label='MED')
        return [doc.char_span(term['start'], term['end'], label) for term in terms['terms']]
