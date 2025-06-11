from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.strings import StringStore
from spacy.tokens import Span

from bioengine.nlp_processor import TaggerFactory
from re import compile


def meta_map_trigger_regex():
    """
    A function hat wraps around a regular expression for getting the object from the trigger value given by metamap.
    :return: a compiled regex
    """
    return compile(r'\[\"%*([a-zA-Z\d*\s]*)\"{1}')


class SpacyMetaMap:
    name = 'metamap_ner'

    def __init__(self, nlp, label='MISC'):
        nlp.vocab.strings.add('MISC')
        self.label = nlp.vocab.strings[label]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.terms = None
        Span.set_extension('medical_mapping', default=[])

    def __call__(self, doc, *args, **kwargs):
        nlp = English()
        with TaggerFactory.factory(sentences=list(doc.sents)) as tagger:
            entities = tagger.tag_sentences()
            patterns = [meta_map_trigger_regex().match(concept.trigger).group(1).lower()
                        for concept in entities[0]]
            doc_patterns = []
            for pattern in set(patterns):
                if pattern not in nlp.vocab:
                    nlp.vocab.strings.add(pattern)
                doc_patterns.append(nlp(pattern))

            self.matcher.add('MEDICAL', None, *doc_patterns)
            matches = self.matcher(doc)
            spans = []
            for _, start, end in matches:
                entity = Span(doc, start, end, label=self.label)
                spans.append(entity)
                doc.ents = list(doc.ents) + [entity]
            for span in spans:
                span.merge()


def main():
    # For simplicity, we start off with only the blank English Language class
    # and no model or pre-defined pipeline loaded.
    nlp = English()
    rest_countries = SpacyMetaMap(nlp)  # initialise component
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(rest_countries)  # add it to the pipeline
    corpus = "e.g. Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited the " \
             "proliferation of human leukemic cell lines. Both ethanol and methanol extracts of Gracilaria tenuistipitata " \
             "reportedly had anti-proliferative effects on Ca9-22 oral cancer cells and were involved in cellular " \
             "apoptosis, DNA damage, and oxidative stress. [example source: PMC3674937]"

    doc = nlp(corpus)
    print(doc.ents)


main()
