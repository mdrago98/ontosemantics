from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span, Token
import spacy

from src.bioengine.preprocessor import TaggerFactory


def main():
    # For simplicity, we start off with only the blank English Language class
    # and no model or pre-defined pipeline loaded.
    nlp = spacy.blank('xx')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    becas_ner = BecasNamedEntity(nlp)  # initialise component
    nlp.add_pipe(becas_ner) # add it to the pipeline
    doc = nlp(u"Recombinant neuregulin-2beta induces the tyrosine phosphorylation of ErbB2, ErbB3 and ErbB4 in cell "
              u"line express all of these erbb family receptor")
    print()


class BecasNamedEntity:
    name = 'becas_ner'

    def __init__(self, nlp, label='MISC'):
        nlp.vocab.strings.add('MISC')
        self.label = nlp.vocab.strings[label]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.terms = None
        Span.set_extension('medical_mapping', default=[])

    def __call__(self, doc, *args, **kwargs):
        nlp = English()
        becas = TaggerFactory.factory([str(sentence) for sentence in doc.sents], 'becas')
        named_ents = becas.tag_sentences()
        terms = {term['text']: term['terms'] for sentence in named_ents for term in sentence['terms']}
        patterns = [nlp(term) for term in terms.keys()]
        self.matcher.add('MEDICAL', None, *patterns)
        matches = self.matcher(doc)
        spans = []
        for _, start, end in matches:
            entity = Span(doc, start, end, label=self.label)
            entity._.set('medical_mapping', terms[entity.text])
            spans.append(entity)
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            span.merge()
        return doc


if __name__ == '__main__':
    main()
