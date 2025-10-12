from unittest import TestCase

import spacy
from benepar.spacy_plugin import BeneparComponent
from nose_parameterized import parameterized, param

from bioengine.nlp_processor.entity_normalization import get_base_word, remove_stop_words, convert_parenthesis, remove_punctuation
from bioengine.nlp_processor.extensions import BiologicalNamedEntity


class TestEntityNormalization(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Sets up the test suite
        """
        # Emulate the medical spacy component
        cls.nlp = spacy.load('en_coref_sm', disable=['ner'])
        cls.nlp.add_pipe(BiologicalNamedEntity(cls.nlp))
        cls.nlp.add_pipe(BeneparComponent("benepar_en2"))
        stopword_loc = '../../resources/stopwords.txt'
        with open(stopword_loc) as stop_word_file:
            stop_words = stop_word_file.readlines()
        for stop_word in stop_words:
            lexeme = cls.nlp.vocab[stop_word.strip()]
            lexeme.is_stop = True

    @classmethod
    def tearDownClass(cls):
        """
        Tears down the test case dependencies
        """
        cls.nlp = None

    @parameterized.expand([
        param(word='the cell', expected='cell'),
        param(word='the', expected=''),
        param(word='as cell function', expected='cell function'),
    ])
    def test_stop_word_removal(self, word: str, expected: str):
        value = remove_stop_words(word, self.nlp)
        assert value == expected

    @parameterized.expand([
        param(word='entities', expected='entity'),
        param(word='cells', expected='cell'),
        param(word='cell', expected='cell')
    ])
    def test_lemmatization(self, word: str, expected: str):
        value = get_base_word(word, self.nlp)
        assert value == expected

    @parameterized.expand([
        param(word='NNN(MMM)', expected='NNN-MMM'),
        param(word='NNN(A),(a)', expected='NNN-A-a'),
        param(word='NNN', expected='NNN'),
        param(word='', expected='')
    ])
    def test_hyphen_rule(self, word: str, expected: str):
        value = convert_parenthesis(word)
        assert value == expected

    @parameterized.expand([
        param(word='NNN,MMM.!)', expected='NNN MMM'),
        param(word='NNN)', expected='NNN'),
        param(word='', expected=''),
    ])
    def test_strip_punctuation(self, word: str, expected: str):
        value = remove_punctuation(word, self.nlp)
        assert value == expected
