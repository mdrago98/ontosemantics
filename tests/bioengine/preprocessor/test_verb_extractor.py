from pymetamap import ConceptMMI

from src.bioengine.preprocessor.verb_extractor import chunk_genia_output, meta_map_trigger_regex, \
    align_concepts_with_sentences
from unittest import TestCase


class TestVerbExtractor(TestCase):
    """
    A test suite that tests functions related to the verb extractor
    """

    def test_chunk_genia_output_returns_list_of_tags_with_one_chunk(self):
        sample_genia_output = [[('e.g.', 'e.g.', 'FW', 'B-NP', 'O'),
                                ('Red', 'Red', 'NNP', 'I-NP', 'O'),
                                ('algae', 'algae', 'FW', 'I-NP', 'O'),
                                (':', ':', ':', 'O', 'O')]]
        res = chunk_genia_output(sample_genia_output)
        assert res == [[['e.g.', 'Red', 'algae']]]

    def test_chunk_genia_output_returns_list_of_tags_with_half_a_chunk(self):
        sample_genia_output = [[('e.g.', 'e.g.', 'FW', 'B-NP', 'O'),
                                ('Red', 'Red', 'NNP', 'I-NP', 'O'),
                                ('algae', 'algae', 'FW', 'I-NP', 'O')]]
        res = chunk_genia_output(sample_genia_output)
        assert res == [[['e.g.', 'Red', 'algae']]]

    def test_chunk_genia_output_returns_list_of_tags_with_one_and_a_half_a_chunks(self):
        sample_genia_output = [[('e.g.', 'e.g.', 'FW', 'B-NP', 'O'),
                                ('Red', 'Red', 'NNP', 'I-NP', 'O'),
                                ('algae', 'algae', 'FW', 'I-NP', 'O'),
                                ('Aqueous', 'Aqueous', 'JJ', 'B-NP', 'O'),
                                ('extracts', 'extract', 'NNS', 'I-NP', 'O')]]
        res = chunk_genia_output(sample_genia_output)
        assert res == [[['e.g.', 'Red', 'algae'], ['Aqueous', 'extract']]]

    def test_meta_map_trigger_regex_returns_regex_that_matches_concept_names(self):
        regex = meta_map_trigger_regex()
        concept = regex.match('["Extract"-tx-2-"extracts"-noun-0]').group(1)
        assert concept == 'Extract'

    def test_meta_map_trigger_regex_returns_regex_that_matches_concept_names_with_digit(self):
        regex = meta_map_trigger_regex()
        concept = regex.match('["CA9"-tx-1-"Ca9"-noun-0]').group(1)
        assert concept == 'CA9'

    def test_meta_map_trigger_regex_returns_regex_that_matches_concept_names_with_percentage_sign(self):
        regex = meta_map_trigger_regex()
        concept = regex.match('["% Activity"-tx-1-"activity"-noun-0]').group(1)
        assert concept == 'Activity'

    # TODO finish test assertion and add test with two sentences
    def test_align_concepts_to_sentences_with_one_sentence(self):
        sentence = 'Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited'
        concept = [ConceptMMI(index='tmpddqi0zz9',
                              mm='MMI', score='11.49',
                              preferred_name='Rhodophyta',
                              cui='C0002033', semtypes='[plnt]',
                              trigger='["Red Algae"-tx-1-"Red algae"-noun-0]',
                              location='TX', pos_info='0/9', tree_codes='B01.650.700')]
        result = align_concepts_with_sentences([sentence], concepts=concept)
        assert result[0][0] == sentence
        assert result[0][1] == concept

    def test_align_concepts_to_sentences_with_two_sentences_shifted_position(self):
        first_sentence = 'Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited'
        second_sentence = 'Both ethanol and methanol extracts of Gracilaria tenuistipitata'
        concept = [ConceptMMI(index='tmpddqi0zz9',
                              mm='MMI', score='11.49',
                              preferred_name='Rhodophyta',
                              cui='C0002033', semtypes='[plnt]',
                              trigger='["Red Algae"-tx-1-"Red algae"-noun-0]',
                              location='TX', pos_info='0/9', tree_codes='B01.650.700')]
        second_concept = [ConceptMMI(index='tmprda1cogu', mm='MMI',
                                     score='14.64',
                                     preferred_name='Ethanol',
                                     cui='C0001962',
                                     semtypes='[orch,phsu]',
                                     trigger='["ETHANOL"-tx-1-"ethanol"-noun-0]',
                                     location='TX',
                                     pos_info='5/7',
                                     tree_codes='D02.033.375;x.x.x.x')]
        result = align_concepts_with_sentences([first_sentence, second_sentence], concepts=second_concept + concept)
        assert result[0][0] == first_sentence
        assert result[0][1] == concept
        assert result[1][0] == second_sentence
        assert result[1][1] == second_concept
