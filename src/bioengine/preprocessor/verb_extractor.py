import spacy
from multiprocessing.dummy import Pool as ThreadPool
from src.bioengine.preprocessor import TaggerFactory
from re import compile

# spacy.prefer_gpu()
from collections import deque


def meta_map_trigger_regex():
    """
    A function hat wraps around a regular expression for getting the object from the trigger value given by metamap.
    :return: a compiled regex
    """
    return compile(r'\[\"([a-zA-Z\d*\s]*)\"{1}')


def align_concepts_with_sentences(sentences: list, concepts: list):
    """
    A method that combines concepts to sentences for easier processing.
    :param sentences: a list of sentences
    :param concepts: a list of concepts
    :return: a list of merged sentences and concepts
    """
    matched_sentences = []
    for sentence in sentences:
        matched_sentences.append((sentence, list(filter(lambda x: meta_map_trigger_regex().match(x.trigger.lower())
                                                        .group(1) in str(sentence).lower(), concepts))))
    return matched_sentences


def __chunk_genia_output(genia_sentence: deque, sentence_chunks=None) -> list:
    """
    A function that recursively combines tags from genia tagger's output into noun chunks
    :param genia_sentence: the sentence to get the noun chunks from
    :param sentence_chunks: a list of noun chunks currently extracted
    :return: a list of noun chunks
    """
    if sentence_chunks is None:
        sentence_chunks = []
    if len(genia_sentence) is not 0:
        tag_tuple = genia_sentence.popleft()
        if str(tag_tuple[3]).startswith('B'):
            sentence_chunks.append([tag_tuple[1]])
        elif str(tag_tuple[3]).startswith('I'):
            previous_chunk = sentence_chunks[len(sentence_chunks) - 1]
            previous_chunk.append(tag_tuple[1])
        return __chunk_genia_output(genia_sentence, sentence_chunks)
    else:
        return sentence_chunks


def chunk_genia_output(genia_output):
    """
    A multi threaded function that returns noun chunks from a list of sentences.
    :param genia_output: the output from the genia tagger
    :return: a list of noun chunks ordered by sentence
    """
    dequeue_genia_sentences = map(lambda x: deque(x), genia_output)
    pool = ThreadPool(4)
    return pool.map(__chunk_genia_output, dequeue_genia_sentences)


nlp = spacy.load('en_core_web_sm')

corpus = "e.g. Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited the " \
         "proliferation of human leukemic cell lines. Both ethanol and methanol extracts of Gracilaria tenuistipitata " \
         "reportedly had anti-proliferative effects on Ca9-22 oral cancer cells and were involved in cellular " \
         "apoptosis, DNA damage, and oxidative stress. [example source: PMC3674937]"
doc = nlp(corpus)
sentences = list(doc.sents)

with TaggerFactory.factory(sentences=sentences) as tagger:
    entities = tagger.tag_sentences()
    sent_concept = align_concepts_with_sentences(sentences, concepts=entities[0])
    print(sent_concept)

genia_sentences = [str(sentence) for sentence in sentences]
genia = TaggerFactory.factory(genia_sentences, tagger_type='genia')
genia_pos = genia.tag_sentences()
spacy_pos = [(token.text, token.pos_)for token in sentences[0]]

chunky_genia = chunk_genia_output(genia_pos)
print(spacy_pos)

