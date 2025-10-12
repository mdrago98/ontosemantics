from collections import deque
from multiprocessing.dummy import Pool as ThreadPool
from re import compile

from spacy.tokens import Doc, Token

from bioengine.nlp_processor.extensions import get_noun_verb_noun_phrases_from_sentence
from bioengine.nlp_processor.spacy_factory import MedicalSpacyFactory


def meta_map_trigger_regex():
    """
    A function hat wraps around a regular expression for getting the object from the trigger value given by metamap.
    :return: a compiled regex
    """
    return compile(r'\[\"%*([a-zA-Z\d*\s]*)\"{1}')


def meta_map_pos_regex():
    """
    A function that wraps around a regular expression to extract entity positions in a document/corpus
    :return: a compiled regex
    """
    return compile(r'((\d*)/(\d*))*')


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


def get_spans_from_entities(doc, positions):
    """
    A method that returns spans from a set of entities and documents
    :param doc: A spacy document representation
    :param positions: A list of positions in the form ['x/y']
    :return: a list of spacy spans of the entities
    """
    spans = []
    for position in positions:
        for split_positions in position.split(','):
            start, _, length = split_positions.partition('/')
            spans.append(doc.char_span(int(start), int(start) + int(length)))
    return spans


def merge_multiple_word_tokens(doc, entities):
    """
    A function that merges multiword tokens into one token depending on the metamap entities
    :param doc: a spacy parsed document
    :param entities: a list of lower case entities
    :return a new parsed document
    """
    spacy_ents = []
    for token in doc:
        for entity in entities:
            if token.i + 1 != len(doc):
                new_token = doc[token.i: token.nbor().i + 1]
                if new_token.lower_ == entity:
                    spacy_ents.append(new_token)
    for span in spacy_ents:
        span.merge()
    return doc


def get_spans_from_becas(entities: list, doc: Doc):
    return [doc.char_span(entity['start'], entity['end']) for entity in entities]


def filter_tokens_by_meta_map_ents(merged_doc: Doc, entities: list, merge_tokens=merge_multiple_word_tokens) -> iter:
    """
    A method that filers tokens in respect to a list of metamap entities
    :param merged_doc: a spacy document representation
    :param entities: a list of metamap entities
    :param merge_tokens: a function that merges multiword tokens
    :return:
    """
    return filter(lambda x: x.lower_ in entities, merged_doc)


def __get_dep_tree(tree: dict, children: deque = None):
    if len(children) == 0:
        return tree
    else:
        children = deque(filter(lambda x: x.dep_ != 'punct', children))
        child = children.popleft()
        additional_children = list(child.children)
        if child.text in tree.keys():
            tree[child] += additional_children
        else:
            tree[child] = additional_children
        children.extend(additional_children)
        return __get_dep_tree(tree, children)


def get_dep_tree(document: Doc) -> list:
    roots = list(filter(lambda x: x.dep_ == 'ROOT', document))
    return [__get_dep_tree({root: list(filter(lambda x: x.dep_ != 'punct',
                                              root.children))}, deque(root.children)) for root in roots]


def bfs(graph: dict, start: Token, end: Token):
    queue = [[start]]
    visited = set()

    while queue:
        # Gets the first path in the queue
        path = queue.pop(0)

        # Gets the last node in the path
        vertex = path[-1]

        # Checks if we got to the end
        if vertex == end:
            return path
        elif vertex not in visited:
            # enumerate all adjacent nodes, construct a new path and push it into the queue
            for current_neighbour in graph.get(vertex, []):
                new_path = list(path)
                new_path.append(current_neighbour)
                queue.append(new_path)

            visited.add(vertex)


def get_all_paths_root_leaf(root: Token, tree: dict) -> list:
    leaves = {key: value for key, value in tree.items() if len(value) == 0}
    return [bfs(tree, root, leaf) for leaf in leaves]


# TODO: Experiment with sending noun chunks to metamap instead of sentences


nlp = MedicalSpacyFactory.factory()

# corpus = "e.g. Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited the " \
#          "proliferation of human leukemic cell lines. Both ethanol and methanol extracts of Gracilaria tenuistipitata " \
#          "reportedly had anti-proliferative effects on Ca9-22 oral cancer cells and were involved in cellular " \
#          "apoptosis, DNA damage, and oxidative stress. [example source: PMC3674937]"

corpus = """CASE SUMMARY:
A 57-year old female with hyponatraemia. Her medications included desvenlafaxine, and symptoms included nausea, anxiety and confusion. The serum sodium at this time was 120 mmol/L, serum osmolality was 263 mosmol/kg, urine osmolality 410 mosmol/kg and urine sodium 63 mmol/L, consistent with a diagnosis of SIADH. Desvenlafaxine was ceased and fluid restriction implemented. After 4 days the sodium increased to 128 mmol/L and fluid restriction was relaxed. During her further 3 weeks inpatient admission the serum sodium ranged from 134 to 137 mmol/L during treatment with mirtazapine."""


doc = nlp(corpus)
sentences = [str(sentence) for sentence in doc.sents]
doc._.noun_verb_chunks

sent = list(doc.sents)[1]
test = get_noun_verb_noun_phrases_from_sentence(sent)
print()

