from re import compile


def apa_citation():
    """
    A wrapper function that returns a compiled regex for catching apa style citations.
    :return: a complied regex
    """
    return compile(r'\(\D*\d{4}(;\D*\d{4})*\)')


def pmc_single_citation():
    """
    A wrapper function that returns a compiled regex for catching pmc style citations.
    :return: a complied regex
    """
    return compile(r"(\s*)?\(\s*REF\s*\.\s*\d*\s*\)(\s*)?")


def pmc_multiple_citations():
    """
    A wrapper function that returns a compiled regex for catching multiple pmc style citations.
    :return: a complied regex
    """
    return compile(r'(\s*)?\(\s*REFS\s*\.*\s*\d*\,*\s*\d*\s*\)(\s*)?')


def wiki_style_citation():
    """
    A wrapper function that returns a compiled regex for catching apa style citations
    :return: a complied regex
    """
    return compile(r'\[\d+\]')


def strip_citations(text) -> str:
    """
    A method that removes all citations from a given text.
    :param text: the text containing the citations
    :return: a
    """
    text = apa_citation().sub(' ', text)
    text = pmc_single_citation().sub(' ', text)
    text = pmc_multiple_citations().sub(' ', text)
    return wiki_style_citation().sub(' ', text)
