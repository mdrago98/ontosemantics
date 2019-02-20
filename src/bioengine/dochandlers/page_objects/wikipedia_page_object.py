from urllib import request
from bs4 import BeautifulSoup

from src.bioengine.dochandlers.page_objects.page_object import PageObject
from settings import Config

import re


class WikipediaPageObject(PageObject):

    def __init__(self, name, lang: str = 'en'):
        self.page_name = f'https://{lang}.wikipedia.org/wiki/{name}?printable=yes'
        self.raw_html = request.urlopen(self.page_name).read()
        self.pipeline = Config().get_config("wikipedia_pipeline")
        self.soup = BeautifulSoup(self.raw_html, 'lxml')
        self.delete_after('References')
        self.clean_wikipedia_citations()
        self.soup = self.clean_tags()

    def delete_after(self, header_name, tag='h2'):
        tags = list(filter(lambda x: x.text.lower() == header_name.lower(), self.soup.findAll(tag)))
        for tag in tags:
            for sibling in tag.next_elements:
                sibling.decompose()
            tag.decompose()

    def clean_wikipedia_citations(self):
        citation_regex = re.compile("\[\d+\]")
        tags = self.soup.findAll('a')
        citations = filter(lambda x: citation_regex.match(x.text), tags)
        for citation in citations:
            citation.decompose()

    def get_text(self):
        paragraphs = self.soup.findAll('p')
        return [paragraph.text for paragraph in paragraphs]

