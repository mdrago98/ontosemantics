from urllib.error import HTTPError
from bs4 import BeautifulSoup

from bioengine.dochandlers.page_objects.page_object import PageObject
from settings import Config
from utils.citation_utils import strip_citations


class PMCPageObject(PageObject):
    """
    A page object abstracting the PMC website.
    """

    @staticmethod
    def remove_js_warning(txt_paragraph: list) -> list:
        """

        :param txt_paragraph:
        :return:
        """
        return list(filter(lambda x: 'web site requires JavaScript' not in x, txt_paragraph))

    def __init__(self, pmid: str, link: str, local=False):
        # self.page_name = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/'
        self.id = pmid
        self.link = link
        self.local = local
        if not local:
            self.soup = self.get_page()
        else:
            with open(link) as file:
                self.soup = BeautifulSoup(file.read())
        if self.soup is not None:
            self.pipeline = Config().get_property("pmc_pipeline")
            # self.soup = self.clean_tags()
            self.text = [strip_citations(paragraph)
                         for paragraph in self.get_text()]
            self.text = self.remove_js_warning(self.text)

    def get_page(self):
        """
        A helper method for opening a pmc webpage and returns a parsed bs4 object
        :return:
        """
        try:
            result = BeautifulSoup(self.open_page(self.link), 'lxml')
        except HTTPError as error:
            result = None
        return result

    def get_article_details(self) -> tuple:
        """
        A helper function that returns the author details
        :return: a tuple containing the title and a list of authors
        """
        name = self.soup.find('h1', attrs={'class': 'content-title'}).text
        authors = [author.text for author in
                   self.soup.find('div', attrs='contrib-group fm-author').findAll('a', attrs='affpopup')]
        return name, authors

    def get_abstract(self) -> str:
        """
        A function that returns the abstract from the text
        :return: a string representing the abstract
        """
        return ''.join(self.text[0:3])

    def get_text(self):
        """
        A method that returns a list of paragraphs in the article.
        :return: a list of paragraphs
        """
        paragraphs = self.soup.findAll('p')
        return [paragraph.text for paragraph in paragraphs]
