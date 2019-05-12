from urllib.error import HTTPError
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

from src.bioengine.dochandlers.page_objects.page_object import PageObject
from settings import Config
from utils.citation_utils import strip_citations


class SpringerPageObject(PageObject):
    """
    A page object abstracting the PMC website.
    """

    @staticmethod
    def remove_js_warning(txt_paragraph: list) -> list:
        """

        :param txt_paragraph:
        :return:
        """
        return list(filter(lambda x: 'requires JavaScript' not in x, txt_paragraph))

    @staticmethod
    def open_page(page_name):
        """
        A helper function that opens a springer webpage.
        :return: a string containing the raw html
        """
        req = Request(page_name, headers=Config().get_property('headers'))
        return urlopen(req).read()

    def __init__(self, pmid: str, link: str):
        # self.page_name = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/'
        self.id = pmid
        self.link = link
        self.soup = self.get_page()
        if self.soup is not None:
            self.pipeline = Config().get_property("oup_pipeline")
            self.soup = self.clean_tags()
            self.text = [strip_citations(paragraph)
                         for paragraph in self.get_text()]
            self.text = self.remove_js_warning(self.text)

    def get_article_details(self) -> tuple:
        """
        A helper function that returns the author details
        :return: a tuple containing the title and a list of authors
        """
        top_widget = self.soup.find('div', attrs={'class': 'ArticleHeader main-context'})
        name = top_widget.find('h1', attrs={'class': 'ArticleTitle'}).text
        authors = [author.text for author in top_widget.findAll('li', attrs='authors__name')]
        return name, authors

    def get_abstract(self) -> str:
        """
        A function that returns the abstract from the text
        :return: a string representing the abstract
        """
        return ' '.join([item.text for item in self.soup.find('section', attrs={'class': 'Abstract'}).findAll('p')])

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

    def get_text(self) -> list:
        """
        A method that returns a list of paragraphs in the article.
        :return: a list of paragraphs
        """
        paragraphs = self.soup.find('div', attrs={'id': 'body'}).findAll('p')
        return [paragraph.text for paragraph in paragraphs]