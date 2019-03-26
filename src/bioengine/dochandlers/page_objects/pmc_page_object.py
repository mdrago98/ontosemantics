from urllib.error import HTTPError
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

from src.bioengine.dochandlers.page_objects.page_object import PageObject
from settings import Config

from src.bioengine.dochandlers.utils import strip_citations


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

    @staticmethod
    def open_page(page_name):
        """
        A helper function that opens a pmc webpage.
        :return: a string containing the raw html
        """
        req = Request(page_name, headers=Config().get_property('headers'))
        return urlopen(req).read()

    def __init__(self, pmc_id: str):
        # self.page_name = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/'
        self.page_name = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/'
        self.soup = self.get_page()
        if self.soup is not None:
            self.pipeline = Config().get_property("pmc_pipeline")
            self.soup = self.clean_tags()
            self.text = [strip_citations(paragraph)
                         for paragraph in self.get_text()]
            self.text = self.remove_js_warning(self.text)

    def get_page(self):
        """
        A helper method for opening a pmc webpage and returns a parsed bs4 object
        :return:
        """
        try:
            result = BeautifulSoup(self.open_page(self.page_name), 'lxml')
        except HTTPError as error:
            result = None
        return result

    def get_text(self):
        """
        A method that returns a list of paragraphs in the article.
        :return: a list of paragraphs
        """
        paragraphs = self.soup.findAll('p')
        return [paragraph.text for paragraph in paragraphs]
