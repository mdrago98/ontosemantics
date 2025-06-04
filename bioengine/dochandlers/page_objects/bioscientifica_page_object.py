from urllib.error import HTTPError
from bs4 import BeautifulSoup

from src.bioengine.dochandlers.page_objects.page_object import PageObject
from utils.citation_utils import strip_citations


class BioScientificaPageObject(PageObject):
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

    def __init__(self, pmid: str, link: str, local: bool = False):
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
            self.text = [strip_citations(paragraph)
                         for paragraph in self.get_text()]
            self.text = self.remove_js_warning(self.text)

    def get_article_details(self) -> tuple:
        """
        A helper function that returns the author details
        :return: a tuple containing the title and a list of authors
        """
        top_widget = self.soup.find('div', attrs={'class': 'widget-ArticleTopInfo'})
        name = self.soup.find('h1', attrs={'class': 'title text-headline mb-3'}).text
        authors = [author.text for author in
                   self.soup.find('div', attrs={'class': 'contributor-line text-subheading'}).findAll('a', attrs='c-Button--link c-Button--primary')]
        return name, authors

    def get_abstract(self) -> str:
        """
        A function that returns the abstract from the text
        :return: a string representing the abstract
        """
        return self.soup.find('section', attrs={'class': 'abstract'}).find('p').text

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

    def get_text(self):
        """
        A method that returns a list of paragraphs in the article.
        :return: a list of paragraphs
        """
        paragraphs = self.soup.find('div', attrs={'id': 'articleBody'}).findAll('p')
        return [paragraph.text for paragraph in paragraphs]
