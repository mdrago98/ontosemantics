from bioengine.dochandlers.page_objects.page_object import PageObject
from settings import Config
from utils.citation_utils import strip_citations


class NaturePageObject(PageObject):
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

    def __init__(self, pmid: str, link: str):
        # self.page_name = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/'
        self.id = pmid
        self.link = link
        self.soup = self.get_page()
        if self.soup is not None:
            self.pipeline = Config().get_property("nature_pipeline")
            self.text = [strip_citations(paragraph)
                         for paragraph in self.get_text()]
            self.text = self.remove_js_warning(self.text)

    def get_article_details(self) -> tuple:
        """
        A helper function that returns the author details
        :return: a tuple containing the title and a list of authors
        """
        name = self.soup.find('h1', attrs={'class': 'tighten-line-height small-space-below'}).text
        authors = [author.text for author in self.soup.findAll('li', attrs={'itemprop':'author'})]
        return name, authors

    def get_abstract(self) -> str:
        """
        A function that returns the abstract from the text
        :return: a string representing the abstract
        """
        return ''

    def get_text(self):
        """
        A method that returns a list of paragraphs in the article.
        :return: a list of paragraphs
        """
        paragraphs = self.soup.find('div', attrs={'class': 'article-body clear'}).findAll('p')
        return [paragraph.text for paragraph in paragraphs]
