from bs4 import BeautifulSoup


class PageObject:
    def clean_tags(self, soup: BeautifulSoup = None, pipeline: dict = None) -> BeautifulSoup:
        """
        A helper function that given a pipeline cleans elements found on the page.
        :param soup: A Beautiful object representing an html file
        :param pipeline: A dictionary representing elements to clean
        :return:
        """
        if soup is None:
            soup = self.soup
        if pipeline is None:
            pipeline = self.pipeline
        decompose = []
        for tag, attributes in pipeline.items():
            if attributes is not None:
                for attribute in attributes.items():
                    for value in attribute[1]:
                        decompose += soup.findAll(tag, {attribute[0]: value})
            else:
                decompose += soup.findAll(tag)
        for item in decompose:
            item.decompose()
        return soup
