from src.bioengine.dochandlers.page_objects.wikipedia_page_object import WikipediaPageObject

name = 'Insulin_resistance'

wiki_page = WikipediaPageObject(name=name)


def write_to_file(file_path: str, contents: list):
    with open(file_path, 'w') as file_handler:
        for entry in contents:
            file_handler.write(entry)


write_to_file(f'{name}.txt', wiki_page.get_text())
