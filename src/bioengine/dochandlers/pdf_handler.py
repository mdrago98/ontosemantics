import io

from PyPDF2 import PdfFileReader
from pyocr import pyocr, builders
from PIL import Image as Pil
from wand.image import Image
from pkg_resources import resource_string as resource
from settings import Config


def rip_text(file_name: str) -> list:
    """
    A helper function that rips out text from a pdf file
    :param file_name: string representing the file_name
    :return: a list of strings representing the content
    """
    file_path = '/'.join((Config().get_config('resources'), file_name))
    with open(file_path, 'rb') as pdf_file_object:
        pdf_file_reader = PdfFileReader(pdf_file_object)
        return [pdf_file_reader.getPage(i).extractText() for i in range(0, pdf_file_reader.numPages)]


def read_pdf(file_name: str) -> list:
    """
    A method for reading a pdf through an ocr.
    :param file_name: string representing the file_name
    :return: a list of strings representing the content
    """
    tool = pyocr.get_available_tools()[0]
    lang = tool.get_available_languages()[0]
    file_path = '/'.join((Config().get_config('resources'), file_name))
    with Image(filename=file_path, resolution=500) as img_pdf:
        image_jpeg = img_pdf.convert('jpeg')
        image_blobs = [Image(image=img).make_blob('jpeg') for img in image_jpeg.sequence]
        return [tool.image_to_string(Pil.open(io.BytesIO(img)),
                                     lang=lang, builder=builders.TextBuilder()) for img in image_blobs]


def write_to_file(file_name: str, contents: list):
    file_path = '/'.join(Config().get_config('resource_dir'))
    with open(resource(file_path, file_name), 'w') as file_handler:
        for entry in contents:
            file_handler.write(entry)
