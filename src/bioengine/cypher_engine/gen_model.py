
import plac
from json import loads
from os.path import join
from re import sub, compile
from pystache import render

ols_properties = ['id', 'iri', 'olsId', 'short_form', 'obo_id', 'ontology_name', 'has_children', 'ontology_prefix',
                  'description', 'label', 'is_defining_ontology', 'is_root', 'is_obsolete', 'ontology_iri',
                  'superClassDescription', 'equivalentClassDescription']


def convert_to_snake_case_from_camelcase(name: str):
    """
    A method that converts a camel case name (partPart) into snake case (part-part)
    :param name: the variable name to convert
    :return:
    """
    s1 = sub('([a-zA-Z])([A-Z][a-z]+)', r'\1_\2', name)
    return sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def convert_to_snake_from_kebab_case(name: str) -> str:
    """
    A method that converts kebab case (name-name) to snek case (name_name)
    :param name: the variable name to convert
    :return: the converted variable name
    """
    kebab_regex = compile(r'(.*?)(-)([a-zA-Z\d_]*)')
    matches = kebab_regex.match(name)
    if '-' in name:
        name = f'{matches.group(1)}_{matches.group(3)}'
    return name


def convert_whitespace(name: str) -> str:
    """
    A function that converts whitespace to a dash
    :param name: the variable to convert
    :return: the converted name
    """
    return sub(r'\s+', '_', name)


def get_pythonic_name(name) -> str:
    """
    A method that converts variable names to one that conforms with python's conventions
    :param name: the variable name
    :return: the converted variable name
    """
    return convert_to_snake_from_kebab_case(
        convert_to_snake_case_from_camelcase(
            convert_whitespace(name)
        )
    )


def open_json_schema(path: str) -> dict:
    """
    A method that opens and reads a json schema file
    :param path:
    :return:
    """
    with open(path) as json_file:
        return loads(json_file.read())


def main(path, objects_to_map, neo_schema_path):
    """
    A method for generating the ogm representation of entities in the ontology store
    :param path: the output path where to write the outputted objects
    :param objects_to_map: a list of entities to map
    :param neo_schema_path: a path to a neo4j json schema
    """
    objects_to_map = objects_to_map.split(',')
    schema = open_json_schema(neo_schema_path)
    for graph_obj in objects_to_map:
        graph_properties = [entry for entry in schema[graph_obj.upper()]['properties'].keys()
                            if entry not in ols_properties]
        pythonic_variable_names = [get_pythonic_name(entry) for entry in graph_properties]
        context = {
            'className': graph_obj.upper(),
            'properties': [{'name': name, 'property': graph_properties[index]} for index, name in
                           enumerate(pythonic_variable_names)]
        }
        file_name = f'{graph_obj.lower()}_graph_object.py'
        with open('./templates/graph_obj_template.txt') as file:
            template = file.read()
            with open(join(path, file_name), 'w') as output:
                out_model = render(template, context)
                output.write(out_model)


if __name__ == '__main__':
    plac.call(main)
