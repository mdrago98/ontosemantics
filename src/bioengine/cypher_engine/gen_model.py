
import plac
from json import loads
from os.path import join
from pystache import render
from utils.pythonic_name import get_pythonic_name

ols_properties = ['id', 'iri', 'olsId', 'short_form', 'obo_id', 'ontology_name', 'has_children', 'ontology_prefix',
                  'description', 'label', 'is_defining_ontology', 'is_root', 'is_obsolete', 'ontology_iri',
                  'superClassDescription', 'equivalentClassDescription']


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
