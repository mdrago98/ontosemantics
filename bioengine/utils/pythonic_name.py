from re import sub, compile


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