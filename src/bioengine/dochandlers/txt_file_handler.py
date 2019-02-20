
def write_to_file(file_path: str, contents: list):
    with open(file_path, 'w') as file_handler:
        for entry in contents:
            file_handler.write(entry)


def read_to_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
    return content
