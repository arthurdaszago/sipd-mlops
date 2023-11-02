import json

def open_file(path_file):
    file = open(path_file)
    data = json.load(file)
    file.close()

    return data