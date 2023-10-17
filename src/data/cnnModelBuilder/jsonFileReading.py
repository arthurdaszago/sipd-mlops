import json

class JSONFileReading:
    def __init__(self):
        self.json_file = None
        self.content = None

    def importJSONfile(self, json_file):
        self.json_file = json_file

        try:
          f = open(self.json_file)
        except FileNotFoundError as e:
            raise e

        try:
            self.content = json.load(f)
        except Exception as e:
            raise e
        finally:
            f.close()

        return self.content