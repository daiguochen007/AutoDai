import datetime
import json
import os
import time
from xml.sax import ContentHandler, parse

import pandas as pd
import yaml


def get_file_timestamp(file_path):
    """
    get file timestamp
    """
    return datetime.datetime.fromtimestamp(os.path.getmtime(file_path))


def change_file_timestamp(file_path, datetime_str):
    """
    get file timestamp
    """
    d = datetime.datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
    ftime = time.mktime(d.timetuple())
    os.utime(file_path, (ftime, ftime))


def get_abs_path(file_path):
    rel_path = os.path.join(os.path.dirname(__file__), file_path)
    abs_path = os.path.abspath(rel_path)
    return abs_path


def read_yaml(file_path):
    """
    file_path = r"C:\\Users\Dai\Desktop\investment\Git\AutoDai\DaiToolkit\login.yaml"
    """
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_yaml_rel_path(rel_path):
    abs_path = get_abs_path(rel_path)
    return read_yaml(abs_path)


def read_txt_rel_path(rel_path):
    abs_path = get_abs_path(rel_path)
    with open(abs_path, 'r') as f:
        data = f.read().replace('\n', '')
    return data


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def read_XML_fmt_xls(xmlfilepath):
    """
    xml reader
    """

    class ExcelHandler(ContentHandler):
        def __init__(self):
            self.chars = []
            self.cells = []
            self.rows = []
            self.tables = {}
            self.table_name = None
            self.cell_column = None
            self.prefix = "ss:"

        def characters(self, content):
            self.chars.append(content)

        def startElement(self, name, atts):
            if name == "Data":
                self.chars = []
            elif name == "Row":
                self.cells = []
            elif name == "Table":
                self.rows = []
            elif name == "Worksheet":
                self.table_name = atts.getValue(self.prefix + "Name")
            elif name == "Cell":
                self.cell_column = int(atts.get(self.prefix + "Index", len(self.cells) + 1))

        def endElement(self, name):
            if name == "Data":
                for i in range(self.cell_column - len(self.cells) - 1):
                    self.cells.append('')
                v = ''.join(self.chars)
                self.cells.append(v)
            elif name == "Row":
                self.rows.append(self.cells)
            elif name == "Table":
                self.tables[self.table_name] = self.rows

    excelHandler = ExcelHandler()
    parse(xmlfilepath, excelHandler)
    res = excelHandler.tables
    for k in list(res.keys()):
        res[k] = pd.DataFrame(res[k][1:], columns=res[k][0])
    return res
