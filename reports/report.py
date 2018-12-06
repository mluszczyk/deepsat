import os

import json
import nbformat
import nbconvert
import shutil
import subprocess
import uuid

from reports import spreadsheet
from google.cloud import storage

OUTPUT_DIR = "/tmp/outputs"

FIRST_CELL_TEMPLATE = """META_DIR = "{meta_dir}"
NOTEBOOK_PARAMS = '''{notebook_params}'''
"""

NOTEBOOK_AS_VERSION = 4
BUCKET_NAME = "neural-guidance-report-html"
EXECUTE_NOTEBOOKS = True
STORAGE_URL = "https://storage.googleapis.com/neural-guidance-report-html/"


def make_cell_text(checkpoint_dir, notebook_params):
    assert "'''" not in notebook_params
    assert "\\" not in notebook_params
    return FIRST_CELL_TEMPLATE.format(
        meta_dir=checkpoint_dir, notebook_params=json.dumps(notebook_params, indent=2))


def make_report(notebook_template, notebook_params, checkpoint_dir, output_file, execute_timeout=3600):
    notebook = nbformat.read(notebook_template, NOTEBOOK_AS_VERSION)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(make_cell_text(checkpoint_dir, notebook_params)))
    if EXECUTE_NOTEBOOKS:
        ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=execute_timeout, kernel_name='python3')
        ep.preprocess(notebook, {'metadata': {'path': 'notebooks/'}})
    html_exporter = nbconvert.HTMLExporter()
    (body, _) = html_exporter.from_notebook_node(notebook)
    with open(output_file, "w") as f:
        f.write(body)


def fetch_checkpoint(host, path, checkpoint_dir):
    from_path = "{}:{}/*".format(host, path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    subprocess.check_call(["rsync", "-e", "ssh -o StrictHostKeyChecking=no", "-avx", from_path, checkpoint_dir])


class SpreadsheetRegistry:
    def __init__(self):
        pass

    def list_trainings(self):
        sheet = spreadsheet.Spreadsheet()
        data = sheet.read_all()
        for row in data[1:]:
            yield Training(row[1], row[2], row[0], self)


class Training:
    def __init__(self, host, path, uid, registry):
        self.host = host
        self.path = path
        self.uid = uid

    def lookup_row(self):
        sheet = spreadsheet.Spreadsheet()
        data = sheet.read_all()
        for num, row in enumerate(data):
            if row[0] == self.uid:
                return num
        raise Exception("uid not found")

    def set_status(self, status):
        print("set status {}: {}".format(self.uid, status))
        spreadsheet.Spreadsheet().set_cell(self.lookup_row(), 4, status)

    def set_report_url(self, report_url):
        print("set report url {}: {}".format(self.uid, report_url))
        spreadsheet.Spreadsheet().set_cell(self.lookup_row(), 5, report_url)


def upload_report(path, report_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(report_name)
    blob.upload_from_filename(filename=path)
    return report_name


def main():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=False)

    registry = SpreadsheetRegistry()

    for training in registry.list_trainings():

        print("Fetch")
        training_dir = os.path.join(OUTPUT_DIR, training.uid)
        checkpoint_dir = os.path.join(training_dir, "checkpoint")
        try:
            fetch_checkpoint(training.host, training.path, checkpoint_dir)
        except Exception:
            training.set_status("problem with fetching")
            continue

        print("Make report {}".format(training.uid))
        report_file_name = "{}-{}.html".format(training.uid, uuid.uuid4())

        params = {
            "input_method": {
                "type": "sequence",
                "variable_num": 5,
            }
        }

        report_file = os.path.join(training_dir, report_file_name)

        try:
            make_report(os.path.join("notebooks", "templates", "dpll sequential rf.ipynb"),
                        params,
                        checkpoint_dir=checkpoint_dir,
                        output_file=report_file)
        except Exception:
            training.set_status("problem with making report")
            continue

        try:
            report_url = upload_report(report_file, report_file_name)
        except Exception:
            training.set_status("problem with uploading report")
            continue

        training.set_report_url(STORAGE_URL + report_url)
        training.set_status("OK")


if __name__ == '__main__':
    main()
