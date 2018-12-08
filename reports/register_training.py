import json
import logging
import os
import traceback

import uuid
import requests

from reports.spreadsheet import Spreadsheet


def register_training(checkpoint_dir, neptune_context=None, settings=None, representation=None):
    print("Register training")
    uid = str(uuid.uuid4())

    # Google Compute Engine?
    try:
        r = requests.get("http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
                         headers={'Metadata-Flavor': 'Google'})
    except requests.exceptions.ConnectionError:
        host = "not supported"
    else:
        if r.status_code == 200:
            host = r.text
        else:
            host = "not supported"
    path = os.path.abspath(checkpoint_dir)
    if neptune_context is not None:
        try:
            neptune_url = neptune_context.experiment_id
        except AttributeError:  # neptune 1?
            neptune_url = ""
    else:
        neptune_url = ""

    items = [uid, host, path, "", "", neptune_url,
             representation if representation is not None else "", "", str(settings)]
    try:
        print("Add to spreadsheet {}".format(items))
        spreadsheet = Spreadsheet()
        spreadsheet.append_row(items)
    except Exception:
        print("Failed to access training list spreadsheet.")
        logging.error(traceback.format_exc())
    else:
        print("OK, added")
