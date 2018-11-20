from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools, clientsecrets

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1Qh6ewiZLtbwm1wutvbAYtRKhuF6d01t9c5Q7cxhgbj4'
SAMPLE_RANGE_NAME = 'Arkusz1!A1:Z10'
SAMPLE_VALUE_INPUT_OPTION = 'RAW'


class SpreadsheetException(Exception):
    pass


def make_range(sheet, start, end):
    return "{}!{}{}:{}{}".format(sheet,
                                 chr(ord('A') + start[1] - 1), start[0],
                                 chr(ord('A') + end[1] - 1), end[0])


class Spreadsheet:
    def __init__(self):
        pass

    def get_sheet(self):
        store = file.Storage('token.json')
        creds = store.get()
        if not creds or creds.invalid:
            try:
                flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
            except clientsecrets.InvalidClientSecretsError as e:
                raise SpreadsheetException("invalid client secrets") from e
            creds = tools.run_flow(flow, store)
        service = build('sheets', 'v4', http=creds.authorize(Http()))

        # Call the Sheets API
        return service.spreadsheets()

    def read_all(self):
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
        sheet = self.get_sheet()
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                    range=SAMPLE_RANGE_NAME).execute()
        values = result.get('values', [])

        return values

    def append_row(self, items):
        sheet = self.get_sheet()
        sheet.values().append(
            spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=SAMPLE_RANGE_NAME,
            valueInputOption=SAMPLE_VALUE_INPUT_OPTION,
            body={'values': [items]}
        ).execute()

    def set_cell(self, row_num, col_num, val):
        sheet = self.get_sheet()
        result = sheet.values().update(
            spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=make_range("Arkusz1", (row_num + 1, col_num), (row_num + 1, col_num)),
            valueInputOption=SAMPLE_VALUE_INPUT_OPTION,
            body={'values': [[val]]}).execute()


if __name__ == '__main__':
    s = Spreadsheet()
    s.read_all()
    s.append_row(["1", "2", "3", "4"])
