"""Reusable Google Sheets helper built on a service account.

Setup:
    pip install google-api-python-client google-auth python-dotenv
    Share the target spreadsheet with the service account's client_email
    (found inside the JSON key file) and give it Editor access.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

load_dotenv(r"C:\Users\Joseph\Desktop\Creator_enrichment_pipeline\.env")  # loads values from a .env file in the working directory

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = os.getenv("SHEETS_KEY_FILE", "")
SPREADSHEET_ID = os.getenv("SHEETS_SPREADSHEET_ID", "")
CREATOR_SHEET = os.getenv("SHEET_NAME", "")


class SheetsClient:
    def __init__(
        self,
        spreadsheet_id: str = SPREADSHEET_ID,
        service_account_file: str = SERVICE_ACCOUNT_FILE,
        scopes: Optional[List[str]] = None,
    ):
        creds = Credentials.from_service_account_file(
            service_account_file, scopes=scopes or SCOPES
        )
        self.service = build("sheets", "v4", credentials=creds)
        self.spreadsheet_id = spreadsheet_id
        self._values = self.service.spreadsheets().values()

    # ---------- Read ----------
    def read_range(self, range_name: str) -> List[List[Any]]:
        """Read one range, e.g. 'Sheet1!A1:C10'. Returns a list of rows."""
        result = self._values.get(
            spreadsheetId=self.spreadsheet_id, range=range_name
        ).execute()
        return result.get("values", [])

    def batch_read(self, ranges: List[str]) -> Dict[str, List[List[Any]]]:
        """Read several ranges in one API call. Returns {range: rows}."""
        result = self._values.batchGet(
            spreadsheetId=self.spreadsheet_id, ranges=ranges
        ).execute()
        return {
            vr.get("range", ""): vr.get("values", [])
            for vr in result.get("valueRanges", [])
        }

    def read_records(self, range_name: str) -> List[Dict[str, Any]]:
        """Read a range whose first row is a header; return a list of dicts."""
        rows = self.read_range(range_name)
        if not rows:
            return []
        header, *data = rows
        return [
            {h: (row[i] if i < len(row) else "") for i, h in enumerate(header)}
            for row in data
        ]

    # ---------- Write ----------
    def update_range(
        self,
        range_name: str,
        values: List[List[Any]],
        input_option: str = "USER_ENTERED",
    ) -> Dict[str, Any]:
        """Overwrite one range with `values`."""
        return self._values.update(
            spreadsheetId=self.spreadsheet_id,
            range=range_name,
            valueInputOption=input_option,
            body={"values": values},
        ).execute()

    def batch_update(
        self,
        updates: Dict[str, List[List[Any]]],
        input_option: str = "USER_ENTERED",
    ) -> Dict[str, Any]:
        """Overwrite several ranges in one call. `updates` is {range: values}."""
        body = {
            "valueInputOption": input_option,
            "data": [{"range": r, "values": v} for r, v in updates.items()],
        }
        return self._values.batchUpdate(
            spreadsheetId=self.spreadsheet_id, body=body
        ).execute()

    def append_rows(
        self,
        range_name: str,
        rows: List[List[Any]],
        input_option: str = "USER_ENTERED",
    ) -> Dict[str, Any]:
        """Append rows after the last row of the table found in `range_name`."""
        return self._values.append(
            spreadsheetId=self.spreadsheet_id,
            range=range_name,
            valueInputOption=input_option,
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        ).execute()

    def clear_range(self, range_name: str) -> Dict[str, Any]:
        """Clear values in a range (formatting is kept)."""
        return self._values.clear(
            spreadsheetId=self.spreadsheet_id, range=range_name, body={}
        ).execute()


if __name__ == "__main__":
    client = SheetsClient()

    print("Single Range:", client.read_range("Sheet1!A1:C10"))
    print("Batch Read:", client.batch_read(["Sheet1!A1:B5", "Sheet1!D1:E5"]))

    client.update_range("Sheet1!A1:B2", [["Header 1", "Header 2"], ["Value 1", "Value 2"]])
    client.batch_update(
        {
            "Sheet1!A5:B5": [["Updated A5", "Updated B5"]],
            "Sheet1!D5:E5": [["Updated D5", "Updated E5"]],
        }
    )
    print("Workflows completed successfully.")