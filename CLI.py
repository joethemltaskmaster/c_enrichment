"""Command-line interface for testing sheets_client.SheetsClient.

Examples (run from the folder containing sheets_client.py):

  python CLI.py read "Sheet1!A1:C10"
  python CLI.py batch-read "Sheet1!A1:B5" "Sheet1!D1:E5"
  python CLI.py records "Creators!A1:F"
  python CLI.py update "Sheet1!A1:B2" --row "Header 1,Header 2" --row "Value 1,Value 2"
  python CLI.py append "Creators!A:C" --row "Jane,jane@example.com,Sent"
  python CLI.py batch-update --data-file updates.json
  python CLI.py clear "Sheet1!A1:B2"

Row/values input (for update and append), pick one:
  --row "a,b,c"          repeat for each row; cells split on commas
  --values '[["a","b"]]' JSON list of rows
  --values-file f.json   JSON file containing a list of rows

Connection settings (flag > environment variable > sheets_client default):
  --spreadsheet-id / SHEETS_SPREADSHEET_ID
  --key-file       / SHEETS_KEY_FILE
"""

import argparse
import csv
import io
import json
import os
import sys

from googleapiclient.errors import HttpError

from services.sheets_client import SERVICE_ACCOUNT_FILE, SPREADSHEET_ID, SheetsClient


def _parse_row(text):
    """Split 'a,b,"c, d"' into cells using CSV rules."""
    return next(csv.reader(io.StringIO(text)))


def _load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_values(args):
    if args.row:
        return [_parse_row(r) for r in args.row]
    if args.values:
        return json.loads(args.values)
    if args.values_file:
        return _load_json_file(args.values_file)
    sys.exit("Error: provide --row, --values, or --values-file.")


def _get_batch_data(args):
    if args.data:
        return json.loads(args.data)
    if args.data_file:
        return _load_json_file(args.data_file)
    sys.exit("Error: provide --data or --data-file.")


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _add_values_options(p):
    g = p.add_argument_group("values (choose one)")
    g.add_argument("--row", action="append", help='One row, comma-separated. Repeatable.')
    g.add_argument("--values", help="JSON list of rows.")
    g.add_argument("--values-file", help="Path to a JSON file with a list of rows.")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Use RAW input instead of USER_ENTERED (no formula/number parsing).",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test Google Sheets operations from the terminal."
    )
    parser.add_argument(
        "--spreadsheet-id",
        default=os.environ.get("SHEETS_SPREADSHEET_ID", SPREADSHEET_ID),
    )
    parser.add_argument(
        "--key-file",
        default=os.environ.get("SHEETS_KEY_FILE", SERVICE_ACCOUNT_FILE),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("read", help="Read a single range.")
    p.add_argument("range")

    p = sub.add_parser("batch-read", help="Read multiple ranges in one call.")
    p.add_argument("ranges", nargs="+")

    p = sub.add_parser("records", help="Read a range with a header row as dicts.")
    p.add_argument("range")

    p = sub.add_parser("update", help="Overwrite a single range.")
    p.add_argument("range")
    _add_values_options(p)

    p = sub.add_parser("batch-update", help="Overwrite multiple ranges in one call.")
    p.add_argument("--data", help='JSON object: {"Sheet1!A1:B1": [["x","y"]], ...}')
    p.add_argument("--data-file", help="Path to a JSON file with the same shape.")
    p.add_argument("--raw", action="store_true")

    p = sub.add_parser("append", help="Append rows below existing data.")
    p.add_argument("range")
    _add_values_options(p)

    p = sub.add_parser("clear", help="Clear values in a range.")
    p.add_argument("range")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.spreadsheet_id == "YOUR_SPREADSHEET_ID_HERE":
        sys.exit(
            "Error: set a real spreadsheet ID via --spreadsheet-id, "
            "SHEETS_SPREADSHEET_ID, or SPREADSHEET_ID in sheets_client.py."
        )

    try:
        client = SheetsClient(
            spreadsheet_id=args.spreadsheet_id,
            service_account_file=args.key_file,
        )
        option = "RAW" if getattr(args, "raw", False) else "USER_ENTERED"

        if args.command == "read":
            _print(client.read_range(args.range))
        elif args.command == "batch-read":
            _print(client.batch_read(args.ranges))
        elif args.command == "records":
            _print(client.read_records(args.range))
        elif args.command == "update":
            _print(client.update_range(args.range, _get_values(args), option))
        elif args.command == "batch-update":
            _print(client.batch_update(_get_batch_data(args), option))
        elif args.command == "append":
            _print(client.append_rows(args.range, _get_values(args), option))
        elif args.command == "clear":
            _print(client.clear_range(args.range))

    except FileNotFoundError as e:
        sys.exit(f"File not found: {e.filename}")
    except json.JSONDecodeError as e:
        sys.exit(f"Invalid JSON: {e}")
    except HttpError as e:
        sys.exit(
            f"Google API error {e.resp.status}: {e._get_reason()}\n"
            "If this is 403/404, share the sheet with the service account's "
            "client_email and check the spreadsheet ID and tab name."
        )


if __name__ == "__main__":
    main()
