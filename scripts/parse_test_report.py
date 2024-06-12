# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json", help="Directory containing reports")
    args = parser.parse_args()

    print(args.report_json)
    with open(args.report_json) as f:
        report = json.load(f)

    for test in report["tests"]:
        print(test["nodeid"], test["outcome"])

    # Print summary in readable format
    print("Summary:")
    print(f'- {report["summary"]["passed"]} passed')
    print(f'- {report["summary"]["failed"]} failed')
    print(f'- {report["summary"]["skipped"]} deselected')
    print(f'- {report["summary"]["total"]} total')
