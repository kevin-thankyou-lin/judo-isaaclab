"""Build an immutable PutPot source-demo strategy card from source keyframes."""

from __future__ import annotations

import argparse
import json

from judo_isaaclab.putpot_repair_policy import write_source_demo_card


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-keyframes-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    receipt = write_source_demo_card(
        args.source_keyframes_json, args.output_json
    )
    print("PUTPOT_SOURCE_DEMO_CARD=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
