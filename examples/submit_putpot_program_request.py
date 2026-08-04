"""Submit one program revision or shutdown to a waiting PutPot worker."""

from __future__ import annotations

import argparse
import json

from judo_isaaclab.putpot_queue import submit_program_request, submit_shutdown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-json", required=True)
    parser.add_argument("--program-spec-json")
    parser.add_argument("--ambiguity-reason")
    parser.add_argument("--shutdown-reason")
    args = parser.parse_args(argv)
    if bool(args.program_spec_json) == bool(args.shutdown_reason):
        raise ValueError(
            "provide exactly one of --program-spec-json or --shutdown-reason"
        )
    if args.program_spec_json:
        value = submit_program_request(
            args.session_json,
            args.program_spec_json,
            ambiguity_reason=args.ambiguity_reason,
        )
    else:
        value = submit_shutdown(args.session_json, reason=args.shutdown_reason)
    print("PUTPOT_QUEUE_SUBMITTED=" + json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
