"""Create a physical MugTree variant with taller visual and collision geometry."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z-scale", required=True, type=float)
    args = parser.parse_args()
    if args.z_scale <= 1.0:
        raise ValueError("z-scale must be greater than one")

    os.makedirs(args.output, exist_ok=True)
    source_usd = next(
        os.path.join(args.source, name)
        for name in os.listdir(args.source)
        if name.endswith(".usd") and "configuration" not in name
    )
    output_usd = os.path.join(args.output, "mug_tree_taller.usd")
    with open(
        os.path.join(args.source, "asset_size.json"), encoding="utf-8"
    ) as stream:
        size = json.load(stream)
    size["size"]["z"] *= args.z_scale
    with open(
        os.path.join(args.output, "asset_size.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(size, stream, indent=2, sort_keys=True)

    with open(output_usd, "w", encoding="utf-8") as stream:
        stream.write(
            f"""#usda 1.0
(
    defaultPrim = "MugTreeTaller"
)

def Xform "MugTreeTaller" (
    prepend references = @{source_usd}@</MugTree_000>
)
{{
    double3 xformOp:scale = (1, 1, {args.z_scale})
    uniform token[] xformOpOrder = ["xformOp:scale"]
}}
"""
        )
    print(
        "TALLER_MUGTREE_ASSET="
        + json.dumps(
            {
                "output": output_usd,
                "source": source_usd,
                "z_scale": args.z_scale,
                "size": size["size"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
