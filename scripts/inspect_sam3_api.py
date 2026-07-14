"""Print the public API surface of the installed sam3 package."""

import inspect

import sam3
from sam3 import model_builder


def main() -> None:
    print("sam3 package:", sam3.__file__)
    print("\npublic names in sam3:")
    for name in sorted(n for n in dir(sam3) if not n.startswith("_")):
        print("  ", name)

    print("\nbuilders in sam3.model_builder:")
    for name, obj in inspect.getmembers(model_builder, inspect.isfunction):
        if name.startswith("_"):
            continue
        print(f"\n  {name}{inspect.signature(obj)}")
        doc = inspect.getdoc(obj)
        if doc:
            print("    ", doc.splitlines()[0])


if __name__ == "__main__":
    main()