#!/usr/bin/env python3
"""Purge the body of ``_get_rewards`` in ARD task environments.

The ARD framework treats ``_get_rewards`` as the sole edit target: an agent
regenerates the reward shaping from scratch. Before a run we strip the existing
implementation so only the signature and the final ``return`` remain, e.g.::

    def _get_rewards(self) -> torch.Tensor:


        return total_reward

Everything between the signature and the last ``return`` statement (including any
docstring and all reward computation) is removed.

Usage::

    python scripts/purge_rewards.py path/to/env.py [more.py ...]
    python scripts/purge_rewards.py source/ard_tasks            # recurse a dir
    python scripts/purge_rewards.py --dry-run source/ard_tasks  # preview only

By default the last ``return`` line in the method is preserved verbatim, so a
method ending in ``return total_reward`` keeps exactly that. Use
``--return-stmt`` to force a specific return statement instead.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

METHOD_NAME = "_get_rewards"


def _find_method(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the first ``_get_rewards`` function definition, or ``None``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == METHOD_NAME:
            return node
    return None


def purge_source(source: str, return_stmt: str | None = None) -> tuple[str, bool]:
    """Purge ``_get_rewards`` in *source*.

    Returns ``(new_source, changed)``. ``changed`` is ``False`` when the method
    is absent or already purged.
    """
    tree = ast.parse(source)
    method = _find_method(tree)
    if method is None:
        return source, False

    lines = source.splitlines(keepends=True)

    # The def may decorate or span multiple lines; the signature ends on the
    # first line whose stripped text ends with ':'.
    sig_start = method.lineno - 1  # ast linenos are 1-based
    sig_end = sig_start
    while sig_end < len(lines) and not lines[sig_end].rstrip().endswith(":"):
        sig_end += 1
    header = lines[sig_start : sig_end + 1]

    # Body indentation = one level deeper than the def.
    def_indent = len(lines[sig_start]) - len(lines[sig_start].lstrip())
    body_indent = " " * (def_indent + 4)

    # Determine the return statement to keep.
    if return_stmt is not None:
        ret_line = f"{body_indent}{return_stmt.strip()}\n"
    else:
        # ast.walk is breadth-first, so sort by position to get the source-last return.
        returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
        if returns:
            last = max(returns, key=lambda n: (n.lineno, n.col_offset))
            ret_text = "".join(lines[last.lineno - 1 : last.end_lineno]).strip()
            ret_line = f"{body_indent}{ret_text}\n"
        else:
            ret_line = f"{body_indent}return total_reward\n"

    # Preserve the trailing newline style of the original method block.
    method_end = method.end_lineno  # 1-based, inclusive
    new_block = header + ["\n", "\n", ret_line]
    new_lines = lines[:sig_start] + new_block + lines[method_end:]
    new_source = "".join(new_lines)

    return new_source, new_source != source


def iter_target_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: skipping non-existent path {p}", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Python files or directories to process.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument(
        "--return-stmt",
        default=None,
        help="Force this return statement (e.g. 'return total_reward') instead of preserving the original.",
    )
    args = parser.parse_args(argv)

    changed_any = False
    for path in iter_target_files(args.paths):
        try:
            source = path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        try:
            new_source, changed = purge_source(source, args.return_stmt)
        except SyntaxError as exc:
            print(f"warning: cannot parse {path}: {exc}", file=sys.stderr)
            continue

        if not changed:
            continue
        changed_any = True
        if args.dry_run:
            print(f"would purge {METHOD_NAME} in {path}")
        else:
            path.write_text(new_source)
            print(f"purged {METHOD_NAME} in {path}")

    if not changed_any:
        print("no changes made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
