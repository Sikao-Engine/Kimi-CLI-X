"""Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax.

Provides the ``pwsh_transform`` function that rewrites modern PowerShell 7.x
syntax features (ternary operator, null-coalescing operators, pipeline chain
operators, and null-conditional operators) into equivalent PowerShell 5.1
constructs.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# ============================================================================
# String & Comment Region Detection
# ============================================================================


def _find_string_regions(code: str) -> List[Tuple[int, int]]:
    """Find all string and comment regions in PowerShell code.

    Returns a sorted list of ``(start, end)`` character ranges that are inside
    strings, here-strings, or comments.  These regions must be skipped during
    transformation to avoid rewriting operator-like text inside strings.
    """
    regions: List[Tuple[int, int]] = []
    i = 0
    n = len(code)

    while i < n:
        c = code[i]

        # Block comment: <# ... #>  (nestable per PowerShell spec)
        if c == "<" and i + 1 < n and code[i + 1] == "#":
            start = i
            depth = 1
            i += 2
            while i < n and depth > 0:
                if code[i] == "<" and i + 1 < n and code[i + 1] == "#":
                    depth += 1
                    i += 2
                elif code[i] == "#" and i + 1 < n and code[i + 1] == ">":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            regions.append((start, i))
            continue

        # Line comment: # to end of line  (but NOT <# which is block comment)
        if c == "#":
            # We already handled "<#" above, so this is a real line comment.
            start = i
            while i < n and code[i] != "\n":
                i += 1
            regions.append((start, i))
            continue

        # Here-string: @' ... '@  or  @" ... "@
        if c == "@" and i + 1 < n and code[i + 1] in ("'", '"'):
            start = i
            quote_char = code[i + 1]
            i += 2
            # The closing '@ must sit at the beginning of a line
            # (only whitespace is allowed before it).
            while i < n:
                if code[i] == quote_char and i + 1 < n and code[i + 1] == "@":
                    # Check it is at line start.
                    line_start = code.rfind("\n", 0, i)
                    if line_start == -1:
                        line_start = 0
                    else:
                        line_start += 1  # step past the newline
                    before = code[line_start:i]
                    if before.strip() == "":
                        i += 2
                        break
                i += 1
            regions.append((start, i))
            continue

        # Single-quoted string: '...'  ('' is an escaped literal single-quote)
        if c == "'":
            start = i
            i += 1
            while i < n:
                if code[i] == "'":
                    if i + 1 < n and code[i + 1] == "'":
                        i += 2  # skip escaped ''
                    else:
                        i += 1
                        break
                else:
                    i += 1
            regions.append((start, i))
            continue

        # Double-quoted string: "..."  (backtick escapes, nested $())
        if c == '"':
            start = i
            i += 1
            while i < n:
                ch = code[i]
                if ch == "`" and i + 1 < n:
                    i += 2  # backtick-escaped character
                elif ch == '"':
                    i += 1
                    break
                elif ch == "$" and i + 1 < n and code[i + 1] == "(":
                    i = _skip_subexpression(code, i)
                else:
                    i += 1
            regions.append((start, i))
            continue

        i += 1

    return regions


def _skip_subexpression(code: str, start: int) -> int:
    """Skip a ``$(...)`` subexpression and return the index after its closing ``)``.

    ``start`` must point at the ``$`` character of ``$(``.
    """
    assert code[start] == "$"
    i = start + 2  # skip $(
    depth = 1
    n = len(code)

    while i < n and depth > 0:
        c = code[i]

        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
        elif c == "'":
            # Single-quoted string inside subexpression
            i += 1
            while i < n:
                if code[i] == "'":
                    if i + 1 < n and code[i + 1] == "'":
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
        elif c == '"':
            # Double-quoted string inside subexpression
            i += 1
            while i < n:
                if code[i] == "`" and i + 1 < n:
                    i += 2
                elif code[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif c == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1

    return i


def _region_at(regions: List[Tuple[int, int]], pos: int) -> Tuple[int, int] | None:
    """Return the region that contains *pos*, or ``None``."""
    for lo, hi in regions:
        if lo <= pos < hi:
            return (lo, hi)
    return None


def _outside_regions(regions: List[Tuple[int, int]], pos: int) -> bool:
    """Return ``True`` if *pos* is not inside any string/comment region."""
    return _region_at(regions, pos) is None


# ============================================================================
# Helper: join continuation lines
# ============================================================================

_BACKTICK_CONTINUATION = re.compile(r"`\s*\n\s*")


def _join_continuation_lines(code: str) -> str:
    """Collapse backtick line-continuations into single logical lines."""
    return _BACKTICK_CONTINUATION.sub(" ", code)


# ============================================================================
# Transform 1: Null-coalescing assignment  (??=)
# ============================================================================

# Matches:  $var ??= <value>
# Captures: (1) variable, (2) value expression (runs to end of line / pipe)
_NCA_RE = re.compile(
    r"(\$\w+(?:\.\w+)*)\s*\?\?=\s*(.+)"
)


def _transform_null_coalescing_assign(code: str) -> str:
    """Transform ``$a ??= "default"`` → ``if ($null -eq $a) { $a = "default" }``."""
    regions = _find_string_regions(code)
    lines = code.split("\n")
    result: List[str] = []

    for line in lines:
        # Find all ??= on this line, working right-to-left to preserve indices
        matches = list(_NCA_RE.finditer(line))
        if not matches:
            result.append(line)
            continue

        # Build new line by replacing each match that is outside strings
        new_line = line
        for m in reversed(matches):
            abs_pos = sum(len(l) + 1 for l in result) + m.start()
            # We approximate: check if the match is in a region on the original line
            if not _outside_regions(regions, abs_pos):
                continue  # inside string/comment – skip
            var = m.group(1)
            value = m.group(2).rstrip()
            replacement = f"if ($null -eq {var}) {{ {var} = {value} }}"
            new_line = new_line[: m.start()] + replacement + new_line[m.end() :]

        result.append(new_line)

    return "\n".join(result)


# ============================================================================
# Transform 2: Null-coalescing  (??)
# ============================================================================

# Matches:  <expr> ?? <default>
# This is tricky because the left side can be an arbitrary expression.
# We use a character-scanning approach that finds ?? at depth 0.

def _transform_null_coalescing(code: str) -> str:
    """Transform ``$x = $a ?? "default"`` → ``$x = if ($null -ne $a) { $a } else { "default" }``."""
    # NOTE: We do NOT use _depth_at for ?? because previous transforms
    # (e.g. ??=) may have wrapped code in {braces}.  Instead we only
    # skip string/comment regions.
    return _transform_binary_op_no_depth(
        code,
        op_str="??",
        transform_fn=_nc_rewrite,
    )


def _nc_rewrite(line: str, op_pos: int) -> str | None:
    """Rewrite a single ``??`` occurrence at *op_pos* in *line*."""
    # Find the left-hand expression
    left_end = op_pos
    while left_end > 0 and line[left_end - 1] == " ":
        left_end -= 1

    # Walk backwards finding the expression before ??
    left_start = _find_expr_start(line, left_end)

    # Find the right-hand expression
    right_start = op_pos + 2
    while right_start < len(line) and line[right_start] == " ":
        right_start += 1
    right_end = _find_expr_end(line, right_start)

    left_expr = line[left_start:left_end].strip()
    right_expr = line[right_start:right_end].strip()

    if not left_expr or not right_expr:
        return None

    # Check if this is an assignment:  $var = <left> ?? <right>
    before = line[:left_start].rstrip()
    assign_match = re.match(r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$", before)
    if assign_match:
        prefix = assign_match.group(1)
        var_name = assign_match.group(2)
        replacement = (
            f"{prefix}{var_name} = "
            f"if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
        )
        suffix = line[right_end:]
        return replacement + suffix
    else:
        # Bare expression (not an assignment)
        prefix = line[:left_start]
        replacement = (
            f"{prefix}"
            f"if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
        )
        suffix = line[right_end:]
        return replacement + suffix


# ============================================================================
# Transform 3: Ternary operator  (? :)
# ============================================================================

def _transform_ternary(code: str) -> str:
    """Transform ``$x = $cond ? "a" : "b"`` → ``$x = if ($cond) { "a" } else { "b" }``."""
    regions = _find_string_regions(code)
    lines = code.split("\n")
    result: List[str] = []

    for line_idx, line in enumerate(lines):
        new_line = _transform_ternary_line(line, regions, line_idx, lines, result)
        result.append(new_line)

    return "\n".join(result)


def _transform_ternary_line(
    line: str,
    regions: List[Tuple[int, int]],
    line_idx: int,
    lines: List[str],
    result: List[str],
) -> str:
    """Transform ternary operators on a single line."""
    # Find ? at depth 0 outside strings
    abs_offset = sum(len(l) + 1 for l in result)

    pos = 0
    while pos < len(line):
        abs_pos = abs_offset + pos
        if line[pos] == "?" and _outside_regions(regions, abs_pos):
            # Check if this ? is at depth 0 and has a matching :
            if _depth_at(line, pos) == 0:
                colon_pos = _find_matching_colon(line, pos + 1)
                if colon_pos != -1:
                    # Found ternary: condition ? true_expr : false_expr
                    cond_end = pos
                    while cond_end > 0 and line[cond_end - 1] == " ":
                        cond_end -= 1
                    cond_start = _find_expr_start(line, cond_end)
                    condition = line[cond_start:cond_end].strip()

                    # True expression (between ? and :)
                    true_start = pos + 1
                    while true_start < len(line) and line[true_start] == " ":
                        true_start += 1
                    true_end = colon_pos
                    while true_end > true_start and line[true_end - 1] == " ":
                        true_end -= 1
                    true_expr = line[true_start:true_end].strip()

                    # False expression (after :)
                    false_start = colon_pos + 1
                    while false_start < len(line) and line[false_start] == " ":
                        false_start += 1
                    false_end = _find_expr_end(line, false_start)
                    false_expr = line[false_start:false_end].strip()

                    # Check if this is an assignment
                    before = line[:cond_start].rstrip()
                    assign_match = re.match(
                        r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$", before
                    )
                    if assign_match:
                        prefix = assign_match.group(1)
                        var = assign_match.group(2)
                        replacement = (
                            f"{prefix}{var} = "
                            f"if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                        )
                    else:
                        prefix = line[:cond_start]
                        replacement = (
                            f"{prefix}"
                            f"if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                        )

                    suffix = line[false_end:]
                    line = replacement + suffix
                    pos = len(replacement)  # continue after replacement
                    continue
        pos += 1

    return line


def _depth_at(line: str, pos: int) -> int:
    """Return the nesting depth at position *pos* (counting ``(``, ``[``, ``{``)."""
    depth = 0
    for i in range(pos):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
    return depth


def _find_matching_colon(line: str, start: int) -> int:
    """Find the ``:`` at depth 0 after *start* that matches the ``?``."""
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ":" and depth == 0:
            return i
    return -1


def _find_expr_start(line: str, end: int) -> int:
    """Walk backwards from *end* to find the start of an expression.

    Stops at ``=``, ``;``, ``|``, ``&``, ``,``, line start, or various brackets.
    The ``=``, ``;``, ``|``, ``&``, and ``,`` characters always act as
    delimiters regardless of bracket depth.
    """
    depth = 0
    for i in range(end - 1, -1, -1):
        c = line[i]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            depth -= 1
            if depth < 0:
                # We hit an opening bracket at the "root" — stop.
                return i + 1
        elif c in "=;|&,":
            # These delimiters break expressions at any depth.
            # But check for multi-char operators like ==, !=, etc.
            if c == "=":
                if i > 0 and line[i - 1] in "=!<>+-*/.":
                    continue  # part of a comparison/compound assignment
            return i + 1
    return 0


def _find_expr_end(line: str, start: int) -> int:
    """Walk forward from *start* to find the end of an expression.

    Stops at ``;``, ``|``, ``&``, ``,``, ``#``, or end of line.
    These delimiters always act regardless of bracket depth.
    """
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                # Closing bracket at root — include it and stop.
                return i
        elif depth >= 0 and c in ";|&,":
            return i
        elif c == "#":
            if i > 0 and line[i - 1] == "<":
                continue  # <# is block comment
            return i
    return len(line)


# ============================================================================
# Transform 4+5: Pipeline chain operators  (&& and ||)
# ============================================================================

def _transform_pipeline_chains(code: str) -> str:
    """Transform ``&&`` and ``||`` pipeline chain operators.

    Both operators are processed together on each line so that mixed chains
    like ``cmd1 && cmd2 || cmd3`` are handled correctly.  The rightmost
    operator is transformed first, producing properly nested ``if`` blocks.

    ``cmd1 && cmd2``  →  ``cmd1; if ($?) { cmd2 }``
    ``cmd1 || cmd2``  →  ``cmd1; if (-not $?) { cmd2 }``
    """
    regions = _find_string_regions(code)
    lines = code.split("\n")
    result: List[str] = []

    abs_offset = 0
    for line in lines:
        new_line = _transform_chain_ops_line(line, regions, abs_offset)
        result.append(new_line)
        abs_offset += len(line) + 1

    return "\n".join(result)


def _transform_chain_ops_line(
    line: str, regions: List[Tuple[int, int]], line_offset: int
) -> str:
    """Transform all ``&&`` and ``||`` on *line* using rightmost-first strategy."""

    def _inside_region(pos: int) -> bool:
        abs_pos = line_offset + pos
        for lo, hi in regions:
            if lo <= abs_pos < hi:
                return True
        return False

    while True:
        # Find rightmost && or || at depth 0 and outside regions.
        best_pos = -1
        best_op = ""
        best_len = 0

        for op, condition in [("&&", "$?"), ("||", "-not $?")]:
            pos = 0
            while True:
                idx = line.find(op, pos)
                if idx == -1:
                    break
                if _depth_at(line, idx) == 0 and not _inside_region(idx):
                    if idx > best_pos:
                        best_pos = idx
                        best_op = op
                        best_len = len(op)
                pos = idx + len(op)

        if best_pos == -1:
            break

        # Map op to condition
        condition = "$?" if best_op == "&&" else "-not $?"

        left = line[:best_pos].strip()
        right = line[best_pos + best_len:].strip()

        line = f"{left}; if ({condition}) {{ {right} }}"

    return line


# ============================================================================
# Transform 6: Null-conditional operators  (?. and ?[])
# ============================================================================

def _transform_null_conditional(code: str) -> str:
    """Transform ``$a?.Length`` and ``$a?[0]`` to null-check if-else forms.

    ``$a?.Length`` → ``if ($null -ne $a) { $a.Length }``
    ``$a?[0]``     → ``if ($null -ne $a) { $a[0] }``
    """
    regions = _find_string_regions(code)

    # Process both ?. and ?[, passing regions and line offsets
    code = _transform_null_conditional_dot(code, regions)
    code = _transform_null_conditional_bracket(code, regions)
    return code


def _transform_null_conditional_dot(code: str, regions: List[Tuple[int, int]]) -> str:
    """Transform ``?.`` (null-conditional member access).

    Handles chained null-conditionals by first identifying the base variable
    and then walking forward through the ``?.member`` chain, building nested
    ``if`` blocks from the inside out.

    ``$a?.Prop1?.Prop2`` →
    ``if ($null -ne $a) { if ($null -ne $a.Prop1) { $a.Prop1.Prop2 } }``.
    """
    lines = code.split("\n")
    result: List[str] = []

    abs_offset = 0
    for line in lines:
        line_offset = abs_offset
        abs_offset += len(line) + 1

        # Use a moving pos to skip past ?. inside regions
        pos = 0
        while True:
            idx = line.find("?.", pos)
            if idx == -1:
                break

            abs_pos = line_offset + idx

            if _depth_at(line, idx) != 0 or not _outside_regions(regions, abs_pos):
                # Inside string/comment or brackets — skip past this one
                pos = idx + 2
                continue

            # Find the base variable/expression before the first ?.
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end)
            base_expr = line[expr_start:expr_end].strip()

            if not base_expr:
                pos = idx + 2
                continue

            # Walk forward through the ?. chain.
            # Collect (member, has_args, args_str, end_pos) tuples.
            chain: list[tuple[str, bool, str, int]] = []
            pos = idx  # position of current ?.

            while pos < len(line) and line[pos:pos + 2] == "?.":
                member_start = pos + 2
                # skip whitespace
                while member_start < len(line) and line[member_start] == " ":
                    member_start += 1

                # Read member name
                member = ""
                i = member_start
                while i < len(line) and (line[i].isalnum() or line[i] == "_"):
                    member += line[i]
                    i += 1

                if not member:
                    break

                # Check for method call (...)
                has_args = False
                args_str = ""
                j = i
                while j < len(line) and line[j] == " ":
                    j += 1
                if j < len(line) and line[j] == "(":
                    paren_depth = 1
                    k = j + 1
                    while k < len(line) and paren_depth > 0:
                        if line[k] == "(":
                            paren_depth += 1
                        elif line[k] == ")":
                            paren_depth -= 1
                        k += 1
                    args_str = line[j:k]
                    has_args = True
                    i = k

                chain.append((member, has_args, args_str, i))
                pos = i  # check if next is also ?.

            if not chain:
                pos = idx + 2
                continue

            # First, compute all partial access paths (base, base.M1, base.M1.M2, ...)
            paths = [base_expr]
            for member, has_args, args_str, _ in chain:
                if has_args:
                    paths.append(f"{paths[-1]}.{member}{args_str}")
                else:
                    paths.append(f"{paths[-1]}.{member}")

            full_accessor = paths[-1]  # base.M1.M2...

            # Build inner from inside out, null-checking each partial path
            inner = full_accessor
            for check_path in reversed(paths[:-1]):
                inner = f"if ($null -ne {check_path}) {{ {inner} }}"

            # Check if this is part of an assignment
            before = line[:expr_start]
            after = line[chain[-1][3]:]  # after the last chain element

            assign_match = re.match(
                r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$", before
            )
            if assign_match:
                prefix = assign_match.group(1)
                target_var = assign_match.group(2)
                replacement = f"{prefix}{target_var} = {inner}"
            else:
                replacement = f"{before}{inner}"

            line = replacement + after

        result.append(line)

    return "\n".join(result)
    return "\n".join(result)


def _transform_null_conditional_bracket(
    code: str, regions: List[Tuple[int, int]]
) -> str:
    """Transform ``?[]`` (null-conditional index access).

    Uses the same iterative approach as the ``?.`` transform, checking
    string/comment regions.
    """
    lines = code.split("\n")
    result: List[str] = []

    abs_offset = 0
    for line in lines:
        line_offset = abs_offset
        abs_offset += len(line) + 1

        pos = 0
        while True:
            idx = line.find("?[", pos)
            if idx == -1:
                break

            abs_pos = line_offset + idx

            if _depth_at(line, idx) != 0 or not _outside_regions(regions, abs_pos):
                # Inside string/comment or brackets — skip past this one
                pos = idx + 2
                continue

            # Extract expression before ?[
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end)
            expr = line[expr_start:expr_end].strip()

            if not expr:
                pos = idx + 2
                continue

            # Find matching ]
            bracket_depth = 1
            bracket_end = idx + 2
            while bracket_end < len(line) and bracket_depth > 0:
                c = line[bracket_end]
                if c == "[":
                    bracket_depth += 1
                elif c == "]":
                    bracket_depth -= 1
                bracket_end += 1

            index_expr = line[idx + 2 : bracket_end - 1]

            before = line[:expr_start]
            after = line[bracket_end:]

            # Check if part of assignment
            assign_match = re.match(
                r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$", before
            )
            if assign_match:
                prefix = assign_match.group(1)
                target_var = assign_match.group(2)
                replacement = (
                    f"{prefix}{target_var} = "
                    f"if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
                )
            else:
                replacement = (
                    f"{before}"
                    f"if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
                )

            line = replacement + after

        result.append(line)

    return "\n".join(result)


def _find_var_start(line: str, end: int) -> int:
    """Walk backwards from *end* to find the start of a variable reference."""
    for i in range(end - 1, -1, -1):
        c = line[i]
        if c == " ":
            continue
        if c == "$":
            return i
        # If we hit something that's not a valid variable character, stop
        if not (c.isalnum() or c in "_.:"):
            return i + 1
    return 0


# ============================================================================
# Helper: generic binary-operator line transformer (NO depth check)
# ============================================================================


def _transform_binary_op_no_depth(
    code: str,
    op_str: str,
    transform_fn,
) -> str:
    """Apply *transform_fn* to every occurrence of *op_str* outside strings/comments.

    Unlike ``_transform_binary_op``, this does **not** skip matches inside
    ``{}`` brackets.  This is important for the ``??`` transform which runs
    after ``??=`` and may find ``??`` inside braces introduced by the
    previous transform.
    """
    regions = _find_string_regions(code)
    lines = code.split("\n")
    result: List[str] = []

    abs_offset = 0
    for line in lines:
        new_line = line
        pos = 0
        while pos < len(new_line):
            idx = new_line.find(op_str, pos)
            if idx == -1:
                break

            abs_pos = abs_offset + idx
            if _outside_regions(regions, abs_pos):
                rewritten = transform_fn(new_line, idx)
                if rewritten is not None:
                    new_line = rewritten
                    pos = 0  # restart scanning on rewritten line
                    continue
            pos = idx + len(op_str)

        result.append(new_line)
        abs_offset += len(line) + 1

    return "\n".join(result)


# ============================================================================
# Helper: generic binary-operator line transformer
# ============================================================================

def _transform_binary_op(
    code: str,
    regions: List[Tuple[int, int]],
    op_str: str,
    transform_fn,
) -> str:
    """Apply *transform_fn* to every occurrence of *op_str* at depth 0."""
    lines = code.split("\n")
    result: List[str] = []

    abs_offset = 0
    for line in lines:
        new_line = line
        pos = 0
        while pos < len(new_line):
            idx = new_line.find(op_str, pos)
            if idx == -1:
                break

            abs_pos = abs_offset + idx
            if _depth_at(new_line, idx) == 0 and _outside_regions(regions, abs_pos):
                rewritten = transform_fn(new_line, idx)
                if rewritten is not None:
                    new_line = rewritten
                    pos = 0  # restart scanning on rewritten line
                    continue
            pos = idx + len(op_str)

        result.append(new_line)
        abs_offset += len(line) + 1

    return "\n".join(result)


# ============================================================================
# Public API
# ============================================================================


def pwsh_transform(code: str) -> str:
    """Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax.

    Applies the following transformations in order:

    1. Null-coalescing assignment (``??=``)
    2. Null-coalescing (``??``)
    3. Ternary operator (``? :``)
    4. Pipeline chain AND (``&&``)
    5. Pipeline chain OR (``||``)
    6. Null-conditional operators (``?.`` and ``?[]``)

    The function is **idempotent**: applying it to already-transformed code
    produces the same result.
    """
    code = _join_continuation_lines(code)
    code = _transform_null_coalescing_assign(code)
    code = _transform_null_coalescing(code)
    code = _transform_ternary(code)
    code = _transform_pipeline_chains(code)
    code = _transform_null_conditional(code)
    return code


# ============================================================================
# CLI entry point (for ad-hoc testing)
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    print(pwsh_transform(text))
