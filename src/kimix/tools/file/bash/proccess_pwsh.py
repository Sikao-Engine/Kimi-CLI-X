"""Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax.

PowerShell 7 introduced several expression-level operators that do not exist in
PowerShell 5.1:

  * Ternary:          $cond ? $true_expr : $false_expr
  * Null-coalescing:  $a ?? $fallback
  * Null-assign:      $a ??= $default
  * Pipeline chains:  cmd1 && cmd2   /   cmd1 || cmd2
  * Null-conditional: $obj?.Property / $obj?[0]

This module performs a *source-to-source* transformation.  It operates on raw
text rather than an AST because the target environment (5.1) cannot parse the
new syntax in the first place.  The main challenges are:

1. **Strings and comments** – operators inside quoted strings or comments must
   never be touched.  We pre-compute "protected regions" for every line.

2. **Expression boundaries** – `??`, `?`, `?.`, `?[` and `:` must only match
   when they sit between real PowerShell expressions.  We scan backwards and
   forwards to locate expression boundaries, respecting parentheses and brackets.

3. **Backtick continuations** – PowerShell uses `` ` `` at end-of-line to
   continue a logical line.  We collapse those first so that a single ternary
   or chain operator that was split across physical lines becomes one line.

4. **Idempotency** – running the transformer twice must yield the same output.
   Every rewrite produces 5.1-only syntax, so a second pass is a no-op.

5. **Depth tracking** – some operators are valid only at brace/paren depth 0
   (e.g. ternary `?` inside a hashtable `{ $a ? $b : $c }` is intentionally
   skipped because the colon would clash with hashtable syntax).  Others, such
   as `&&`/`||`, are valid inside script blocks and sub-expressions, so we do
   NOT require depth 0 for them.

The public entry point is `pwsh_transform(code, warn_chain=False)`.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Region scanners  (strings, comments, here-strings)
# ---------------------------------------------------------------------------

def _find_string_regions(code: str) -> list[tuple[int, int]]:
    """Return a list of (start, end) intervals covering all string/comment
    regions in *code*.

    The scanner is a hand-written state machine.  It walks the source once
    (O(n)) and recognises five kinds of protected regions:

    1. Block comments   `` <# ... #> ``  (nestable)
    2. Line comments    `` # ... \n ``
    3. Here-strings     `` @'...'@ `` or `` @"..."@ ``
       A here-string start (``@'`` / ``@"``) is only valid when it appears at
       the end of a line (ignoring trailing whitespace).  The terminator must
       be the very first non-whitespace text on its own line.
    4. Single-quoted strings   `` '...' ``
       Two consecutive single quotes inside the string represent an escaped
       quote (PowerShell literal string syntax).
    5. Double-quoted strings   `` "..." ``
       Handles backtick escapes (`` `" ``) and sub-expressions `` $(...) ``.

    Any operator found inside one of these intervals must be ignored by the
    downstream transformers.
    """
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == "<" and i + 1 < n and code[i + 1] == "#":
            # --- block comment <# ... #> (nestable) ---
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
        elif c == "#":
            # --- line comment #...\n ---
            start = i
            while i < n and code[i] != "\n":
                i += 1
            regions.append((start, i))
        elif c == "@" and i + 1 < n and code[i + 1] in ("'", '"'):
            # --- here-string  @'...'@  or  @"..."@ ---
            # A real here-string start must be the last thing on its line
            # (only whitespace may follow).  Anything else is just an @
            # followed by a quote inside normal code and must be ignored.
            j = i + 2
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] != "\n":
                i += 1
                continue
            start = i
            quote_char = code[i + 1]
            i += 2
            while i < n:
                if code[i] == quote_char and i + 1 < n and code[i + 1] == "@":
                    line_start = code.rfind("\n", 0, i)
                    line_start = 0 if line_start == -1 else line_start + 1
                    # The terminator line must contain only whitespace before
                    # the closing quote+@.
                    if code[line_start:i].strip() == "":
                        i += 2
                        break
                i += 1
            regions.append((start, i))
        elif c == "'":
            # --- single-quoted string ---
            start = i
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
            regions.append((start, i))
        elif c == '"':
            # --- double-quoted string ---
            start = i
            i += 1
            while i < n:
                ch = code[i]
                if ch == "`" and i + 1 < n:
                    i += 2
                elif ch == '"':
                    i += 1
                    break
                elif ch == "$" and i + 1 < n and code[i + 1] == "(":
                    i = _skip_subexpression(code, i)
                else:
                    i += 1
            regions.append((start, i))
        else:
            i += 1
    return regions


def _skip_subexpression(code: str, start: int) -> int:
    """Skip past a ``$(...)`` sub-expression starting at *start*.

    The caller guarantees that ``code[start] == '$'`` and that the next
    character is '('.  We track parenthesis depth and also recurse into nested
    ``$(...)`` sub-expressions.  Single- and double-quoted strings inside the
    sub-expression are skipped so that a parenthesis inside a string literal
    does not affect depth tracking.

    Returns the index *after* the closing ')'.
    """
    assert code[start] == "$"
    i = start + 2
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


def _find_line_regions(line: str) -> list[tuple[int, int]]:
    """Line-level variant of `_find_string_regions`.

    Here-strings are omitted because they always span multiple lines and are
    handled at the whole-code level.  Everything else (block comments, line
    comments, single/double quoted strings) is recognised identically.

    This function is called by every per-line transformer so that each line
    has its own small region list, keeping the lookups fast.
    """
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "<" and i + 1 < n and line[i + 1] == "#":
            start = i
            depth = 1
            i += 2
            while i < n and depth:
                if line[i] == "<" and i + 1 < n and line[i + 1] == "#":
                    depth += 1
                    i += 2
                elif line[i] == "#" and i + 1 < n and line[i + 1] == ">":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            regions.append((start, i))
        elif c == "#":
            regions.append((i, n))
            break
        elif c in "'\"":
            start = i
            quote = c
            i += 1
            while i < n:
                if line[i] == quote:
                    if quote == "'" and i + 1 < n and line[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                if quote == '"' and line[i] == "`" and i + 1 < n:
                    i += 2
                elif quote == '"' and line[i] == "$" and i + 1 < n and line[i + 1] == "(":
                    i = _skip_subexpression(line, i)
                else:
                    i += 1
            regions.append((start, i))
        else:
            i += 1
    return regions


def _outside_regions(regions: list[tuple[int, int]], pos: int) -> bool:
    """Return ``True`` iff *pos* is not inside any of the supplied regions."""
    return not any(start <= pos < end for start, end in regions)


# ---------------------------------------------------------------------------
# Depth tracking
# ---------------------------------------------------------------------------

def _compute_depths(line: str) -> list[int]:
    """Return a list giving the nesting depth of ``()``, ``{}`` at every
    character position in *line*.

    The result has ``len(line) + 1`` elements.  ``depths[i]`` is the depth
    *before* processing ``line[i]``.  This allows transformers to ask
    "what is the depth at the current character?" without doing extra math.

    Note: square brackets ``[]`` are intentionally *not* tracked here.
    Array indexing and type literals use ``[]``, and we do not want them to
    affect depth because operators like ``??`` can legitimately appear next to
    array access (e.g. ``$a[0] ?? $b``).
    """
    depths: list[int] = []
    depth = 0
    for ch in line:
        depths.append(depth)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    depths.append(depth)
    return depths


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _join_continuation_lines(code: str) -> str:
    """Collapse backtick line-continuations into single logical lines.

    PowerShell allows a backtick (`` ` ``) at end-of-line to continue the
    statement on the next line.  If we do not collapse those first, a ternary
    or chain operator split across two physical lines would be invisible to the
    per-line transformers.

    Algorithm:
      * Scan the whole code.
      * When a backtick is found outside any string/comment region, look ahead:
        - Skip trailing whitespace.
        - If the next character is ``\n``, consume it and any leading whitespace
          on the following line, and emit a single space in place of the entire
          continuation.
        - Otherwise the backtick is not a continuation (e.g. inside a name);
          emit it literally.
    """
    regions = _find_string_regions(code)
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        if code[i] == "`" and _outside_regions(regions, i):
            j = i + 1
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] == "\n":
                j += 1
                while j < n and code[j] in " \t\r":
                    j += 1
                result.append(" ")
                i = j
                continue
        result.append(code[i])
        i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Assignment detection helpers
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$")

_COMMAND_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*\s+")


def _match_assignment(before: str) -> tuple[str, str] | None:
    """Try to match an assignment prefix like ``$var = `` at the end of *before*.

    Returns ``(prefix, var_name)`` when the pattern matches, else ``None``.
    *prefix* is everything before the variable name; it may be empty.

    Examples::

        "$x = "          -> ("", "$x")
        "$obj.Property = " -> ("", "$obj.Property")
        "  $a = "        -> ("  ", "$a")
    """
    m = _ASSIGN_RE.match(before)
    if m:
        return m.group(1), m.group(2)
    return None


# ---------------------------------------------------------------------------
# Null-coalescing assignment  (??=)
# ---------------------------------------------------------------------------

_NCA_RE = re.compile(r"(\$\w+(?:\.\w+)*)\s*\?\?=\s*(.+)")


def _transform_nca_line(line: str) -> str:
    """Rewrite null-coalescing assignment ``$var ??= value``.

    In PS 7, ``??=`` assigns *value* only when *$var* is ``$null``.
    PS 5.1 has no such operator, so we expand it to an explicit ``if``::

        $var ??= value
        ->  if ($null -eq $var) { $var = value }

    We process matches right-to-left so that earlier replacements do not
    invalidate the string indices of later matches.
    """
    regions = _find_line_regions(line)
    new_line = line
    for m in reversed(list(_NCA_RE.finditer(line))):
        if _outside_regions(regions, m.start()):
            var = m.group(1)
            value = m.group(2).rstrip()
            replacement = f"if ($null -eq {var}) {{ {var} = {value} }}"
            new_line = new_line[: m.start()] + replacement + new_line[m.end() :]
    return new_line


# ---------------------------------------------------------------------------
# Expression boundary helpers
# ---------------------------------------------------------------------------

def _find_expr_start(line: str, end: int, regions: list[tuple[int, int]]) -> int:
    """Scan backwards from *end* to locate the start of the expression.

    The expression start is the first character that belongs to the operand
    preceding an operator.  We stop when we encounter:

    * A boundary character ``= ; | & ,`` (outside strings/comments).
      ``=`` is special: compound assignment operators (``+=``, ``-=``, etc.)
      are ignored so that the left side of ``$a += $b ?? $c`` is still ``$b``.
    * An unmatched opening bracket ``(``, ``[``, ``{`` while scanning backwards
      through its matching closer.
    * The beginning of the line.

    Spaces are handled by treating them like closing brackets: each space
    increments a "depth" counter.  Because there is no corresponding opener,
    the counter never drops back to zero, which naturally causes the scan to
    continue all the way to the start of the line.  This is intentional—it lets
    multi-token expressions such as ``$a -gt 5`` be captured as a single unit.
    """
    depth = 0
    for i in range(end - 1, -1, -1):
        c = line[i]
        if c in ")]} ":
            depth += 1
        elif c in "([{":
            depth -= 1
            if depth < 0:
                return i + 1
        elif c in "=;|&," and _outside_regions(regions, i):
            if c == "=" and i > 0 and line[i - 1] in "=!<>+-*/.":
                continue
            return i + 1
    return 0


def _find_expr_end(line: str, start: int, regions: list[tuple[int, int]]) -> int:
    """Scan forwards from *start* to locate the end of the expression.

    The scan stops at:

    * A boundary character ``; | & ,`` (outside strings/comments).
    * A ``#`` that starts a line comment.
    * A closing bracket ``)``, ``]``, ``}`` that drops the depth below zero.
    * End of line.

    Opening brackets ``(``, ``[``, ``{`` increment depth so that expressions
    inside parentheses are consumed as a single unit.
    """
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]} ":
            depth -= 1
            if depth < 0:
                return i
        elif depth >= 0 and c in ";|&," and _outside_regions(regions, i):
            return i
        elif c == "#" and _outside_regions(regions, i):
            if i > 0 and line[i - 1] == "<":
                continue
            return i
    return len(line)


# ---------------------------------------------------------------------------
# Null-coalescing  (??)
# ---------------------------------------------------------------------------

def _nc_rewrite_line(line: str, op_pos: int, regions: list[tuple[int, int]]) -> str | None:
    """Rewrite a single ``??`` at *op_pos* into an ``if`` statement.

    We first discover the left and right operands using the expression boundary
    helpers.  If either operand is empty (e.g. trailing ``??`` with nothing
    after it) we return ``None`` and skip the rewrite.

    When the text before the left operand is an assignment, we fold the result
    back into that assignment::

        $x = $a ?? "default"
        ->  $x = if ($null -ne $a) { $a } else { "default" }

    Otherwise we emit a standalone ``if`` expression::

        $a ?? "default"
        ->  if ($null -ne $a) { $a } else { "default" }
    """
    left_end = op_pos
    while left_end > 0 and line[left_end - 1] == " ":
        left_end -= 1
    left_start = _find_expr_start(line, left_end, regions)
    right_start = op_pos + 2
    while right_start < len(line) and line[right_start] == " ":
        right_start += 1
    right_end = _find_expr_end(line, right_start, regions)
    left_expr = line[left_start:left_end].strip()
    right_expr = line[right_start:right_end].strip()
    if not left_expr or not right_expr:
        return None
    before = line[:left_start].rstrip()
    assign = _match_assignment(before)
    if assign:
        prefix, var_name = assign
        replacement = f"{prefix}{var_name} = if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
    else:
        replacement = f"{line[:left_start]}if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
    return replacement + line[right_end:]


def _transform_nc_line(line: str) -> str:
    """Transform every ``??`` on *line* into PS 5.1 compatible ``if`` form.

    Because a rewrite can change the length of the line, we restart the scan
    after every successful replacement (outer ``while True``).  This also
    handles nested null-coalescing naturally: ``$a ?? $b ?? $c`` becomes
    ``if ($null -ne $a) { $a } else { $b ?? $c }`` on the first pass, and the
    second iteration then transforms the remaining ``??``.
    """
    while True:
        regions = _find_line_regions(line)
        rewritten = False
        pos = 0
        while pos < len(line):
            idx = line.find("??", pos)
            if idx == -1:
                break
            if _outside_regions(regions, idx):
                replacement = _nc_rewrite_line(line, idx, regions)
                if replacement is not None:
                    line = replacement
                    rewritten = True
                    break
            pos = idx + 2
        if not rewritten:
            break
    return line


# ---------------------------------------------------------------------------
# Ternary  (? :)
# ---------------------------------------------------------------------------

def _find_matching_colon(
    line: str, start: int, regions: list[tuple[int, int]], depth_arr: list[int]
) -> int:
    """Find the colon that separates the true/false branches of a ternary.

    Starting from *start* (the character just after ``?``), scan forward and
    return the first ``:`` that is:

    * at nesting depth 0 (not inside parentheses or braces),
    * outside string/comment regions,
    * not part of a double-colon operator ``::`` (used for static members).

    If no such colon exists, return -1.
    """
    for i in range(start, len(line)):
        if line[i] == ":" and depth_arr[i] == 0 and _outside_regions(regions, i):
            if i > 0 and line[i - 1] == ":":
                continue
            if i + 1 < len(line) and line[i + 1] == ":":
                continue
            return i
    return -1


def _transform_ternary_line(line: str) -> str:
    """Rewrite ternary ``$cond ? $true : $false`` into an ``if`` statement.

    Design notes:

    * We scan left-to-right looking for ``?`` at depth 0, outside strings.
      The ``$?`` automatic variable is explicitly excluded (``?`` immediately
      preceded by ``$``).
    * For each candidate ``?``, we find the matching ``:`` with
      `_find_matching_colon`.  If none is found, the ``?`` is not a ternary
      operator and we move on.
    * The condition, true-expression and false-expression are extracted using
      the expression-boundary helpers.
    * When the text before the condition is an assignment, the ``if`` is
      folded into it::

          $x = $cond ? $a : $b
          ->  $x = if ($cond) { $a } else { $b }

    * **Command-prefix heuristic** – If there is NO assignment and the
      condition text starts with a bare command word (e.g. ``Write-Output``)
      followed by a real expression token, we strip the command word.  This
      prevents ``Write-Output $a ? $b : $c`` from being parsed with
      ``Write-Output $a`` as the condition.  The heuristic only fires when
      the remaining token starts with ``$``, ``(``, ``[``, a quote, ``@``, or
      a digit—characters that unambiguously begin a PowerShell expression.
    """
    regions = _find_line_regions(line)
    depth_arr = _compute_depths(line)
    pos = 0
    while pos < len(line):
        if (
            line[pos] == "?"
            and _outside_regions(regions, pos)
            and depth_arr[pos] == 0
            and not (pos > 0 and line[pos - 1] == "$")
        ):
            colon_pos = _find_matching_colon(line, pos + 1, regions, depth_arr)
            if colon_pos != -1:
                cond_end = pos
                while cond_end > 0 and line[cond_end - 1] == " ":
                    cond_end -= 1
                cond_start = _find_expr_start(line, cond_end, regions)
                condition = line[cond_start:cond_end].strip()
                true_start = pos + 1
                while true_start < len(line) and line[true_start] == " ":
                    true_start += 1
                true_end = colon_pos
                while true_end > true_start and line[true_end - 1] == " ":
                    true_end -= 1
                true_expr = line[true_start:true_end].strip()
                false_start = colon_pos + 1
                while false_start < len(line) and line[false_start] == " ":
                    false_start += 1
                false_end = _find_expr_end(line, false_start, regions)
                false_expr = line[false_start:false_end].strip()
                before = line[:cond_start].rstrip()
                assign = _match_assignment(before)
                if not assign and condition:
                    m = _COMMAND_PREFIX_RE.match(condition)
                    if m:
                        expr_part = condition[m.end():]
                        if expr_part and expr_part[0] in "$([\"'@0123456789":
                            cond_start += m.end()
                            condition = expr_part
                            before = line[:cond_start].rstrip()
                            assign = _match_assignment(before)
                if assign:
                    prefix, var = assign
                    replacement = f"{prefix}{var} = if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                else:
                    replacement = f"{line[:cond_start]}if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                suffix = line[false_end:]
                line = replacement + suffix
                regions = _find_line_regions(line)
                depth_arr = _compute_depths(line)
                pos = len(replacement)
                continue
        pos += 1
    return line


# ---------------------------------------------------------------------------
# Pipeline chain operators  (&& / ||)
# ---------------------------------------------------------------------------

def _transform_chain_line(line: str) -> str:
    """Rewrite pipeline chain operators ``&&`` and ``||``.

    PS 7 allows::

        cmd1 && cmd2   -> run cmd2 only if cmd1 succeeded
        cmd1 || cmd2   -> run cmd2 only if cmd1 failed

    PS 5.1 has no such operator.  We expand them into ``if ($?)`` checks::

        cmd1 && cmd2
        ->  cmd1; if ($?) { cmd2 }

        cmd1 || cmd2
        ->  cmd1; if (-not $?) { cmd2 }

    Design notes:

    * Unlike ternary and null-conditional, ``&&``/``||`` are valid inside
      script blocks ``{ ... }`` and sub-expressions ``$(...)``.  Therefore we
      do **not** require depth == 0; we only skip operators that lie inside
      strings or comments.
    * We always pick the *rightmost* operator on the line to transform first.
      This mirrors the right-associative expansion that PS 7 performs::

          cmd1 && cmd2 && cmd3
          ->  cmd1; if ($?) { cmd2; if ($?) { cmd3 } }
    """
    while True:
        regions = _find_line_regions(line)
        best_pos = -1
        best_op = ""
        best_len = 0
        for op in ("&&", "||"):
            pos = 0
            op_len = len(op)
            while True:
                idx = line.find(op, pos)
                if idx == -1:
                    break
                if _outside_regions(regions, idx):
                    if idx > best_pos:
                        best_pos = idx
                        best_op = op
                        best_len = op_len
                pos = idx + op_len
        if best_pos == -1:
            break
        condition = "$?" if best_op == "&&" else "-not $?"
        left = line[:best_pos].strip()
        right = line[best_pos + best_len :].strip()
        line = f"{left}; if ({condition}) {{ {right} }}"
    return line


# ---------------------------------------------------------------------------
# Null-conditional member access  (?.)
# ---------------------------------------------------------------------------

def _transform_null_conditional_dot_line(line: str) -> str:
    """Rewrite null-conditional member access ``$obj?.Member``.

    PS 7 syntax::

        $obj?.Property
        $obj?.Method($arg)
        $obj?.A?.B?.C

    is expanded into a nested ladder of ``if ($null -ne ...)`` guards::

        $obj?.Property
        ->  if ($null -ne $obj) { $obj.Property }

        $obj?.A?.B
        ->  if ($null -ne $obj) { if ($null -ne $obj.A) { $obj.A.B } }

    Design notes:

    * We consume a chain of ``?.`` operators in one go, building a list of
      member names (and optional argument lists for method calls).
    * Method calls are detected by looking for ``(`` immediately after the
      member name; we balance parentheses to capture the full argument list.
    * The inner-most expression is the final member access; each outer layer
      wraps it in another ``if`` guard.
    * Assignment context is detected so that ``$x = $obj?.Prop`` produces
      ``$x = if ($null -ne $obj) { $obj.Prop }``.
    """
    while True:
        regions = _find_line_regions(line)
        depth_arr = _compute_depths(line)
        pos = 0
        matched = False
        while True:
            idx = line.find("?.", pos)
            if idx == -1:
                break
            if depth_arr[idx] != 0 or not _outside_regions(regions, idx):
                pos = idx + 2
                continue
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end, regions)
            base = line[expr_start:expr_end].strip()
            if not base:
                pos = idx + 2
                continue
            chain: list[tuple[str, str, int]] = []
            pos = idx
            while pos < len(line) and line[pos : pos + 2] == "?.":
                ms = pos + 2
                while ms < len(line) and line[ms] == " ":
                    ms += 1
                me = ms
                while me < len(line) and (line[me].isalnum() or line[me] == "_"):
                    me += 1
                if me == ms:
                    break
                mem = line[ms:me]
                args = ""
                j = me
                while j < len(line) and line[j] == " ":
                    j += 1
                if j < len(line) and line[j] == "(":
                    d = 1
                    k = j + 1
                    while k < len(line) and d > 0:
                        if _outside_regions(regions, k):
                            if line[k] == "(":
                                d += 1
                            elif line[k] == ")":
                                d -= 1
                        k += 1
                    args = line[j:k]
                    me = k
                chain.append((mem, args, me))
                pos = me
            if not chain:
                pos = idx + 2
                continue
            paths = [base]
            for mem, args, _ in chain:
                paths.append(f"{paths[-1]}.{mem}{args}")
            inner = paths[-1]
            for p in reversed(paths[:-1]):
                inner = f"if ($null -ne {p}) {{ {inner} }}"
            before = line[:expr_start]
            after = line[chain[-1][2] :]
            assign = _match_assignment(before)
            repl = (f"{assign[0]}{assign[1]} = " if assign else before) + inner
            line = repl + after
            depth_arr = _compute_depths(line)
            matched = True
            break
        if not matched:
            break
    return line


# ---------------------------------------------------------------------------
# Null-conditional index access  (?[)
# ---------------------------------------------------------------------------

def _transform_null_conditional_bracket_line(line: str) -> str:
    """Rewrite null-conditional index access ``$obj?[index]``.

    PS 7 syntax::

        $obj?[0]
        $obj?[$i + 1]

    becomes::

        if ($null -ne $obj) { $obj[0] }
        if ($null -ne $obj) { $obj[$i + 1] }

    The implementation is analogous to the dot-variant but simpler because
    there is no method-call chaining: we locate the matching ``]`` by
    bracket-depth tracking, extract the index expression, and wrap the whole
    thing in a single ``if`` guard.
    """
    while True:
        regions = _find_line_regions(line)
        depth_arr = _compute_depths(line)
        pos = 0
        matched = False
        while True:
            idx = line.find("?[", pos)
            if idx == -1:
                break
            if depth_arr[idx] != 0 or not _outside_regions(regions, idx):
                pos = idx + 2
                continue
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end, regions)
            expr = line[expr_start:expr_end].strip()
            if not expr:
                pos = idx + 2
                continue
            bracket_depth = 1
            bracket_end = idx + 2
            while bracket_end < len(line) and bracket_depth > 0:
                c = line[bracket_end]
                if _outside_regions(regions, bracket_end):
                    if c == "[":
                        bracket_depth += 1
                    elif c == "]":
                        bracket_depth -= 1
                bracket_end += 1
            index_expr = line[idx + 2 : bracket_end - 1]
            before = line[:expr_start]
            after = line[bracket_end:]
            assign = _match_assignment(before)
            if assign:
                prefix, target_var = assign
                repl = f"{prefix}{target_var} = if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
            else:
                repl = f"{before}if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
            line = repl + after
            depth_arr = _compute_depths(line)
            matched = True
            break
        if not matched:
            break
    return line


# ---------------------------------------------------------------------------
# Warning helpers
# ---------------------------------------------------------------------------

def _has_chain_operators(code: str) -> bool:
    """Return ``True`` if *code* contains ``&&`` or ``||`` outside strings/comments.

    This is used by the public ``pwsh_transform`` entry point to decide
    whether to emit a semantic warning about pipeline chaining.
    """
    regions = _find_string_regions(code)
    for op in ("&&", "||"):
        pos = 0
        while True:
            idx = code.find(op, pos)
            if idx == -1:
                break
            if _outside_regions(regions, idx):
                return True
            pos = idx + 2
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pwsh_transform(code: str, *, warn_chain: bool = False) -> tuple[str, str]:
    """Transform PowerShell 7.x syntax into PowerShell 5.1 compatible syntax.

    Parameters
    ----------
    code:
        The PowerShell source code to transform.
    warn_chain:
        When ``True``, inspect the original *code* for ``&&``/``||`` operators
        and return a warning string reminding the caller that these operators
        test the ``$?`` automatic variable (command success), not the raw exit
        code.

    Returns
    -------
    A 2-tuple ``(transformed_code, warning)``.  *warning* is an empty string
    when *warn_chain* is ``False`` or when no chain operators are present.

    Transformation pipeline (applied per line, after backtick continuation
    lines have been collapsed):

    1. Null-coalescing assignment ``??=``
    2. Null-coalescing ``??``
    3. Ternary ``? :``
    4. Pipeline chains ``&&`` / ``||``
    5. Null-conditional member access ``?.``
    6. Null-conditional index access ``?[``

    Multi-line string regions (here-strings, block comments that span lines)
    are left completely untouched.
    """
    has_chain = _has_chain_operators(code) if warn_chain else False
    code = _join_continuation_lines(code)
    lines = code.split("\n")
    regions = _find_string_regions(code)
    # Build offset table so we can map absolute positions back to line indices.
    offs = [0]
    for line in lines[:-1]:
        offs.append(offs[-1] + len(line) + 1)
    # Any line that falls inside a multi-line protected region is skipped.
    multi: set[int] = set()
    for s, e in regions:
        if "\n" not in code[s:e]:
            continue
        for i, o in enumerate(offs):
            if o < e and o + len(lines[i]) > s:
                multi.add(i)
    result: list[str] = []
    for i, line in enumerate(lines):
        if i in multi:
            result.append(line)
            continue
        line = _transform_nca_line(line)
        line = _transform_nc_line(line)
        line = _transform_ternary_line(line)
        line = _transform_chain_line(line)
        line = _transform_null_conditional_dot_line(line)
        line = _transform_null_conditional_bracket_line(line)
        result.append(line)
    result_code = "\n".join(result)
    warning = ""
    if has_chain:
        warning = (
            "WARNING: PowerShell `&&` and `||` check the `$?` automatic variable "
            "(success of last native command), NOT the raw exit code like bash. "
        )
    return result_code, warning


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    print(pwsh_transform(text))
