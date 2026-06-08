"""Comprehensive tests for pwsh_transform (PowerShell 7.x → 5.1 syntax transformer)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module directly to avoid the bash_tool.py Python 3.14 issue
_MODULE_PATH = Path(__file__).parent.parent / "src" / "kimix" / "tools" / "file" / "bash" / "proccess_pwsh.py"
_spec = importlib.util.spec_from_file_location("proccess_pwsh", str(_MODULE_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pwsh_transform = _mod.pwsh_transform


# ============================================================================
# Ternary operator  (? :)
# ============================================================================


class TestTernaryOperator:
    def test_simple_ternary(self) -> None:
        result, _ = pwsh_transform('$x = $cond ? "a" : "b"')
        assert "if ($cond)" in result
        assert '{ "a" }' in result
        assert '{ "b" }' in result

    def test_ternary_with_comparison(self) -> None:
        result, _ = pwsh_transform("$x = $a -gt 5 ? $a : 0")
        assert "if ($a -gt 5)" in result
        assert "{ $a }" in result
        assert "{ 0 }" in result

    def test_ternary_in_assignment(self) -> None:
        result, _ = pwsh_transform('$status = $count -eq 0 ? "empty" : "non-empty"')
        assert "$status = " in result
        assert "($count -eq 0)" in result

    def test_ternary_with_function_calls(self) -> None:
        result, _ = pwsh_transform('$x = Test-Path $p ? (Get-Item $p) : $null')
        assert "if (Test-Path $p)" in result
        assert "(Get-Item $p)" in result
        assert "$null" in result

    def test_ternary_no_assignment(self) -> None:
        result, _ = pwsh_transform('$cond ? "yes" : "no"')
        assert 'if ($cond) { "yes" } else { "no" }' in result


# ============================================================================
# Null-coalescing  (??)
# ============================================================================


class TestNullCoalescing:
    def test_simple_null_coalescing(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? "default"')
        assert "if ($null -ne $a)" in result
        assert '{ $a }' in result
        assert '{ "default" }' in result

    def test_null_coalescing_with_variable(self) -> None:
        result, _ = pwsh_transform("$x = $a ?? $b")
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $b }" in result

    def test_null_coalescing_with_literal_default(self) -> None:
        result, _ = pwsh_transform("$path = $env:HOME ?? 'C:\\Users\\Default'")
        assert "if ($null -ne $env:HOME)" in result
        assert "{ $env:HOME }" in result

    def test_nested_null_coalescing(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? $b ?? "default"')
        # After first ?? transform, the result contains another ??
        # which should also be transformed
        assert "default" in result
        assert "if ($null -ne " in result

    def test_null_coalescing_no_assignment(self) -> None:
        result, _ = pwsh_transform('$a ?? "fallback"')
        assert 'if ($null -ne $a) { $a } else { "fallback" }' in result


# ============================================================================
# Null-coalescing assignment  (??=)
# ============================================================================


class TestNullCoalescingAssignment:
    def test_simple_assign(self) -> None:
        result, _ = pwsh_transform('$a ??= "default"')
        assert "if ($null -eq $a)" in result
        assert '$a = "default"' in result

    def test_assign_with_expression(self) -> None:
        result, _ = pwsh_transform("$count ??= (Get-ChildItem).Count")
        assert "if ($null -eq $count)" in result
        assert "$count = (Get-ChildItem).Count" in result

    def test_assign_does_not_conflict_with_null_coalescing(self) -> None:
        """??= should be transformed before ?? so ??= is not partially matched."""
        result, _ = pwsh_transform("$a ??= $b ?? $c")
        # ??= should be fully resolved
        assert "??=" not in result
        assert "??" not in result


# ============================================================================
# Pipeline chain AND  (&&)
# ============================================================================


class TestPipelineChainAnd:
    def test_simple_and_chain(self) -> None:
        result, _ = pwsh_transform("cmd1 && cmd2")
        assert ";" in result
        assert "if ($?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_and_chain(self) -> None:
        result, _ = pwsh_transform("cmd1 && cmd2 && cmd3")
        assert "cmd1;" in result
        assert "if ($?) { cmd2; if ($?) { cmd3 } }" in result

    def test_and_chain_with_pipeline(self) -> None:
        result, _ = pwsh_transform("Get-Process | Where-Object CPU && Write-Output done")
        assert "Get-Process | Where-Object CPU" in result
        assert "Write-Output done" in result
        assert "if ($?)" in result


# ============================================================================
# Pipeline chain OR  (||)
# ============================================================================


class TestPipelineChainOr:
    def test_simple_or_chain(self) -> None:
        result, _ = pwsh_transform("cmd1 || cmd2")
        assert ";" in result
        assert "if (-not $?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_or_chain(self) -> None:
        result, _ = pwsh_transform("cmd1 || cmd2 || cmd3")
        assert "cmd1;" in result
        assert "if (-not $?) { cmd2; if (-not $?) { cmd3 } }" in result


# ============================================================================
# Null-conditional  (?. and ?[])
# ============================================================================


class TestNullConditional:
    def test_property_access(self) -> None:
        result, _ = pwsh_transform("$a?.Length")
        assert "if ($null -ne $a) { $a.Length }" in result

    def test_index_access(self) -> None:
        result, _ = pwsh_transform("$a?[0]")
        assert "if ($null -ne $a) { $a[0] }" in result

    def test_chained_null_conditional(self) -> None:
        result, _ = pwsh_transform("$a?.Property?.SubProperty")
        # Both ?. should be transformed
        assert "?." not in result

    def test_null_conditional_with_method(self) -> None:
        result, _ = pwsh_transform("$a?.ToString()")
        assert "if ($null -ne $a) { $a.ToString() }" in result

    def test_null_conditional_assignment(self) -> None:
        result, _ = pwsh_transform("$x = $a?.Length")
        assert "$x = if ($null -ne $a) { $a.Length }" in result


# ============================================================================
# Combined transformations
# ============================================================================


class TestCombinedTransformations:
    def test_multiple_features(self) -> None:
        code = '$x = $a ?? "default"\nGet-Process && Write-Output done'
        result, _ = pwsh_transform(code)
        assert "??" not in result
        assert "&&" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($?)" in result

    def test_no_false_positives_in_strings(self) -> None:
        code = "Write-Output 'The ?? operator is new'"
        result, _ = pwsh_transform(code)
        # The ?? inside the string should not be transformed
        assert "??" in result
        assert "if ($null -ne" not in result

    def test_no_false_positives_in_comments(self) -> None:
        code = "# This ?? is a comment\nWrite-Output hello"
        result, _ = pwsh_transform(code)
        assert "??" in result  # still in comment

    def test_no_false_positives_in_double_quoted_string(self) -> None:
        code = 'Write-Output "The ?? operator"'
        result, _ = pwsh_transform(code)
        assert "??" in result

    def test_combined_and_or(self) -> None:
        result, _ = pwsh_transform("cmd1 && cmd2 || cmd3")
        assert "&&" not in result
        assert "||" not in result


# ============================================================================
# Idempotency
# ============================================================================


class TestIdempotency:
    def test_double_transform_same_result(self) -> None:
        code = '$x = $a ?? "default"\n$y = $cond ? "yes" : "no"\nGet-Process && Write-Output done'
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_ternary_idempotent(self) -> None:
        code = '$x = $cond ? "a" : "b"'
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_null_coalescing_idempotent(self) -> None:
        code = '$x = $a ?? "default"'
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_pipeline_chain_idempotent(self) -> None:
        code = "cmd1 && cmd2"
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_null_conditional_idempotent(self) -> None:
        code = "$a?.Length"
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_strings_with_operators_not_transformed(self) -> None:
        code = """Write-Output 'Use ?? for null-coalescing'
Write-Output "A ? B : C is ternary"
Write-Output 'cmd1 && cmd2 is chain'"""
        result, _ = pwsh_transform(code)
        assert "?? for null-coalescing" in result
        assert "A ? B : C is ternary" in result
        assert "cmd1 && cmd2 is chain" in result

    def test_comments_not_transformed(self) -> None:
        code = """# The ?? operator is new in PS7
# $x = $cond ? "a" : "b"
# cmd1 && cmd2
Write-Output hello"""
        result, _ = pwsh_transform(code)
        assert "The ?? operator" in result
        assert '$cond ? "a" : "b"' in result
        assert "cmd1 && cmd2" in result

    def test_here_string_not_transformed(self) -> None:
        code = """$text = @'
The ?? operator is preserved here.
And so is the ?. operator.
'@
Write-Output $text"""
        result, _ = pwsh_transform(code)
        assert "??" in result  # preserved inside here-string
        assert "?." in result

    def test_multiline_with_backtick(self) -> None:
        code = "Get-Process `\n| Where-Object CPU `\n&& Write-Output done"
        result, _ = pwsh_transform(code)
        assert "&&" not in result
        assert "if ($?)" in result

    def test_empty_code(self) -> None:
        result, _ = pwsh_transform(""); assert result == ""

    def test_no_operators(self) -> None:
        code = "Write-Output 'hello world'"
        result, _ = pwsh_transform(code); assert result == code

    def test_ternary_in_pipeline(self) -> None:
        code = "$x = $a ? $b : $c | ForEach-Object { $_ }"
        result, _ = pwsh_transform(code)
        assert "?" not in result
        assert "if ($a)" in result

    def test_null_coalescing_with_property(self) -> None:
        code = '$name = $obj.Name ?? "Unknown"'
        result, _ = pwsh_transform(code)
        assert "if ($null -ne $obj.Name)" in result
        assert "Unknown" in result

    def test_block_comment_not_transformed(self) -> None:
        code = "<# The ?? and ?. operators are new #>\nWrite-Output hello"
        result, _ = pwsh_transform(code)
        assert "??" in result  # preserved in block comment
        assert "?." in result

    def test_null_conditional_bracket_with_expression(self) -> None:
        result, _ = pwsh_transform("$a?[$i + 1]")
        assert "if ($null -ne $a) { $a[$i + 1] }" in result


# ============================================================================
# Corner case: nested ternary
# ============================================================================


class TestNestedTernary:
    def test_nested_in_true_branch(self) -> None:
        """Nested ternary: only the outer ?: is transformed in one pass."""
        result, _ = pwsh_transform('$x = $a ? ($b ? "c" : "d") : "e"')
        # Outer ternary is transformed; inner remains (one-pass limitation)
        assert 'if ($a)' in result
        assert '($b ? "c" : "d")' in result or '"c"' in result
        assert '"e"' in result

    def test_nested_in_false_branch(self) -> None:
        """Nested ternary in false branch: outer transformed, inner remains."""
        result, _ = pwsh_transform('$x = $a ? "yes" : ($b ? "maybe" : "no")')
        assert 'if ($a)' in result

    def test_deeply_nested_ternary(self) -> None:
        """Deeply nested ternary: only outermost ?: transformed per pass."""
        result, _ = pwsh_transform('$x = $a ? ($b ? ($c ? 1 : 2) : 3) : 4')
        assert "if ($a)" in result
        # inner ternaries preserved
        assert "?" in result  # inner ? operators still present


# ============================================================================
# Corner case: multiple operators on one line
# ============================================================================


class TestMultipleOperatorsOneLine:
    def test_multiple_null_coalescing_one_line(self) -> None:
        """$a ?? $b on same line as $c ?? $d (separated by semicolon)."""
        result, _ = pwsh_transform('$x = $a ?? "x"; $y = $b ?? "y"')
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result

    def test_multiple_null_conditional_one_line(self) -> None:
        result, _ = pwsh_transform('$x = $a?.Name; $y = $b?.Count')
        assert "?." not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result

    def test_mixed_operators_one_line(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? $b; $y = $c ? "t" : "f"')
        assert "??" not in result
        assert "?" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($c)" in result

    def test_multiple_null_coalescing_assign_one_line(self) -> None:
        """Multiple ??= on one line: only the leftmost is fully captured.
        Known limitation: the regex greedily captures everything after ??=."""
        result, _ = pwsh_transform('$a ??= "x"; $b ??= "y"')
        # At minimum, the first ??= is processed
        assert "if ($null -eq $a)" in result


# ============================================================================
# Corner case: chained null-conditional with methods
# ============================================================================


class TestNullConditionalMethodChain:
    def test_method_with_args(self) -> None:
        result, _ = pwsh_transform("$a?.GetValue($param)")
        assert "?." not in result
        assert "if ($null -ne $a) { $a.GetValue($param) }" in result

    def test_method_with_multiple_args(self) -> None:
        result, _ = pwsh_transform("$a?.Invoke($x, $y, $z)")
        assert "?." not in result
        assert "$a.Invoke($x, $y, $z)" in result

    def test_method_with_no_args(self) -> None:
        result, _ = pwsh_transform("$a?.Dispose()")
        assert "?." not in result
        assert "$a.Dispose()" in result

    def test_chained_method_calls(self) -> None:
        result, _ = pwsh_transform("$a?.ToString()?.Split()")
        assert "?." not in result
        assert "$a.ToString()" in result
        assert "$a.ToString().Split()" in result


# ============================================================================
# Corner case: mixed null-conditional dot and bracket
# ============================================================================


class TestMixedNullConditional:
    def test_dot_then_bracket(self) -> None:
        """?. followed by ?[ is tricky: ?. is processed first."""
        result, _ = pwsh_transform("$a?.Items?[0]")
        # The ?. should be transformed; ?[ may remain depending on order
        assert "?." not in result
        assert "$a.Items" in result

    def test_bracket_then_dot(self) -> None:
        """?[ followed by ?.: ?. processed first, ?[ may remain inside braces.
        This no longer hangs (infinite-loop bug fixed); result preserves ?[ at depth > 0."""
        result, _ = pwsh_transform("$a?[0]?.Name")
        # ?. should be transformed
        assert "?." not in result
        assert "$a" in result

    def test_dot_bracket_dot_chain(self) -> None:
        """Long chain: ?. processed first, ?[ preserved at depth > 0. No hang."""
        result, _ = pwsh_transform("$a?.Items?[0]?.LastName")
        assert "$a.Items" in result

    def test_bracket_with_nested_expr(self) -> None:
        result, _ = pwsh_transform("$a?[$i?.ToString()]")
        # The inner ?. is inside brackets; depends on implementation whether it's transformed
        # At minimum, the outer ?[ should be transformed
        assert "if ($null -ne $a)" in result


# ============================================================================
# Corner case: chain operators with complex pipelines
# ============================================================================


class TestChainComplexPipelines:
    def test_and_chain_with_pipe_and_args(self) -> None:
        result, _ = pwsh_transform("Get-ChildItem -Path $env:USERPROFILE -Recurse && Write-Output 'done'")
        assert "&&" not in result
        assert "if ($?)" in result
        assert "Get-ChildItem -Path $env:USERPROFILE -Recurse" in result

    def test_or_chain_after_failed_command(self) -> None:
        result, _ = pwsh_transform("Test-Path $f || New-Item $f")
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_and_or_chain_sequence(self) -> None:
        result, _ = pwsh_transform("cmd1 && cmd2 || cmd3")
        assert "&&" not in result
        assert "||" not in result
        # Should be: cmd1; if ($?) { cmd2; if (-not $?) { cmd3 } }
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_or_and_chain_sequence(self) -> None:
        result, _ = pwsh_transform("cmd1 || cmd2 && cmd3")
        assert "&&" not in result
        assert "||" not in result
        assert "if (-not $?)" in result
        assert "if ($?)" in result

    def test_triple_and_chain(self) -> None:
        result, _ = pwsh_transform("cmd1 && cmd2 && cmd3 && cmd4")
        assert "&&" not in result
        # Check that all three chain points are there
        assert result.count("if ($?)") == 3


# ============================================================================
# Corner case: edge literal / variable patterns
# ============================================================================


class TestEdgeLiteralPatterns:
    def test_dollar_question_not_transformed(self) -> None:
        """$? is an automatic variable, should not be confused with ?. or ternary."""
        result, _ = pwsh_transform("if ($?) { Write-Output ok }")
        assert "$?" in result  # $? preserved
        assert "if ($?) { Write-Output ok }" == result

    def test_question_mark_in_variable_name(self) -> None:
        """Variable with ? in name like ${foo?} should not cause transformation."""
        # This is unusual but let's make sure it doesn't crash
        result, _ = pwsh_transform('Write-Output ${foo?}')
        # Should not have transformed anything
        assert "Write-Output" in result

    def test_null_coalescing_with_null_literal(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? $null')
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $null }" in result

    def test_null_coalescing_with_true_false(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? $true')
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $true }" in result

    def test_ternary_with_null(self) -> None:
        result, _ = pwsh_transform('$x = $cond ? $null : "default"')
        assert "if ($cond)" in result
        assert "{ $null }" in result


# ============================================================================
# Corner case: here-string double-quoted variant
# ============================================================================


class TestHereStringDoubleQuoted:
    def test_double_quoted_here_string(self) -> None:
        code = r'''$text = @"
The ?? operator is preserved.
And so is ?. and ?[
"@
Write-Output $text'''
        result, _ = pwsh_transform(code)
        assert "??" in result  # preserved
        assert "?." in result
        assert "?[" in result

    def test_at_quote_single_line_here_string(self) -> None:
        code = "$text = @'?? is not transformed here'@\nWrite-Output $text"
        result, _ = pwsh_transform(code)
        assert "??" in result


# ============================================================================
# Corner case: backtick continuation with various operators
# ============================================================================


class TestBacktickContinuationOperators:
    def test_null_coalescing_with_backtick(self) -> None:
        code = '$x = $a ??`\n  "default"'
        result, _ = pwsh_transform(code)
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_ternary_with_backtick_continuation(self) -> None:
        code = '$x = $cond ?`\n  "yes" :`\n  "no"'
        result, _ = pwsh_transform(code)
        assert "?" not in result
        assert "if ($cond)" in result

    def test_null_conditional_with_backtick(self) -> None:
        code = "$a?.`\n  Property"
        result, _ = pwsh_transform(code)
        # After backtick join, the ?. is on a single line w/spaces
        assert "?." not in result

    def test_chain_with_backtick_continuation(self) -> None:
        code = "cmd1 `\n&& cmd2"
        result, _ = pwsh_transform(code)
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: expression boundaries
# ============================================================================


class TestExprBoundaries:
    def test_null_coalescing_with_parenthesized_left(self) -> None:
        result, _ = pwsh_transform('$x = (Get-Item $p) ?? "default"')
        assert "if ($null -ne (Get-Item $p))" in result

    def test_null_coalescing_with_subexpression(self) -> None:
        result, _ = pwsh_transform('$x = $(Get-Date) ?? "never"')
        assert "if ($null -ne $(Get-Date))" in result

    def test_null_conditional_on_subexpression(self) -> None:
        """$()?.Property - null conditional on a subexpression."""
        result, _ = pwsh_transform("$(Get-Item $p)?.Length")
        # The subexpression $(...) should be detected as the base
        assert "?." not in result
        assert "if ($null -ne $(Get-Item $p))" in result

    def test_ternary_with_expression_condition(self) -> None:
        result, _ = pwsh_transform('$x = (Get-Date).Year -gt 2020 ? "new" : "old"')
        assert "?" not in result
        assert "if ((Get-Date).Year -gt 2020)" in result

    def test_ternary_with_complex_true_branch(self) -> None:
        result, _ = pwsh_transform('$x = $cond ? (Get-Process | Select -First 1) : $null')
        assert "?" not in result
        assert "if ($cond)" in result
        assert "(Get-Process | Select -First 1)" in result


# ============================================================================
# Corner case: ??= idempotency and edge patterns
# ============================================================================


class TestNCAEdgeCases:
    def test_nca_idempotent(self) -> None:
        code = '$a ??= "default"'
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_nca_with_same_line_code(self) -> None:
        result, _ = pwsh_transform('$a ??= "x"; Write-Output $a')
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "Write-Output $a" in result

    def test_nca_right_side_with_spaces(self) -> None:
        result, _ = pwsh_transform("$a ??= (Get-ChildItem).Count")
        assert "??=" not in result
        assert "$a = (Get-ChildItem).Count" in result


# ============================================================================
# Corner case: warn_chain flag
# ============================================================================


class TestWarnChain:
    def test_warn_chain_produces_warning(self) -> None:
        _, warning = pwsh_transform("cmd1 && cmd2", warn_chain=True)
        assert "WARNING" in warning
        assert "$?" in warning

    def test_warn_chain_no_warning_without_operator(self) -> None:
        _, warning = pwsh_transform("Write-Output hello", warn_chain=True)
        assert warning == ""

    def test_warn_chain_with_or(self) -> None:
        _, warning = pwsh_transform("cmd1 || cmd2", warn_chain=True)
        assert "WARNING" in warning

    def test_warn_chain_defaults_to_no_warning(self) -> None:
        _, warning = pwsh_transform("cmd1 && cmd2")
        assert warning == ""


# ============================================================================
# Corner case: operators inside splatting / hashtable context
# ============================================================================


class TestOperatorsInSpecialContext:
    def test_question_in_hashtable_access(self) -> None:
        """@{}.Keys - accessing a hashtable's Keys property."""
        result, _ = pwsh_transform("$x = @{ key = 'val' }.Keys")
        assert "@{" in result

    def test_colon_in_hashtable_not_confused(self) -> None:
        """Ternary inside @{ } is at depth > 0 so it is NOT transformed.
        This is intentional: colons inside braces could be switch/hashtable syntax."""
        result, _ = pwsh_transform('$x = @{ key = $a ? "t" : "f" }')
        # Ternary inside braces is preserved (depth > 0)
        assert "?" in result  # not transformed at depth > 0

    def test_colon_in_string_not_confused(self) -> None:
        """Colon inside a string is not a ternary colon.
        Note: _find_matching_colon may not exclude in-string colons currently."""
        result, _ = pwsh_transform('$x = $cond ? "no-colon" : "default"')
        # Works correctly when strings have no colons
        assert "?" not in result
        assert "if ($cond)" in result


# ============================================================================
# Corner case: whitespace and formatting stress
# ============================================================================


class TestWhitespaceStress:
    def test_no_spaces_around_ternary(self) -> None:
        result, _ = pwsh_transform('$x=$cond?"a":"b"')
        assert "?" not in result
        assert "if ($cond)" in result

    def test_no_spaces_around_null_coalescing(self) -> None:
        result, _ = pwsh_transform('$x=$a??"default"')
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_no_spaces_around_null_conditional(self) -> None:
        result, _ = pwsh_transform("$a?.Property?.SubProperty")
        assert "?." not in result

    def test_extra_spaces_around_operators(self) -> None:
        result, _ = pwsh_transform('$x  =   $a    ??    "default"')
        assert "??" not in result

    def test_tabs_around_operators(self) -> None:
        result, _ = pwsh_transform("$x\t=\t$a\t??\t'default'")
        assert "??" not in result


# ============================================================================
# Corner case: code that looks like operators but at end of line
# ============================================================================


class TestTrickyOperatorPlacement:
    def test_and_at_end_of_command(self) -> None:
        """&& at end of line is still valid operator."""
        result, _ = pwsh_transform("cmd1 &&")
        # After transformation, the trailing && situation might be edge
        assert "cmd1" in result

    def test_question_at_end_of_line(self) -> None:
        """Isolated ? at end should not cause error."""
        result, _ = pwsh_transform("$a ?")
        # No colon, so no ternary transformation
        assert "$a" in result

    def test_double_question_at_end(self) -> None:
        """?? at end of line without right side - should be safe."""
        result, _ = pwsh_transform("$a ??")
        # Should not crash; right side is missing
        assert "$a" in result

    def test_null_conditional_at_end(self) -> None:
        """?. at end of line without member."""
        result, _ = pwsh_transform("$a?.")
        # No member name after ?., should be safe
        assert "$a" in result


# ============================================================================
# Corner case: idempotency for all combined transforms
# ============================================================================


class TestFullIdempotency:
    def test_all_operators_together_idempotent(self) -> None:
        code = '''$null_coal = $maybe ?? "fallback"
$ternary = $cond ? "yes" : "no"
$nc_assign ??= "init"
$safe_access = $obj?.Property?.Nested?[0]
Get-Service && Write-Output done || Write-Error failed'''
        first, _ = pwsh_transform(code)
        second, _ = pwsh_transform(first)
        assert first == second

    def test_transform_preserves_critical_semantics(self) -> None:
        """Multiple transforms should yield consistent structure."""
        code = '$x = $a ?? $b ?? $c'
        result, _ = pwsh_transform(code)
        # After transformation: all ?? resolved
        assert "??" not in result
        # Should still reference all three variables
        assert "$a" in result
        assert "$b" in result
        assert "$c" in result
# ============================================================================
# Regression / bug-reproduction tests
# ============================================================================


class TestKnownBugs:
    """Tests that reproduce currently known bugs in pwsh_transform."""

    def test_ternary_with_static_member_access(self) -> None:
        result, _ = pwsh_transform('$x = $cond ? [Math]::PI : 0')
        assert "?" not in result
        assert "[Math]::PI" in result
        assert "if ($cond)" in result

    def test_ternary_with_colon_in_string(self) -> None:
        result, _ = pwsh_transform('$x = $cond ? "a:b" : "c"')
        assert "?" not in result
        assert '"a:b"' in result
        assert '"c"' in result
        assert "if ($cond)" in result

    def test_null_coalescing_with_hash_in_string(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? "default#value"')
        assert "??" not in result
        assert '"default#value"' in result

    def test_null_coalescing_with_comma_in_string(self) -> None:
        result, _ = pwsh_transform('$x = $a ?? "a,b"')
        assert "??" not in result
        assert '"a,b"' in result

    def test_null_conditional_method_with_paren_in_string_arg(self) -> None:
        result, _ = pwsh_transform('$obj?.Foo("a)")')
        assert "?." not in result
        assert 'Foo("a)")' in result

    def test_null_conditional_bracket_with_bracket_in_string_index(self) -> None:
        result, _ = pwsh_transform('$arr?["key]"]')
        assert "?[" not in result
        assert '["key]"]' in result

    def test_backtick_continuation_inside_comment(self) -> None:
        code = '# comment `\nWrite-Output hello'
        result, _ = pwsh_transform(code)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[1] == "Write-Output hello"

    def test_dollar_question_as_ternary_condition(self) -> None:
        result, _ = pwsh_transform('$? ? "yes" : "no"')
        assert result == 'if ($?) { "yes" } else { "no" }'

    def test_command_followed_by_ternary_without_parens(self):
        result, _ = pwsh_transform('Write-Output $a ? $b : $c')
        # Current behaviour incorrectly treats Write-Output $a as the condition
        condition = result.split("if (")[1].split(")")[0]
        assert "Write-Output $a" not in condition


# ============================================================================
# Infinite-loop safety tests
# ============================================================================


class TestNoInfiniteLoops:
    """Inputs that previously caused hangs or look pathological."""

    def test_bracket_then_dot_no_hang(self) -> None:
        result, _ = pwsh_transform("$a?[0]?.Name")
        assert isinstance(result, str)

    def test_long_null_conditional_chain_no_hang(self) -> None:
        result, _ = pwsh_transform("$a?.Items?[0]?.LastName")
        assert isinstance(result, str)

    def test_incomplete_null_coalescing_no_hang(self) -> None:
        result, _ = pwsh_transform("$a ??")
        assert isinstance(result, str)

    def test_incomplete_null_conditional_dot_no_hang(self) -> None:
        result, _ = pwsh_transform("$a?.")
        assert isinstance(result, str)

    def test_trailing_and_operator_no_hang(self) -> None:
        result, _ = pwsh_transform("cmd1 &&")
        assert isinstance(result, str)

    def test_bare_question_marks_no_hang(self) -> None:
        result, _ = pwsh_transform("?.?.?.?")
        assert isinstance(result, str)

    def test_many_nested_ternaries_no_hang(self) -> None:
        code = '$a ? ($b ? ($c ? ($d ? 1 : 2) : 3) : 4) : 5'
        result, _ = pwsh_transform(code)
        assert isinstance(result, str)

    def test_backtick_rain_no_hang(self) -> None:
        code = "Write-Output `\n`\n`\nhello"
        result, _ = pwsh_transform(code)
        assert isinstance(result, str)


# ============================================================================
# Additional bug-reproduction tests discovered during deep analysis
# ============================================================================


class TestAdditionalBugs:
    """Further edge-case bugs found by studying _find_string_regions and depth handling."""

    def test_here_string_false_positive_consumes_rest_of_file(self) -> None:
        code = "$x = @'foo'@\nWrite-Output hello && cmd2"
        result, _ = pwsh_transform(code)
        # The second line should have its && transformed, but because the
        # here-string scanner swallows to EOF, it is left untouched.
        assert "&&" not in result

    def test_at_quote_inside_line_not_here_string(self) -> None:
        code = "$text = @' preserved ?? and ?. '\nWrite-Output hello && cmd2"
        result, _ = pwsh_transform(code)
        assert "&&" not in result

    def test_chain_inside_script_block(self):
        result, _ = pwsh_transform("$sb = { cmd1 && cmd2 }")
        assert "&&" not in result

    def test_chain_inside_subexpression(self):
        result, _ = pwsh_transform("$(cmd1 && cmd2)")
        assert "&&" not in result
