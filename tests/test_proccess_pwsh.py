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
        result = pwsh_transform('$x = $cond ? "a" : "b"')
        assert "if ($cond)" in result
        assert '{ "a" }' in result
        assert '{ "b" }' in result

    def test_ternary_with_comparison(self) -> None:
        result = pwsh_transform("$x = $a -gt 5 ? $a : 0")
        assert "if ($a -gt 5)" in result
        assert "{ $a }" in result
        assert "{ 0 }" in result

    def test_ternary_in_assignment(self) -> None:
        result = pwsh_transform('$status = $count -eq 0 ? "empty" : "non-empty"')
        assert "$status = " in result
        assert "($count -eq 0)" in result

    def test_ternary_with_function_calls(self) -> None:
        result = pwsh_transform('$x = Test-Path $p ? (Get-Item $p) : $null')
        assert "if (Test-Path $p)" in result
        assert "(Get-Item $p)" in result
        assert "$null" in result

    def test_ternary_no_assignment(self) -> None:
        result = pwsh_transform('$cond ? "yes" : "no"')
        assert 'if ($cond) { "yes" } else { "no" }' in result


# ============================================================================
# Null-coalescing  (??)
# ============================================================================


class TestNullCoalescing:
    def test_simple_null_coalescing(self) -> None:
        result = pwsh_transform('$x = $a ?? "default"')
        assert "if ($null -ne $a)" in result
        assert '{ $a }' in result
        assert '{ "default" }' in result

    def test_null_coalescing_with_variable(self) -> None:
        result = pwsh_transform("$x = $a ?? $b")
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $b }" in result

    def test_null_coalescing_with_literal_default(self) -> None:
        result = pwsh_transform("$path = $env:HOME ?? 'C:\\Users\\Default'")
        assert "if ($null -ne $env:HOME)" in result
        assert "{ $env:HOME }" in result

    def test_nested_null_coalescing(self) -> None:
        result = pwsh_transform('$x = $a ?? $b ?? "default"')
        # After first ?? transform, the result contains another ??
        # which should also be transformed
        assert "default" in result
        assert "if ($null -ne " in result

    def test_null_coalescing_no_assignment(self) -> None:
        result = pwsh_transform('$a ?? "fallback"')
        assert 'if ($null -ne $a) { $a } else { "fallback" }' in result


# ============================================================================
# Null-coalescing assignment  (??=)
# ============================================================================


class TestNullCoalescingAssignment:
    def test_simple_assign(self) -> None:
        result = pwsh_transform('$a ??= "default"')
        assert "if ($null -eq $a)" in result
        assert '$a = "default"' in result

    def test_assign_with_expression(self) -> None:
        result = pwsh_transform("$count ??= (Get-ChildItem).Count")
        assert "if ($null -eq $count)" in result
        assert "$count = (Get-ChildItem).Count" in result

    def test_assign_does_not_conflict_with_null_coalescing(self) -> None:
        """??= should be transformed before ?? so ??= is not partially matched."""
        result = pwsh_transform("$a ??= $b ?? $c")
        # ??= should be fully resolved
        assert "??=" not in result
        assert "??" not in result


# ============================================================================
# Pipeline chain AND  (&&)
# ============================================================================


class TestPipelineChainAnd:
    def test_simple_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2")
        assert ";" in result
        assert "if ($?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 && cmd3")
        assert "cmd1;" in result
        assert "if ($?) { cmd2; if ($?) { cmd3 } }" in result

    def test_and_chain_with_pipeline(self) -> None:
        result = pwsh_transform("Get-Process | Where-Object CPU && Write-Output done")
        assert "Get-Process | Where-Object CPU" in result
        assert "Write-Output done" in result
        assert "if ($?)" in result


# ============================================================================
# Pipeline chain OR  (||)
# ============================================================================


class TestPipelineChainOr:
    def test_simple_or_chain(self) -> None:
        result = pwsh_transform("cmd1 || cmd2")
        assert ";" in result
        assert "if (-not $?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_or_chain(self) -> None:
        result = pwsh_transform("cmd1 || cmd2 || cmd3")
        assert "cmd1;" in result
        assert "if (-not $?) { cmd2; if (-not $?) { cmd3 } }" in result


# ============================================================================
# Null-conditional  (?. and ?[])
# ============================================================================


class TestNullConditional:
    def test_property_access(self) -> None:
        result = pwsh_transform("$a?.Length")
        assert "if ($null -ne $a) { $a.Length }" in result

    def test_index_access(self) -> None:
        result = pwsh_transform("$a?[0]")
        assert "if ($null -ne $a) { $a[0] }" in result

    def test_chained_null_conditional(self) -> None:
        result = pwsh_transform("$a?.Property?.SubProperty")
        # Both ?. should be transformed
        assert "?." not in result

    def test_null_conditional_with_method(self) -> None:
        result = pwsh_transform("$a?.ToString()")
        assert "if ($null -ne $a) { $a.ToString() }" in result

    def test_null_conditional_assignment(self) -> None:
        result = pwsh_transform("$x = $a?.Length")
        assert "$x = if ($null -ne $a) { $a.Length }" in result


# ============================================================================
# Combined transformations
# ============================================================================


class TestCombinedTransformations:
    def test_multiple_features(self) -> None:
        code = '$x = $a ?? "default"\nGet-Process && Write-Output done'
        result = pwsh_transform(code)
        assert "??" not in result
        assert "&&" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($?)" in result

    def test_no_false_positives_in_strings(self) -> None:
        code = "Write-Output 'The ?? operator is new'"
        result = pwsh_transform(code)
        # The ?? inside the string should not be transformed
        assert "??" in result
        assert "if ($null -ne" not in result

    def test_no_false_positives_in_comments(self) -> None:
        code = "# This ?? is a comment\nWrite-Output hello"
        result = pwsh_transform(code)
        assert "??" in result  # still in comment

    def test_no_false_positives_in_double_quoted_string(self) -> None:
        code = 'Write-Output "The ?? operator"'
        result = pwsh_transform(code)
        assert "??" in result

    def test_combined_and_or(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 || cmd3")
        assert "&&" not in result
        assert "||" not in result


# ============================================================================
# Idempotency
# ============================================================================


class TestIdempotency:
    def test_double_transform_same_result(self) -> None:
        code = '$x = $a ?? "default"\n$y = $cond ? "yes" : "no"\nGet-Process && Write-Output done'
        first = pwsh_transform(code)
        second = pwsh_transform(first)
        assert first == second

    def test_ternary_idempotent(self) -> None:
        code = '$x = $cond ? "a" : "b"'
        first = pwsh_transform(code)
        second = pwsh_transform(first)
        assert first == second

    def test_null_coalescing_idempotent(self) -> None:
        code = '$x = $a ?? "default"'
        first = pwsh_transform(code)
        second = pwsh_transform(first)
        assert first == second

    def test_pipeline_chain_idempotent(self) -> None:
        code = "cmd1 && cmd2"
        first = pwsh_transform(code)
        second = pwsh_transform(first)
        assert first == second

    def test_null_conditional_idempotent(self) -> None:
        code = "$a?.Length"
        first = pwsh_transform(code)
        second = pwsh_transform(first)
        assert first == second


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_strings_with_operators_not_transformed(self) -> None:
        code = """Write-Output 'Use ?? for null-coalescing'
Write-Output "A ? B : C is ternary"
Write-Output 'cmd1 && cmd2 is chain'"""
        result = pwsh_transform(code)
        assert "?? for null-coalescing" in result
        assert "A ? B : C is ternary" in result
        assert "cmd1 && cmd2 is chain" in result

    def test_comments_not_transformed(self) -> None:
        code = """# The ?? operator is new in PS7
# $x = $cond ? "a" : "b"
# cmd1 && cmd2
Write-Output hello"""
        result = pwsh_transform(code)
        assert "The ?? operator" in result
        assert '$cond ? "a" : "b"' in result
        assert "cmd1 && cmd2" in result

    def test_here_string_not_transformed(self) -> None:
        code = """$text = @'
The ?? operator is preserved here.
And so is the ?. operator.
'@
Write-Output $text"""
        result = pwsh_transform(code)
        assert "??" in result  # preserved inside here-string
        assert "?." in result

    def test_multiline_with_backtick(self) -> None:
        code = "Get-Process `\n| Where-Object CPU `\n&& Write-Output done"
        result = pwsh_transform(code)
        assert "&&" not in result
        assert "if ($?)" in result

    def test_empty_code(self) -> None:
        assert pwsh_transform("") == ""

    def test_no_operators(self) -> None:
        code = "Write-Output 'hello world'"
        assert pwsh_transform(code) == code

    def test_ternary_in_pipeline(self) -> None:
        code = "$x = $a ? $b : $c | ForEach-Object { $_ }"
        result = pwsh_transform(code)
        assert "?" not in result
        assert "if ($a)" in result

    def test_null_coalescing_with_property(self) -> None:
        code = '$name = $obj.Name ?? "Unknown"'
        result = pwsh_transform(code)
        assert "if ($null -ne $obj.Name)" in result
        assert "Unknown" in result

    def test_block_comment_not_transformed(self) -> None:
        code = "<# The ?? and ?. operators are new #>\nWrite-Output hello"
        result = pwsh_transform(code)
        assert "??" in result  # preserved in block comment
        assert "?." in result

    def test_null_conditional_bracket_with_expression(self) -> None:
        result = pwsh_transform("$a?[$i + 1]")
        assert "if ($null -ne $a) { $a[$i + 1] }" in result
