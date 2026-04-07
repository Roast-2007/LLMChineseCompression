"""Tests for code generation mode."""

import pytest

from zippedtext.codegen import (
    CODEGEN_SENTINEL,
    CodegenManifest,
    CodegenSegment,
    apply_codegen,
    restore_codegen,
    safe_eval,
)


class TestSafeEval:
    def test_simple_string(self):
        assert safe_eval("'hello'") == "hello"

    def test_string_multiply(self):
        assert safe_eval("'ab' * 5") == "ababababab"

    def test_join_range(self):
        assert safe_eval("''.join(str(i) for i in range(5))") == "01234"

    def test_chr_expression(self):
        assert safe_eval("chr(65)") == "A"

    def test_complex_expression(self):
        result = safe_eval("''.join(chr(i) for i in range(65, 70))")
        assert result == "ABCDE"

    def test_rejects_import(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_eval("__import__('os')")

    def test_rejects_dunder(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_eval("''.__class__")

    def test_rejects_exec(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_eval("exec('print(1)')")

    def test_rejects_open(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_eval("open('/etc/passwd')")

    def test_rejects_os(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_eval("os.system('ls')")

    def test_timeout(self):
        # An expression that would take too long
        with pytest.raises(ValueError):
            safe_eval("'x' * (10**100)")


class TestCodegenManifest:
    def test_serialize_roundtrip(self):
        segments = (
            CodegenSegment(start=0, end=10, code="'a' * 10", output="a" * 10),
            CodegenSegment(start=20, end=30, code="'b' * 10", output="b" * 10),
        )
        manifest = CodegenManifest(segments=segments)
        data = manifest.serialize()
        restored = CodegenManifest.deserialize(data)
        assert len(restored.segments) == 2
        assert restored.segments[0].output == "a" * 10
        assert restored.segments[1].output == "b" * 10

    def test_empty_manifest(self):
        manifest = CodegenManifest(segments=())
        data = manifest.serialize()
        restored = CodegenManifest.deserialize(data)
        assert restored.segments == ()


class TestApplyRestore:
    def test_roundtrip(self):
        text = "hello" + "x" * 20 + "world"
        segments = (
            CodegenSegment(start=5, end=25, code="'x' * 20", output="x" * 20),
        )
        manifest = CodegenManifest(segments=segments)

        modified = apply_codegen(text, manifest)
        assert CODEGEN_SENTINEL in modified
        assert len(modified) < len(text)

        restored = restore_codegen(modified, manifest)
        assert restored == text

    def test_multiple_segments(self):
        text = "aaa" + "b" * 10 + "ccc" + "d" * 10 + "eee"
        segments = (
            CodegenSegment(start=3, end=13, code="'b' * 10", output="b" * 10),
            CodegenSegment(start=16, end=26, code="'d' * 10", output="d" * 10),
        )
        manifest = CodegenManifest(segments=segments)

        modified = apply_codegen(text, manifest)
        restored = restore_codegen(modified, manifest)
        assert restored == text

    def test_no_segments(self):
        text = "hello world"
        manifest = CodegenManifest(segments=())
        assert apply_codegen(text, manifest) == text
        assert restore_codegen(text, manifest) == text


class TestCodegenCompressDecompress:
    """End-to-end codegen mode test using mock API analysis."""

    def test_manual_codegen_roundtrip(self):
        """Test the full pipeline with pre-built manifest (no LLM needed)."""
        from zippedtext.codegen import CodegenManifest, CodegenSegment
        from zippedtext.encoder import encode
        from zippedtext.decoder import decode
        from zippedtext.codegen import apply_codegen, restore_codegen

        # Simulate what codegen compress does
        text = "prefix" + "a" * 100 + "suffix"
        segments = (
            CodegenSegment(start=6, end=106, code="'a' * 100", output="a" * 100),
        )
        manifest = CodegenManifest(segments=segments)

        modified = apply_codegen(text, manifest)
        compressed_body = encode(modified)
        restored_modified = decode(compressed_body, len(modified))
        restored = restore_codegen(restored_modified, manifest)
        assert restored == text
