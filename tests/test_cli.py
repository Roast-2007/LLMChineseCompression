from click.testing import CliRunner

from zippedtext.cli import main


class _FakeStructuredClient:
    def __init__(self) -> None:
        self.model = "fake-structured-model"
        self.last_model_id = "fake-structured-model"

    def analyze_text(self, text: str):
        from zippedtext.online_manifest import AnalysisManifest

        return AnalysisManifest.from_api_payload(
            {
                "char_frequencies": {"版": 0.3, "本": 0.2, "接": 0.1},
                "phrase_dictionary": ["structured online", "api endpoint"],
                "template_hints": ["key_value"],
                "field_schemas": [
                    {"field": "endpoint", "slot_type": "path_or_url"},
                ],
                "slot_hints": [
                    {"template_kind": "key_value", "slot_index": 0, "slot_type": "path_or_url", "field": "endpoint"},
                ],
            },
            len(text),
        )

    def generate_continuation(self, context: str, max_tokens: int = 200, max_top_logprobs: int = 20):
        from zippedtext.api_client import ChunkResult, GeneratedToken

        token = GeneratedToken(
            text="x",
            logprob=-0.1,
            top_alternatives=[("x", -0.1)],
            char_offset=0,
        )
        return ChunkResult(generated_text="x", tokens=[token], model=self.model)


def test_cli_bench_reports_structured_and_raw_diagnostics(tmp_path, monkeypatch):
    runner = CliRunner()
    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("zippedtext.cli._make_api_client", lambda cfg: _FakeStructuredClient())

    bench_result = runner.invoke(main, ["bench", str(input_path), "--api-key", "test-key"])
    assert bench_result.exit_code == 0
    assert "zippedtext online (structured)" in bench_result.output
    assert "structured raw diagnostic" in bench_result.output
