"""Tests for the CLI."""

from click.testing import CliRunner
from infrawatch.cli import cli


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_demo(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--points", "100", "--anomalies", "3"])
        assert result.exit_code == 0
        assert "InfraWatch Demo" in result.output
        assert "anomalies detected" in result.output.lower() or "Anomalies detected" in result.output

    def test_collect_csv(self, sample_csv_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["collect", str(sample_csv_path)])
        assert result.exit_code == 0
        assert "Collected" in result.output

    def test_detect_csv(self, sample_csv_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["detect", str(sample_csv_path)])
        assert result.exit_code == 0
