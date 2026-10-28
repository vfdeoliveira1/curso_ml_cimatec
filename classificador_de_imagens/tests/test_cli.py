from typer.testing import CliRunner

from classificador_de_imagens.cli import app


def test_cli_lists_training_evaluation_and_prediction_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.exception
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "predict" in result.output
