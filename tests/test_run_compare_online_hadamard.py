import os
import subprocess
from pathlib import Path


SPINQUANT_DIR = Path(__file__).resolve().parents[1]


def test_dry_run_isolates_modes_and_passes_cli_option():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ONLINE_HAD_MODES"] = "factorized zero_padding"
    result = subprocess.run(
        ["bash", "scripts/run_compare_online_hadamard.sh"],
        cwd=SPINQUANT_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout
    assert "--online_had_mode factorized" in output
    assert "--online_had_mode zero_padding" in output
    assert "online-had-comparison/factorized/llama2-7b.json" in output
    assert "online-had-comparison/zero_padding/llama2-7b.json" in output
