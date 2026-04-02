from __future__ import annotations

import socket
from pathlib import Path

from utils.app_launcher import (
    LaunchPlan,
    build_api_command,
    build_streamlit_command,
    build_subprocess_env,
    choose_available_port,
)


def _plan(tmp_path: Path) -> LaunchPlan:
    return LaunchPlan(
        repo_root=tmp_path,
        python_executable="python",
        api_host="127.0.0.1",
        api_port=8000,
        web_host="127.0.0.1",
        web_port=8501,
        launch_api=True,
        api_reload=False,
        headless=False,
    )


def test_build_subprocess_env_sets_pythonpath_and_api_url(tmp_path: Path) -> None:
    env = build_subprocess_env(tmp_path, api_url="http://127.0.0.1:8000")

    assert str(tmp_path / "src") in env["PYTHONPATH"]
    assert str(tmp_path) in env["PYTHONPATH"]
    assert env["PRIORI_API_BASE_URL"] == "http://127.0.0.1:8000"


def test_build_api_command_includes_expected_target(tmp_path: Path) -> None:
    command = build_api_command(_plan(tmp_path))

    assert command[:4] == ["python", "-m", "uvicorn", "apps.api.main:app"]
    assert "--port" in command
    assert "8000" in command


def test_build_streamlit_command_includes_expected_app_path(tmp_path: Path) -> None:
    command = build_streamlit_command(_plan(tmp_path))

    assert command[:3] == ["python", "-m", "streamlit"]
    assert str(tmp_path / "apps" / "research_console" / "app.py") in command
    assert "--server.port" in command
    assert "8501" in command


def test_choose_available_port_skips_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        resolved_port = choose_available_port("127.0.0.1", busy_port, attempts=5)

    assert resolved_port >= busy_port
    assert resolved_port != busy_port
