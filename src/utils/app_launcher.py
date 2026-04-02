from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

from security.secrets import validate_live_llm_config
from utils.config import get_settings


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8501


@dataclass(frozen=True)
class LaunchPlan:
    repo_root: Path
    python_executable: str
    api_host: str
    api_port: int
    web_host: str
    web_port: int
    launch_api: bool
    api_reload: bool
    headless: bool

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def web_url(self) -> str:
        return f"http://{self.web_host}:{self.web_port}"


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_available_port(host: str, preferred: int, *, attempts: int = 25) -> int:
    for port in range(preferred, preferred + attempts):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"No available port found starting at {preferred} on {host}.")


def build_subprocess_env(repo_root: Path, *, api_url: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(repo_root / "src"), str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if api_url:
        env["PRIORI_API_BASE_URL"] = api_url
    return env


def build_api_command(plan: LaunchPlan) -> list[str]:
    command = [
        plan.python_executable,
        "-m",
        "uvicorn",
        "apps.api.main:app",
        "--host",
        plan.api_host,
        "--port",
        str(plan.api_port),
    ]
    if plan.api_reload:
        command.append("--reload")
    return command


def build_streamlit_command(plan: LaunchPlan) -> list[str]:
    return [
        plan.python_executable,
        "-m",
        "streamlit",
        "run",
        str(plan.repo_root / "apps" / "research_console" / "app.py"),
        "--server.address",
        plan.web_host,
        "--server.port",
        str(plan.web_port),
        "--server.headless",
        "true" if plan.headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]


def wait_for_api(api_url: str, *, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{api_url}/api/health"
    last_error: str | None = None
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(health_url)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:  # pragma: no cover - exercised through launcher smoke path
                last_error = str(exc)
            time.sleep(0.4)
    raise RuntimeError(f"API did not become ready at {health_url}. Last error: {last_error or 'unknown'}")


def run_status_probe(plan: LaunchPlan) -> dict[str, object]:
    env = build_subprocess_env(plan.repo_root, api_url=plan.api_url if plan.launch_api else None)
    result = subprocess.run(
        [plan.python_executable, "-m", "eval.experiment_cli", "status"],
        cwd=plan.repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "PRIORI-X status probe failed.")

    settings = get_settings()
    secret_status = validate_live_llm_config(settings)
    return {
        "status_output": result.stdout.strip(),
        "provider_ready": secret_status.provider_ready,
        "allow_live_llm": secret_status.allow_live_llm,
        "missing_required_secrets": secret_status.missing_required_secrets,
    }


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the integrated PRIORI-X workbench from a normal terminal.")
    parser.add_argument("--no-api", action="store_true", help="Launch only the Streamlit workbench.")
    parser.add_argument("--api-host", default=DEFAULT_API_HOST)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--web-host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--reload", action="store_true", help="Run the API with Uvicorn reload enabled.")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit without trying to open a browser.")
    parser.add_argument("--check", action="store_true", help="Validate the integrated startup path and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[2]
    plan = LaunchPlan(
        repo_root=repo_root,
        python_executable=sys.executable,
        api_host=args.api_host,
        api_port=choose_available_port(args.api_host, args.api_port) if not args.no_api else args.api_port,
        web_host=args.web_host,
        web_port=choose_available_port(args.web_host, args.web_port),
        launch_api=not args.no_api,
        api_reload=args.reload,
        headless=args.headless,
    )

    probe = run_status_probe(plan)
    print(f"PRIORI-X status probe passed. Provider ready: {probe['provider_ready']}.")
    print(f"Research app URL: {plan.web_url}")
    if plan.launch_api:
        print(f"API URL: {plan.api_url}")
    if args.check:
        print(probe["status_output"])
        return 0

    env = build_subprocess_env(plan.repo_root, api_url=plan.api_url if plan.launch_api else None)
    api_process: subprocess.Popen[str] | None = None
    try:
        if plan.launch_api:
            api_process = subprocess.Popen(
                build_api_command(plan),
                cwd=plan.repo_root,
                env=env,
            )
            wait_for_api(plan.api_url)
        streamlit_result = subprocess.run(
            build_streamlit_command(plan),
            cwd=plan.repo_root,
            env=env,
            check=False,
        )
        return streamlit_result.returncode
    finally:
        if api_process is not None:
            _terminate_process(api_process)
