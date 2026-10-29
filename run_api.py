"""
Master Launcher for Manufacturing & Supplier Intelligence Suite (FactorySmart AI™).

Usage:
    python run_api.py                   # Launches Streamlit UI on port 8501 & opens browser
                                        # (also runs FastAPI backend on port 8000)
    python run_api.py --mode streamlit  # Launches only the Streamlit Dashboard
    python run_api.py --mode api        # Launches only the FastAPI Backend on port 8000
    python run_api.py --mode supplier   # Launches Standalone Supplier Microservice on port 8001
"""

import os
import sys
import time
import socket
import shutil
import argparse
import threading
import subprocess
import webbrowser

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BANNER = r"""
================================================================================
  FACTORYSMART AI (TM) - MANUFACTURING & SUPPLIER INTELLIGENCE SUITE
================================================================================
  Full-Stack Operational Suite:
    - Streamlit Executive Quality Dashboard  : http://localhost:8501
    - FastAPI REST Inference Backend         : http://localhost:8000
    - Interactive Swagger API Documentation  : http://localhost:8000/docs
    - System Health & Diagnostics            : http://localhost:8000/health
================================================================================
  Tiers Active:
    [Tier 1] Supplier & Vendor Level (30-Day Trends, Defect Leakage, 8D SCAR)
    [Tier 2] Raw Materials & Inventory (199-Batch Master, 216 HV Leakage, Offsets)
    [Tier 3] Manufacturing Level (Station ST-10 to ST-40 Risk, Spindle Telemetry)
    [Tier 4] Production Level (Lines & Orders, Planned vs Actual, Bottlenecks)
================================================================================
"""


def is_codespace_environment() -> bool:
    """Return True if running in a GitHub Codespaces container."""
    return os.environ.get("CODESPACES") == "true" or "CODESPACE_NAME" in os.environ


def get_codespace_url(port: int) -> str | None:
    """Construct the public HTTPS URL for a forwarded port in GitHub Codespaces."""
    codespace_name = os.environ.get("CODESPACE_NAME")
    if codespace_name:
        domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
        return f"https://{codespace_name}-{port}.{domain}"
    return None


def print_server_info(streamlit_port: int, api_port: int, mode: str):
    print(BANNER)
    in_codespace = is_codespace_environment()
    st_codespace_url = get_codespace_url(streamlit_port)
    api_codespace_url = get_codespace_url(api_port)

    print(f"[*] Launcher Mode              : {mode.upper()}")
    if in_codespace:
        print(f"[*] Environment                : GitHub Codespaces Detected")

    if mode in ["all", "streamlit"]:
        print(f"[*] Streamlit Quality Dashboard: http://localhost:{streamlit_port}")
        if st_codespace_url:
            print(f"[*] Codespaces Public Dashboard: {st_codespace_url}")
        print(f"[*] Auto-opening Web Browser   : {st_codespace_url or f'http://localhost:{streamlit_port}'}")

    if mode in ["all", "api", "unified"]:
        print(f"[*] FastAPI Backend Server     : http://localhost:{api_port}")
        if api_codespace_url:
            print(f"[*] Codespaces Public API      : {api_codespace_url}")
            print(f"[*] Interactive Swagger Docs   : {api_codespace_url}/docs")
        else:
            print(f"[*] Interactive Swagger Docs   : http://localhost:{api_port}/docs")

    if mode == "supplier":
        print(f"[*] Supplier Microservice      : http://localhost:{api_port}")
        if api_codespace_url:
            print(f"[*] Codespaces Public URL      : {api_codespace_url}")

    print("-" * 80)
    print("Press CTRL+C to stop all services.\n")


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 15.0) -> bool:
    """Wait until the specified TCP port is accepting connections."""
    start_time = time.time()
    connect_host = "127.0.0.1" if host in ["0.0.0.0", "::"] else host
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((connect_host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.25)
    return False


def open_browser_delayed(port: int, host: str = "127.0.0.1", delay: float = 0.5):
    """
    Waits for the Streamlit server to bind to the port, then opens the browser.
    Works seamlessly in both local environments and GitHub Codespaces.
    """
    # 1. Wait until Streamlit is actively listening on the port
    wait_for_port(port, host=host, timeout=15.0)
    time.sleep(delay)

    in_codespace = is_codespace_environment()
    codespace_url = get_codespace_url(port)
    local_url = f"http://localhost:{port}"

    primary_url = codespace_url if (in_codespace and codespace_url) else local_url
    fallback_url = local_url if primary_url != local_url else None

    print("\n" + "=" * 76)
    print(f"  FACTORYSMART AI: STREAMLIT DASHBOARD ACTIVE ON PORT {port}")
    if codespace_url:
        print(f"  -> Codespaces Public URL : {codespace_url}")
    print(f"  -> Local Container URL   : {local_url}")
    print("=" * 76)

    # In GitHub Codespaces, ensure port visibility is set to public via gh CLI
    if in_codespace:
        try:
            subprocess.run(
                ["gh", "codespace", "ports", "visibility", f"{port}:public"],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass

    opened = False

    # Attempt 1: Standard Python webbrowser.open (works locally and via VS Code Remote shim)
    try:
        opened = webbrowser.open(primary_url, new=2)
    except Exception as e:
        print(f"[*] webbrowser.open notice: {e}")

    # Attempt 2: If primary was codespace_url and returned False, try local_url
    if not opened and fallback_url:
        try:
            opened = webbrowser.open(fallback_url, new=2)
        except Exception:
            pass

    # Attempt 3: On Linux / Codespaces, invoke xdg-open directly (VS Code remote hooks this)
    if not opened and sys.platform.startswith("linux"):
        for target in [primary_url, fallback_url]:
            if not target:
                continue
            try:
                proc = subprocess.Popen(
                    ["xdg-open", target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc.wait(timeout=2)
                if proc.returncode == 0:
                    opened = True
                    break
            except Exception:
                pass

    # Attempt 4: Try VS Code CLI 'code --open-url' if installed in Codespace
    if not opened:
        code_bin = shutil.which("code")
        if code_bin:
            try:
                subprocess.Popen(
                    [code_bin, "--open-url", primary_url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                opened = True
            except Exception:
                pass

    if opened:
        print(f"[*] Browser auto-launch triggered for: {primary_url}\n")
    else:
        print(f"[*] Open the following URL in your browser to view the dashboard:\n    -> {primary_url}\n")


def run_streamlit(host: str, port: int):
    from streamlit.web import cli as stcli
    app_script = os.path.join(PROJECT_ROOT, "streamlit_app.py")
    in_codespace = is_codespace_environment()

    sys.argv = [
        "streamlit",
        "run",
        app_script,
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        f"--server.headless={'true' if in_codespace or not sys.platform.startswith('win') else 'false'}",
        "--browser.gatherUsageStats=false"
    ]
    stcli.main()


def main():
    parser = argparse.ArgumentParser(description="FactorySmart AI Master Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Bind network interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit Dashboard port (default: 8501)")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI backend port (default: 8000)")
    parser.add_argument("--mode", choices=["all", "streamlit", "api", "unified", "supplier"], default="all",
                        help="Launch mode: 'all' (Streamlit + FastAPI, default), 'streamlit', 'api', or 'supplier'")
    parser.add_argument("--no-browser", action="store_true", default=False, help="Do not automatically open web browser")
    args = parser.parse_args()

    # Dynamic port resolution from environment variables (Codespaces / Cloud containers)
    env_port = int(os.environ["PORT"]) if "PORT" in os.environ and os.environ["PORT"].isdigit() else None
    env_st_port = int(os.environ["STREAMLIT_SERVER_PORT"]) if "STREAMLIT_SERVER_PORT" in os.environ and os.environ["STREAMLIT_SERVER_PORT"].isdigit() else None

    streamlit_port = env_st_port or args.port
    api_port = 8001 if args.mode == "supplier" else (env_port or args.api_port)

    # Prevent port collision when launching both services in 'all' or 'unified' mode
    if args.mode in ["all", "unified"] and streamlit_port == api_port:
        if streamlit_port == 8000:
            # If user ran with port 8000, keep FastAPI on 8000 and Streamlit on 8501
            print("[*] Note: Port 8000 assigned to FastAPI backend; launching Streamlit Dashboard on port 8501.")
            streamlit_port = 8501
            api_port = 8000
        else:
            api_port = 8000 if streamlit_port != 8000 else 8001
            print(f"[*] Note: Adjusted FastAPI port to {api_port} to prevent conflict with Streamlit on {streamlit_port}.")

    print_server_info(streamlit_port, api_port, args.mode)

    api_process = None

    try:
        # 1. Start FastAPI backend subprocess if running 'all', 'api', or 'supplier'
        if args.mode in ["all", "api", "unified"]:
            target_app = "src.api:app"
            api_cmd = [sys.executable, "-m", "uvicorn", target_app, "--host", args.host, "--port", str(api_port)]
            if args.mode in ["all", "unified"]:
                # Run API in background subprocess
                api_process = subprocess.Popen(api_cmd)
                time.sleep(1.0)
            else:
                # Run API directly in foreground
                import uvicorn
                uvicorn.run(target_app, host=args.host, port=api_port)
                return

        elif args.mode == "supplier":
            import uvicorn
            uvicorn.run("src.supplier_api:app", host=args.host, port=api_port)
            return

        # 2. Launch Streamlit (for 'all' and 'streamlit' modes)
        if args.mode in ["all", "streamlit"]:
            if not args.no_browser:
                threading.Thread(
                    target=open_browser_delayed,
                    args=(streamlit_port, args.host, 0.5),
                    daemon=True
                ).start()
            run_streamlit(args.host, streamlit_port)

    except KeyboardInterrupt:
        print("\n[*] Stopping FactorySmart AI services...")
    finally:
        if api_process is not None:
            try:
                api_process.terminate()
                api_process.wait(timeout=3)
            except Exception:
                pass
            print("[*] FastAPI service stopped cleanly.")


if __name__ == "__main__":
    main()
