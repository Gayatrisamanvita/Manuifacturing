"""
Master Launcher for Manufacturing & Supplier Intelligence Suite (FactorySmart AI™).

Usage:
    python run_api.py                   # Launches Streamlit UI on http://localhost:8501 & opens browser
                                        # (also runs FastAPI backend on port 8000)
    python run_api.py --mode streamlit  # Launches only the Streamlit Dashboard
    python run_api.py --mode api        # Launches only the FastAPI Backend on port 8000
    python run_api.py --mode supplier   # Launches Standalone Supplier Microservice on port 8001
"""

import os
import sys
import time
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


def print_server_info(streamlit_port: int, api_port: int, mode: str):
    print(BANNER)
    print(f"[*] Launcher Mode              : {mode.upper()}")
    if mode in ["all", "streamlit"]:
        print(f"[*] Streamlit Quality Dashboard: http://localhost:{streamlit_port}")
        print(f"[*] Auto-opening Web Browser   : http://localhost:{streamlit_port}")
    if mode in ["all", "api", "unified"]:
        print(f"[*] FastAPI Backend Server     : http://localhost:{api_port}")
        print(f"[*] Interactive Swagger Docs   : http://localhost:{api_port}/docs")
    if mode == "supplier":
        print(f"[*] Supplier Microservice      : http://localhost:{api_port}")
    print("-" * 80)
    print("Press CTRL+C to stop all services.\n")


def open_browser_delayed(url: str, delay: float = 2.0):
    if os.environ.get("CODESPACES") == "true":
        return  # In GitHub Codespaces, port forwarding automatically opens the browser
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[*] Browser auto-launch notice: {e}")


def run_streamlit(host: str, port: int):
    from streamlit.web import cli as stcli
    app_script = os.path.join(PROJECT_ROOT, "streamlit_app.py")
    is_codespace = os.environ.get("CODESPACES") == "true" or not sys.platform.startswith("win")
    sys.argv = [
        "streamlit",
        "run",
        app_script,
        f"--server.port={port}",
        f"--server.address={host}",
        f"--server.headless={'true' if is_codespace else 'false'}",
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

    api_port = 8001 if args.mode == "supplier" else args.api_port
    print_server_info(args.port, api_port, args.mode)

    api_process = None

    try:
        # 1. Start FastAPI backend subprocess if running 'all', 'api', or 'supplier'
        if args.mode in ["all", "api", "unified"]:
            target_app = "src.api:app"
            api_cmd = [sys.executable, "-m", "uvicorn", target_app, "--host", args.host, "--port", str(api_port)]
            if args.mode == "all":
                # Run API in background subprocess
                api_process = subprocess.Popen(api_cmd)
                time.sleep(1.5)
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
                    args=(f"http://localhost:{args.port}", 2.0),
                    daemon=True
                ).start()
            run_streamlit(args.host, args.port)

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
