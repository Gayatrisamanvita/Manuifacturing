"""
Master Launcher for Manufacturing & Supplier Intelligence API (FactorySmart AI™).

Usage:
    python run_api.py                   # Launches Unified API on http://localhost:8000
    python run_api.py --port 8001       # Custom port
    python run_api.py --mode supplier   # Launches Standalone Supplier Microservice on 8001
"""

import os
import sys
import argparse

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BANNER = r"""
================================================================================
  FACTORYSMART AI (TM) - MANUFACTURING & SUPPLIER INTELLIGENCE SUITE
================================================================================
  REST API & Machine Learning Model Serving Backend
  Tiers Supported:
    [Tier 1] Manufacturing Issue Level (Failure Mode & SHAP Explanations)
    [Tier 2] Raw Material Level (Marginal Lot Risk, Downstream Forecast & Offsets)
    [Tier 3] Supplier / Vendor Level (Trends, Genealogy Trace, Gap, 8D SCAR)
================================================================================
"""

def print_server_info(host: str, port: int, mode: str):
    print(BANNER)
    base_url = f"http://{'localhost' if host in ['0.0.0.0', '127.0.0.1'] else host}:{port}"
    print(f"[*] API Service Mode     : {mode.upper()}")
    print(f"[*] Server Listening on  : {base_url}")
    print(f"[*] Interactive Docs     : {base_url}/docs  (Swagger UI)")
    print(f"[*] Alternative Docs     : {base_url}/redoc (ReDoc)")
    print(f"[*] System Health Check  : {base_url}/health")
    print(f"[*] Web Dashboard        : {base_url}/")
    print("-" * 80)
    print("Active Authentication Keys for Testing:")
    print("  - Admin Master Key     : fs-admin-master-2026-99x")
    print("  - Vendor B Key         : fs-key-vendor-b-3e7a9d0f")
    print("  - Vendor A Key         : fs-key-vendor-a-8f4b2c1d")
    print("  - Demo Partner Key     : fs-demo-partner-quality-2026")
    print("-" * 80)
    print("Press CTRL+C to stop the server.\n")


def main():
    parser = argparse.ArgumentParser(description="FactorySmart AI API Server Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Bind network interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8000 for unified, 8001 for supplier)")
    parser.add_argument("--mode", choices=["unified", "supplier"], default="unified",
                        help="API mode: 'unified' (default) or 'supplier'")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload on code changes")
    args = parser.parse_args()

    env_port = int(os.environ.get("PORT")) if os.environ.get("PORT") else None
    default_port = 8001 if args.mode == "supplier" else (env_port or 8000)
    port = args.port if args.port is not None else default_port

    print_server_info(args.host, port, args.mode)

    import uvicorn
    if args.mode == "supplier":
        target_app = "src.supplier_api:app"
    else:
        target_app = "src.api:app"

    uvicorn.run(target_app, host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
