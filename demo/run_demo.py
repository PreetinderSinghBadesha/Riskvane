"""Interactive Demo Runner for Riskvane.

Can run directly via HTTP against the running Riskvane API (http://localhost:8000)
or standalone in-process if the server is not started yet.

Usage:
    python demo/run_demo.py
    python demo/run_demo.py --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request, error

BASE_DIR = Path(__file__).resolve().parent
CLEAN_VM_PATH = BASE_DIR / "sample_clean_vm.json"
VULN_VM_PATH = BASE_DIR / "sample_vulnerable_vm.json"
BRUTE_FORCE_PATH = BASE_DIR / "sample_brute_force_events.json"
PORT_SCAN_PATH = BASE_DIR / "sample_port_scan_events.json"


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def http_post(url: str, data: dict | list) -> tuple[int, dict | list]:
    payload = json.dumps(data).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def http_get(url: str) -> tuple[int, dict | list]:
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def run_api_demo(base_url: str) -> None:
    print(f"[*] Connecting to Riskvane API at: {base_url}")
    try:
        status, _ = http_get(f"{base_url}/alerts?vm_id=healthcheck")
    except Exception as e:
        print(f"[!] Unable to connect to {base_url}. Error: {e}")
        print("    Please start the server first in another terminal:")
        print("        riskvane-api")
        print("    or run standalone mode: python demo/run_demo.py --standalone")
        sys.exit(1)

    print("[+] API is online and responding!\n")

    # Step 1: Clean VM Scan
    print_banner("Demo Step 1: Posture Scan on a Clean, Hardened VM")
    clean_data = json.loads(CLEAN_VM_PATH.read_text(encoding="utf-8"))
    print(f"Scanning VM: '{clean_data['vm_id']}'...")
    status, res = http_post(f"{base_url}/scan", clean_data)
    print(f"HTTP Status: {status}")
    print(f"-> Risk Score: {res['score']}/100")
    print(f"-> Severity Breakdown: {res['severity_breakdown']}")
    print(f"-> Total Findings: {len(res['findings'])}")
    if not res["findings"]:
        print("   [Clean! No CVE vulnerabilities or hardening issues detected]")

    input("\n[Press Enter to proceed to Step 2: Vulnerable VM Scan...]")

    # Step 2: Vulnerable VM Scan (Rule-Based)
    print_banner("Demo Step 2: Posture Scan on a Vulnerable VM (Rule-Based)")
    vuln_data = json.loads(VULN_VM_PATH.read_text(encoding="utf-8"))
    print(f"Scanning VM: '{vuln_data['vm_id']}'...")
    status, res = http_post(f"{base_url}/scan", vuln_data)
    print(f"HTTP Status: {status}")
    print(f"-> Risk Score: {res['score']}/100")
    print(f"-> Severity Breakdown: {res['severity_breakdown']}")
    print("-> Key Findings Detected:")
    for f in res["findings"]:
        sev = f["severity"].upper()
        print(f"   [{sev:8}] {f['title']}")
        print(f"              Evidence: {f.get('evidence')}")

    input("\n[Press Enter to proceed to Step 3: ML-Based Risk Scoring + SHAP...]")

    # Step 3: Vulnerable VM Scan with ML & TreeSHAP
    print_banner("Demo Step 3: ML Scoring & Explainability (XGBoost + TreeSHAP)")
    print(f"Scanning VM: '{vuln_data['vm_id']}' with ?use_model=true...")
    status, res = http_post(f"{base_url}/scan?use_model=true", vuln_data)
    print(f"HTTP Status: {status}")
    print(f"-> ML Model Risk Score: {res['score']}/100")
    print("-> TreeSHAP Feature Contributions (Why this score was assigned):")
    if res.get("score_explanation"):
        for feat, val in res["score_explanation"].items():
            sign = "+" if val >= 0 else ""
            print(f"   {feat:32}: {sign}{val:.2f}")
    else:
        print("   (Model explanation not available; ensure model is trained via riskvane-train-model)")

    input("\n[Press Enter to proceed to Step 4: Live Attack Detection (SSH Brute Force)...]")

    # Step 4: SSH Brute Force Detection
    print_banner("Demo Step 4: Live Attack Stream - SSH Brute Force Detection")
    bf_events = json.loads(BRUTE_FORCE_PATH.read_text(encoding="utf-8"))
    print(f"Ingesting {len(bf_events)} auth failure events for '{vuln_data['vm_id']}'...")
    status, fired_alerts = http_post(f"{base_url}/events", bf_events)
    print(f"HTTP Status: {status}")
    print(f"-> Alerts Triggered Immediately: {len(fired_alerts)}")
    for a in fired_alerts:
        print(f"   ALERT: [{a['severity'].upper()}] {a['alert_type']}")
        print(f"          Attacker IP : {a['source_ip']}")
        print(f"          Evidence    : {a['evidence']}")
        print(f"          Event IDs   : {len(a['raw_event_ids'])} correlated events")

    input("\n[Press Enter to proceed to Step 5: Port Scan Detection...]")

    # Step 5: Port Scan Detection
    print_banner("Demo Step 5: Live Attack Stream - Port Scan Detection")
    ps_events = json.loads(PORT_SCAN_PATH.read_text(encoding="utf-8"))
    print(f"Ingesting {len(ps_events)} firewall connection events across 10 ports...")
    status, fired_alerts = http_post(f"{base_url}/events", ps_events)
    print(f"HTTP Status: {status}")
    print(f"-> Alerts Triggered Immediately: {len(fired_alerts)}")
    for a in fired_alerts:
        print(f"   ALERT: [{a['severity'].upper()}] {a['alert_type']}")
        print(f"          Attacker IP : {a['source_ip']}")
        print(f"          Evidence    : {a['evidence']}")

    input("\n[Press Enter to proceed to Step 6: Querying All VM Alerts...]")

    # Step 6: Query Alerts by VM
    print_banner("Demo Step 6: Query All Active Alerts for VM")
    print(f"GET /alerts?vm_id={vuln_data['vm_id']}")
    status, all_alerts = http_get(f"{base_url}/alerts?vm_id={vuln_data['vm_id']}")
    print(f"HTTP Status: {status}")
    print(f"-> Total Active Alerts for '{vuln_data['vm_id']}': {len(all_alerts)}")
    for i, a in enumerate(all_alerts, 1):
        print(f"   {i}. [{a['severity'].upper()}] {a['alert_type']} | Source: {a['source_ip']} | {a['evidence']}")

    print_banner("Demo Completed Successfully!")
    print("Summary of Capabilities Demonstrated:")
    print("  1. Deterministic Posture Risk Scoring (CVE + Hardening Rules)")
    print("  2. Machine Learning Tabular Regressor (XGBoost) + TreeSHAP Explainability")
    print("  3. Sliding-Window Real-time Attack Detection (SSH Brute Force & Port Scans)")
    print("  4. Unified HTTP API with zero orchestrator lock-in")
    print("=" * 70 + "\n")


def run_standalone_demo() -> None:
    print_banner("Running Demo in Standalone Mode (Direct Python Modules)")
    from riskvane.common.schema import PostureFacts
    from riskvane.posture.scoring import scorer
    from riskvane.posture.training.features import featurize
    from riskvane.detection.engine import DetectionEngine
    from riskvane.common.events import Event

    # Clean VM
    print("\n--- 1. Clean VM Posture Scan ---")
    clean_facts = PostureFacts.model_validate_json(CLEAN_VM_PATH.read_text(encoding="utf-8"))
    clean_result = scorer.score(clean_facts)
    print(f"VM: {clean_facts.vm_id} -> Score: {clean_result.score}/100, Findings: {len(clean_result.findings)}")

    # Vuln VM
    print("\n--- 2. Vulnerable VM Posture Scan ---")
    vuln_facts = PostureFacts.model_validate_json(VULN_VM_PATH.read_text(encoding="utf-8"))
    vuln_result = scorer.score(vuln_facts)
    print(f"VM: {vuln_facts.vm_id} -> Score: {vuln_result.score}/100, Findings: {len(vuln_result.findings)}")
    for f in vuln_result.findings:
        print(f"   [{f.severity.upper()}] {f.title}")

    # ML & SHAP
    print("\n--- 3. ML Risk Model & TreeSHAP ---")
    try:
        from riskvane.posture.ml.inference import RiskModel
        model = RiskModel.load()
        ml_score, shap = model.predict(featurize(vuln_facts))
        print(f"ML Predicted Score: {ml_score}/100")
        print("SHAP feature contributions:")
        for k, v in shap.items():
            print(f"   {k:30}: {v:+.2f}")
    except Exception as e:
        print(f"Model not loaded: {e}")

    # Attack Detection
    print("\n--- 4. Live Attack Detection (SSH Brute Force) ---")
    engine = DetectionEngine()
    bf_raw = json.loads(BRUTE_FORCE_PATH.read_text(encoding="utf-8"))
    for item in bf_raw:
        ev = Event.model_validate(item)
        alerts = engine.process_event(ev)
        for a in alerts:
            print(f"   ALERT FIRED: [{a.severity.upper()}] {a.alert_type} from {a.source_ip} ({a.evidence})")

    print("\nStandalone demo finished successfully!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Riskvane Demo Runner")
    parser.add_argument("--api", default="http://localhost:8000", help="Riskvane API Base URL")
    parser.add_argument("--standalone", action="store_true", help="Run without HTTP server using internal Python modules")
    args = parser.parse_args()

    if args.standalone:
        run_standalone_demo()
    else:
        run_api_demo(args.api)


if __name__ == "__main__":
    main()
