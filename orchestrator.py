"""
PE Trends Intelligence Pipeline — Orchestrator
================================================

Modular orchestrator with per-stage flags and a canonical-filename bridge.

Each data-collection connector writes its own DATED output (in different
locations — Companies House writes to the project root, others to data/).
This orchestrator runs the connectors as subprocesses, then copies each
connector's NEWEST output to a stable CANONICAL name in data/ so the next
stage can find it without manual renaming. Analysis stages read the canonical
names.

Usage examples:
    python orchestrator.py --collect            # CH + news + reed + trends
    python orchestrator.py --companies-house    # just the CH scan
    python orchestrator.py --filter             # mid-market FTSE tagging
    python orchestrator.py --fca                # FCA validation
    python orchestrator.py --triangulate        # corpus triangulation
    python orchestrator.py --triangulate-live   # + targeted live pass
    python orchestrator.py --trends             # google trends
    python orchestrator.py --all                # full end-to-end
    python orchestrator.py --analyse            # filter + fca + triangulate

Run with no flags to see this help.

NOTE: this is a pragmatic bridge. The connectors have inconsistent interfaces
(dated filenames, different output dirs, no path arguments). Rather than
rewrite four working connectors near a deadline, the orchestrator papers over
this by locating and copying outputs. FUTURE WORK: give every connector's
run_*() a consistent input_path/output_path argument and drop the bridge.
"""

import os
import sys
import glob
import time
import shutil
import logging
import subprocess
from datetime import date, datetime

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CONNECTORS = os.path.join(ROOT, "connectors")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(ROOT, "orchestrator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PY = sys.executable  # use the same interpreter / venv that launched us

# Canonical names each stage reads/writes (stable, undated).
CANON = {
    "candidates": os.path.join(DATA, "pe_acquisitions_latest.csv"),
    "guardian":   os.path.join(DATA, "guardian_latest.csv"),
    "gnews":      os.path.join(DATA, "google_news_latest.csv"),
    "reed":       os.path.join(DATA, "reed_latest.csv"),
    "annotated":  os.path.join(DATA, "candidates_annotated_latest.csv"),
}

# For each connector: the script, and the glob(s) of its dated output (it may
# land in root or data/, so we search both).
CONNECTOR_OUTPUTS = {
    "companies_house": ["pe_acquisitions_*.csv", "data/pe_acquisitions_*.csv"],
    "guardian":        ["data/guardian_articles_*.csv", "guardian_articles_*.csv"],
    "google_news":     ["data/google_news_*.csv", "google_news_*.csv"],
    "reed":            ["data/reed_jobs_*.csv", "reed_jobs_*.csv"],
}

# ── Helpers ────────────────────────────────────────────────────────────────
def _print_header(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

def run_script(script_name, extra_args=None):
    """
    Run a connector script as a subprocess with the project root as CWD (so its
    relative 'data/...' writes and '.env' load behave exactly as standalone).
    Returns True on exit code 0, False otherwise. Never raises.
    """
    path = os.path.join(CONNECTORS, script_name)
    if not os.path.exists(path):
        print(f"  ✗ {script_name} not found in connectors/ — skipping")
        logging.error(f"{script_name} not found")
        return False
    cmd = [PY, path] + (extra_args or [])
    print(f"  → running {script_name} {' '.join(extra_args or [])}".rstrip())
    t0 = time.time()
    try:
        # Stream output through so the user sees progress live.
        result = subprocess.run(cmd, cwd=ROOT)
        ok = result.returncode == 0
        dt = time.time() - t0
        status = "OK" if ok else f"FAILED (exit {result.returncode})"
        print(f"  {('✓' if ok else '✗')} {script_name} — {status} ({dt:.0f}s)")
        logging.info(f"{script_name} {status} in {dt:.0f}s")
        return ok
    except Exception as e:
        print(f"  ✗ {script_name} raised: {e}")
        logging.error(f"{script_name} raised: {e}")
        return False

def newest_matching(patterns):
    """Return the most-recently-modified file across the given glob patterns
    (searched relative to ROOT), or None."""
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(ROOT, pat)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def bridge(connector_key, canon_key):
    """
    Copy a connector's newest dated output to its canonical name. Returns the
    canonical path on success, None if no output was found.
    """
    newest = newest_matching(CONNECTOR_OUTPUTS[connector_key])
    if not newest:
        print(f"  ! no output found for {connector_key} "
              f"(looked for {CONNECTOR_OUTPUTS[connector_key]})")
        logging.warning(f"No output to bridge for {connector_key}")
        return None
    dest = CANON[canon_key]
    shutil.copy2(newest, dest)
    print(f"  ↳ bridged {os.path.basename(newest)} → {os.path.basename(dest)}")
    logging.info(f"bridged {newest} -> {dest}")
    return dest

# ── Stages ─────────────────────────────────────────────────────────────────
def stage_companies_house():
    _print_header("STAGE: Companies House (PE detection)")
    ok = run_script("companies_house.py")
    if ok:
        bridge("companies_house", "candidates")
    return ok

def stage_news():
    _print_header("STAGE: News collection (Guardian + Google News)")
    g = run_script("guardian.py")
    if g:
        bridge("guardian", "guardian")
    n = run_script("google_news.py")
    if n:
        bridge("google_news", "gnews")
    return g or n

def stage_reed():
    _print_header("STAGE: Reed job postings")
    ok = run_script("reed.py")
    if ok:
        bridge("reed", "reed")
    return ok

def stage_trends():
    _print_header("STAGE: Google Trends (sector momentum)")
    return run_script("google_trends.py")

def stage_filter():
    _print_header("STAGE: Mid-market filter (FTSE 350 subsidiary tagging)")
    # midmarket_filter auto-finds newest pe_acquisitions*.csv in data/.
    # Ensure the canonical candidates file is present for it to read.
    if not os.path.exists(CANON["candidates"]):
        newest = newest_matching(CONNECTOR_OUTPUTS["companies_house"])
        if newest:
            shutil.copy2(newest, CANON["candidates"])
            print(f"  ↳ seeded {os.path.basename(CANON['candidates'])} "
                  f"from {os.path.basename(newest)}")
        else:
            print("  ✗ no candidates file available — run --companies-house first")
            return False
    return run_script("midmarket_filter.py")

def stage_fca():
    _print_header("STAGE: FCA Register validation")
    if not os.path.exists(CANON["candidates"]):
        print("  ✗ no candidates file — run --companies-house first")
        return False
    # fca.py reads data/pe_acquisitions.csv by default; point it at canonical.
    return run_script("fca.py")

def stage_triangulate(live=False):
    _print_header("STAGE: Signal triangulation"
                  + (" (+ targeted live pass)" if live else ""))
    if not os.path.exists(CANON["candidates"]):
        print("  ✗ no candidates file — run --companies-house first")
        return False
    args = ["--targeted"] if live else []
    return run_script("triangulate.py", args)

# ── Orchestration ──────────────────────────────────────────────────────────
def main(argv):
    flags = set(argv)

    if not flags or "--help" in flags or "-h" in flags:
        print(__doc__)
        return

    start = datetime.now()
    _print_header(f"PE PIPELINE ORCHESTRATOR — {start.isoformat(timespec='seconds')}")
    print(f"Flags: {' '.join(sorted(flags))}")

    results = {}

    # Expand convenience groups.
    do = lambda *names: any(f in flags for f in names)
    collect = do("--collect", "--all")
    analyse = do("--analyse", "--all")

    # --- Collection stages ---
    if collect or do("--companies-house"):
        results["companies_house"] = stage_companies_house()
    if collect or do("--news"):
        results["news"] = stage_news()
    if collect or do("--reed"):
        results["reed"] = stage_reed()
    if collect or do("--trends"):
        results["trends"] = stage_trends()

    # --- Analysis stages (order matters) ---
    if analyse or do("--filter"):
        results["filter"] = stage_filter()
    if analyse or do("--fca"):
        results["fca"] = stage_fca()
    if analyse or do("--triangulate"):
        results["triangulate"] = stage_triangulate(live=False)
    if do("--triangulate-live"):
        results["triangulate_live"] = stage_triangulate(live=True)

    # --- Summary ---
    _print_header("ORCHESTRATION SUMMARY")
    if not results:
        print("No recognised stage flags. Run --help for options.")
        return
    for stage, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {stage}")
    dur = (datetime.now() - start).total_seconds()
    print(f"\nTotal time: {dur:.0f}s")
    logging.info(f"Run complete: {results} in {dur:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1:])