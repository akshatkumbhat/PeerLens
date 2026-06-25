"""PeerLens command-line entry point.

Phase 1 wires: ingest -> load -> smoke query. Subcommands are added as each
layer lands; for now it exposes `config` so `uv run peerlens config` confirms
the install and resolved paths.
"""

from __future__ import annotations

import argparse

from peerlens import config


def _cmd_ingest(args: argparse.Namespace) -> int:
    from peerlens.ingest.pull import pull_phase1

    paths = pull_phase1(year=args.year, overwrite=args.overwrite)
    print(f"Ingested IPEDS year {args.year or config.get_settings().ipeds_year}:")
    for topic, path in paths.items():
        import polars as pl

        n = pl.read_parquet(path).height
        print(f"  {topic:24} {n:>7,} rows  ->  {path.relative_to(config.REPO_ROOT)}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from peerlens.warehouse.build import build_warehouse

    counts = build_warehouse(year=args.year, cohort_size=args.cohort_size)
    print(f"Built warehouse at {config.WAREHOUSE_DB.relative_to(config.REPO_ROOT)}:")
    for table, n in counts.items():
        print(f"  {table:24} {n:>6,} rows")
    return 0


def _cmd_peers(args: argparse.Namespace) -> int:
    from peerlens.peers.build import build_peer_sets

    summary = build_peer_sets(k=args.k, n_bands=args.n_bands)
    print("Built bridge_peer_set:")
    print(f"  targets:        {summary['n_targets']}")
    print(f"  peer rows:      {summary['peer_rows']:,}")
    print(f"  aspirant rows:  {summary['aspirant_rows']:,}")
    print(f"  k={summary['k']}  bands={summary['n_bands']}  "
          f"diagonal_fallback={summary['used_diagonal_fallback']}")
    return 0


def _cmd_validate(_args: argparse.Namespace) -> int:
    from peerlens.quality.checks import run_checks
    from peerlens.warehouse import db

    con = db.connect(read_only=True)
    try:
        results = run_checks(con)
    finally:
        con.close()
    print("Data-quality checks:")
    failed = 0
    for r in results:
        mark = "ok  " if r.passed else "FAIL"
        if not r.passed:
            failed += 1
        print(f"  [{mark}] {r.name:36} {r.n_violations} violation(s)")
    return 1 if failed else 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from peerlens.agent.model import ModelError, get_plan_model
    from peerlens.agent.pipeline import run_agent
    from peerlens.warehouse import db

    try:
        model = get_plan_model()
    except ModelError as e:
        print(f"Cannot run the agent: {e}")
        print("Set GEMINI_API_KEY in .env (free key: https://aistudio.google.com/apikey).")
        return 1

    con = db.connect(read_only=True)
    try:
        resp = run_agent(con, model, args.question)
    except ModelError as e:
        print(f"Model error: {e}")
        return 1
    finally:
        con.close()

    print(f"Q: {resp.question}\n")
    if resp.answered:
        print(resp.answer)
        rp = resp.resolved_plan
        print(f"\n  confidence: {resp.agreement:.0%} agreement over {resp.n_samples} samples")
        print(f"  plan:       {rp.intent.value} · {rp.metric} · {rp.target_name}")
        print(f"  sql:        {resp.sql}")
    else:
        ab = resp.abstention
        print(f"[abstained — {ab.reason.value}] {ab.message}")
        if ab.options:
            print("  options:", ", ".join(ab.options[:6]))
        print(f"  agreement:  {resp.agreement:.0%} over {resp.n_samples} samples")
    return 0


def _cmd_app(_args: argparse.Namespace) -> int:
    import subprocess
    import sys

    app_path = config.REPO_ROOT / "src" / "peerlens" / "app" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


def _cmd_config(_args: argparse.Namespace) -> int:
    s = config.get_settings()
    print("PeerLens config")
    print(f"  repo root:     {config.REPO_ROOT}")
    print(f"  raw dir:       {config.RAW_DIR}")
    print(f"  warehouse db:  {config.WAREHOUSE_DB}")
    print(f"  ipeds year:    {s.ipeds_year}")
    print(f"  model provider:{s.peerlens_model_provider}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="peerlens", description="PeerLens CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="show resolved config and paths")
    p_config.set_defaults(func=_cmd_config)

    p_ingest = sub.add_parser("ingest", help="pull Phase 1 IPEDS topics to parquet")
    p_ingest.add_argument("--year", type=int, default=None, help="IPEDS year (default: settings)")
    p_ingest.add_argument("--overwrite", action="store_true", help="re-pull even if cached")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_build = sub.add_parser("build", help="build the DuckDB warehouse from cached parquet")
    p_build.add_argument("--year", type=int, default=None, help="IPEDS year (default: settings)")
    p_build.add_argument("--cohort-size", type=int, default=200, help="institutions in the cohort")
    p_build.set_defaults(func=_cmd_build)

    p_peers = sub.add_parser("peers", help="build Mahalanobis peer/aspirant sets")
    p_peers.add_argument("--k", type=int, default=10, help="neighbors per set")
    p_peers.add_argument("--n-bands", type=int, default=5, help="selectivity bands")
    p_peers.set_defaults(func=_cmd_peers)

    p_validate = sub.add_parser("validate", help="run data-quality checks on the warehouse")
    p_validate.set_defaults(func=_cmd_validate)

    p_ask = sub.add_parser("ask", help="ask the grounded agent a question (needs GEMINI_API_KEY)")
    p_ask.add_argument("question", help="natural-language question")
    p_ask.set_defaults(func=_cmd_ask)

    p_app = sub.add_parser("app", help="launch the Streamlit page")
    p_app.set_defaults(func=_cmd_app)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
