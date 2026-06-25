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


def _cmd_eval(args: argparse.Namespace) -> int:
    from peerlens.eval.harness import load_records, run_eval, save_records
    from peerlens.eval.metrics import metrics_at
    from peerlens.eval.report import update_readme, write_report

    out_dir = config.REPO_ROOT / "docs" / "eval"
    records_path = out_dir / "records.json"
    s = config.get_settings()
    updates = {}
    if args.samples:
        updates["agent_samples"] = args.samples
    if args.model:
        updates["gemini_model"] = args.model
    if updates:
        s = s.model_copy(update=updates)

    if args.from_cache:
        if not records_path.exists():
            print(f"No cached records at {records_path}; run without --from-cache first.")
            return 1
        records = load_records(records_path)
        print(f"Loaded {len(records)} cached records.")
    else:
        from peerlens.agent.model import ModelError, get_plan_model
        from peerlens.warehouse import db

        try:
            model = get_plan_model(s)
        except ModelError as e:
            print(f"Cannot run eval: {e}")
            return 1
        con = db.connect(read_only=True)
        try:
            print(f"Running eval (model={s.gemini_model}, samples={s.agent_samples})…")
            records = run_eval(con, model, s, limit=args.limit, pause=args.pause)
        finally:
            con.close()
        if not records:
            print("No records collected (API unavailable?). Nothing written.")
            return 1
        save_records(records, records_path)
        print(f"Saved {len(records)} records -> {records_path.relative_to(config.REPO_ROOT)}")

    op = write_report(records, out_dir, s.agent_tau)
    at_default = metrics_at(records, s.agent_tau)
    print(f"\nOperating point: τ={op.tau:.2f}  coverage={op.coverage:.0%}  "
          f"confident-wrong={op.confident_wrong_rate:.1%}  EX={op.execution_accuracy:.0%}")
    print(f"At default τ={s.agent_tau:.2f}: coverage={at_default.coverage:.0%}  "
          f"confident-wrong={at_default.confident_wrong_rate:.1%}")
    if args.update_readme:
        from peerlens.eval.report import render_markdown
        ok = update_readme(config.REPO_ROOT / "README.md", render_markdown(records, op, s.agent_tau))
        print("README updated." if ok else "README has no EVAL markers — skipped.")
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

    p_eval = sub.add_parser("eval", help="run the evaluation harness and write the report")
    p_eval.add_argument("--samples", type=int, default=None, help="override agent_samples for the eval")
    p_eval.add_argument("--model", type=str, default=None, help="override Gemini model (e.g. gemini-2.5-flash-lite)")
    p_eval.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    p_eval.add_argument("--limit", type=int, default=None, help="limit questions per set")
    p_eval.add_argument("--from-cache", action="store_true", help="recompute from cached records.json")
    p_eval.add_argument("--update-readme", action="store_true", help="write metrics into README markers")
    p_eval.set_defaults(func=_cmd_eval)

    p_app = sub.add_parser("app", help="launch the Streamlit page")
    p_app.set_defaults(func=_cmd_app)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
