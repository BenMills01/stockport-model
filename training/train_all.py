"""Orchestrate training for all unfitted ML models.

Run from the project root:

    python -m training.train_all

or import and call programmatically:

    from training.train_all import train_all_models
    results = train_all_models()

Each model saves its artifact to ``data/`` (see model-level PROJECTION_MODEL_PATH
etc.) and returns a summary dict.  When the DB has insufficient data to build a
meaningful training set, the function logs a warning and skips that model rather
than crashing — so this script is safe to run at any stage of the data build.

proxy_xg requires an external StatsBomb-derived CSV file.  Pass its path via
``statsbomb_path`` (programmatic) or ``--statsbomb-path`` (CLI).  When not
provided the model is skipped with a clear status message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from training.build_training_data import (
    build_availability_training_data,
    build_championship_projection_training_data,
    build_financial_value_training_data,
)
from models.championship_projection import train_projection_model
from models.availability_risk import train_availability_model
from models.financial_value import train_value_model
from models.proxy_xg import train_proxy_xg


_LOG = logging.getLogger(__name__)

# Minimum training rows before we attempt to fit.  Below this, the model will
# over-fit badly and the heuristic fallback is more reliable.
_MIN_ROWS_PROJECTION = 30
_MIN_ROWS_AVAILABILITY = 50
_MIN_ROWS_FINANCIAL = 20


def train_all_models(
    *,
    dry_run: bool = False,
    statsbomb_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build training data and fit all models.

    Parameters
    ----------
    dry_run:
        If True, build data and log statistics but do not fit or save anything.
    statsbomb_path:
        Path to a StatsBomb-derived shot CSV for the proxy_xg model.  When
        omitted or the file does not exist, proxy_xg training is skipped.

    Returns
    -------
    Dict with one key per model, each containing either the training summary
    or an error/skip message.
    """
    results: dict[str, Any] = {}

    # ── 1. Championship projection ────────────────────────────────────────
    _LOG.info("Building championship projection training data…")
    try:
        proj_df = build_championship_projection_training_data()
        _LOG.info("  → %d training rows", len(proj_df))
        if len(proj_df) < _MIN_ROWS_PROJECTION:
            _LOG.warning(
                "  Skipping championship_projection: only %d rows (need %d).",
                len(proj_df),
                _MIN_ROWS_PROJECTION,
            )
            results["championship_projection"] = {
                "status": "skipped",
                "reason": f"Only {len(proj_df)} rows (need {_MIN_ROWS_PROJECTION})",
                "rows_available": len(proj_df),
            }
        elif dry_run:
            results["championship_projection"] = {
                "status": "dry_run",
                "rows_available": len(proj_df),
                "target_distribution": proj_df["target_starter"].value_counts().to_dict()
                if "target_starter" in proj_df.columns else {},
            }
        else:
            summary = train_projection_model(proj_df)
            results["championship_projection"] = {"status": "ok", **summary}
            _LOG.info("  ✓ Saved championship projection model.")
    except Exception as exc:
        _LOG.exception("championship_projection training failed")
        results["championship_projection"] = {"status": "error", "message": str(exc)}

    # ── 2. Availability risk ──────────────────────────────────────────────
    _LOG.info("Building availability risk training data…")
    try:
        avail_df = build_availability_training_data()
        _LOG.info("  → %d training rows", len(avail_df))
        if len(avail_df) < _MIN_ROWS_AVAILABILITY:
            _LOG.warning(
                "  Skipping availability_risk: only %d rows (need %d).",
                len(avail_df),
                _MIN_ROWS_AVAILABILITY,
            )
            results["availability_risk"] = {
                "status": "skipped",
                "reason": f"Only {len(avail_df)} rows (need {_MIN_ROWS_AVAILABILITY})",
                "rows_available": len(avail_df),
            }
        elif dry_run:
            results["availability_risk"] = {
                "status": "dry_run",
                "rows_available": len(avail_df),
                "target_distribution": avail_df["target_available_75pct"].value_counts().to_dict()
                if "target_available_75pct" in avail_df.columns else {},
            }
        else:
            train_availability_model(avail_df)
            results["availability_risk"] = {
                "status": "ok",
                "rows_trained": len(avail_df),
                "positive_rate": float(avail_df["target_available_75pct"].mean())
                if "target_available_75pct" in avail_df.columns else None,
            }
            _LOG.info("  ✓ Saved availability risk model.")
    except Exception as exc:
        _LOG.exception("availability_risk training failed")
        results["availability_risk"] = {"status": "error", "message": str(exc)}

    # ── 3. Financial value ────────────────────────────────────────────────
    _LOG.info("Building financial value training data…")
    try:
        value_df = build_financial_value_training_data()
        _LOG.info("  → %d training rows", len(value_df))
        if len(value_df) < _MIN_ROWS_FINANCIAL:
            _LOG.warning(
                "  Skipping financial_value: only %d rows (need %d).",
                len(value_df),
                _MIN_ROWS_FINANCIAL,
            )
            results["financial_value"] = {
                "status": "skipped",
                "reason": f"Only {len(value_df)} rows (need {_MIN_ROWS_FINANCIAL})",
                "rows_available": len(value_df),
            }
        elif dry_run:
            results["financial_value"] = {
                "status": "dry_run",
                "rows_available": len(value_df),
                "fee_paid_summary": value_df["fee_paid"].describe().to_dict()
                if "fee_paid" in value_df.columns else {},
            }
        else:
            train_value_model(value_df)
            results["financial_value"] = {
                "status": "ok",
                "rows_trained": len(value_df),
                "fee_paid_median": float(value_df["fee_paid"].median())
                if "fee_paid" in value_df.columns else None,
            }
            _LOG.info("  ✓ Saved financial value model.")
    except Exception as exc:
        _LOG.exception("financial_value training failed")
        results["financial_value"] = {"status": "error", "message": str(exc)}

    # ── 4. Proxy xG ───────────────────────────────────────────────────────
    _LOG.info("Checking proxy xG training data…")
    sb_path = Path(statsbomb_path) if statsbomb_path is not None else None
    if sb_path is None or not sb_path.exists():
        reason = (
            "No --statsbomb-path provided"
            if sb_path is None
            else f"File not found: {sb_path}"
        )
        _LOG.warning("  Skipping proxy_xg: %s.", reason)
        results["proxy_xg"] = {"status": "skipped", "reason": reason}
    elif dry_run:
        results["proxy_xg"] = {"status": "dry_run", "statsbomb_path": str(sb_path)}
    else:
        try:
            train_proxy_xg(str(sb_path))
            results["proxy_xg"] = {"status": "ok", "statsbomb_path": str(sb_path)}
            _LOG.info("  ✓ Saved proxy xG model.")
        except Exception as exc:
            _LOG.exception("proxy_xg training failed")
            results["proxy_xg"] = {"status": "error", "message": str(exc)}

    return results


def main() -> int:
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train all Stockport recruitment ML models.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data and log statistics but do not fit or save models.",
    )
    parser.add_argument(
        "--statsbomb-path",
        default=None,
        metavar="CSV",
        help="Path to StatsBomb-derived shot CSV for proxy_xg training.",
    )
    args = parser.parse_args()

    results = train_all_models(dry_run=args.dry_run, statsbomb_path=args.statsbomb_path)
    print(json.dumps(results, indent=2, default=str))

    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    n_skip = sum(1 for r in results.values() if r.get("status") == "skipped")
    n_err = sum(1 for r in results.values() if r.get("status") == "error")
    print(f"\nDone — {n_ok} fitted, {n_skip} skipped (insufficient data), {n_err} errors.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
