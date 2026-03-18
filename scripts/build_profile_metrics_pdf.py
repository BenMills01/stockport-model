#!/usr/bin/env python3
"""Generate a PDF reference sheet of all on-pitch profile metrics and physical weightings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES_JSON = ROOT / "config" / "on_pitch_profiles.json"
PHYSICAL_JSON = ROOT / "config" / "on_pitch_physical_profiles.json"
OUT_HTML = ROOT / "artifacts" / "profile_metrics_reference.html"
OUT_PDF = ROOT / "artifacts" / "profile_metrics_reference.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

METRIC_LABELS = {
    # API-Football metrics
    "goals_scored": "Goals Scored",
    "assists": "Assists",
    "shots_total": "Shots Total",
    "shots_on_target": "Shots on Target",
    "passes_total": "Passes Total",
    "pass_accuracy": "Pass Accuracy %",
    "passes_key": "Key Passes",
    "dribbles_success": "Dribbles Success",
    "dribbles_attempts": "Dribble Attempts",
    "duels_won": "Duels Won",
    "duels_total": "Duels Total",
    "tackles_total": "Tackles Total",
    "tackles_interceptions": "Interceptions",
    "fouls_drawn": "Fouls Drawn",
    "fouls_committed": "Fouls Committed",
    "saves": "Saves",
    "minutes": "Minutes",
    # Wyscout volume metrics (per-90)
    "xg": "xG per 90 ★",
    "xa": "xA per 90 ★",
    "non_penalty_goals": "NP Goals per 90 ★",
    "progressive_passes": "Progressive Passes ★",
    "deep_completions": "Deep Completions ★",
    "shot_assists": "Shot Assists ★",
    "passes_to_final_third": "Passes to Final 3rd ★",
    "progressive_runs": "Progressive Runs ★",
    "touches_in_box": "Touches in Box ★",
    "crosses": "Crosses ★",
    "aerial_duels": "Aerial Duels Vol ★",
    "padj_interceptions": "PAdj Interceptions ★",
    "padj_tackles": "PAdj Tackles ★",
    "ball_recoveries": "Ball Recoveries ★",
    "successful_exits": "GK Exits ★",
    # Wyscout ratio metrics (percentages)
    "aerial_duels_won_pct": "Aerial Duel Win % ★",
    "defensive_duels_won_pct": "Def Duel Win % ★",
    "offensive_duels_won_pct": "Off Duel Win % ★",
    "long_pass_accuracy": "Long Pass Accuracy % ★",
    "save_pct": "Save Rate % ★",
}

PHYSICAL_LABELS = {
    "sc_physical_dist_per_match": "Total Distance",
    "sc_physical_hsr_dist_per_match": "HSR Distance",
    "sc_physical_sprint_dist_per_match": "Sprint Distance",
    "sc_physical_count_hsr_per_match": "HSR Count",
    "sc_physical_count_sprint_per_match": "Sprint Count",
    "sc_physical_count_high_accel_per_match": "High Accel Count",
    "sc_physical_count_high_decel_per_match": "High Decel Count",
    "sc_physical_top_speed_per_match": "Top Speed",
    "sc_physical_intensity_index": "Intensity Index",
}

GI_LABELS = {
    "sc_pressure_ball_retention_ratio_under_high_pressure": "Pressure Ball Retention",
    "sc_pressure_pass_completion_ratio_under_high_pressure": "Pressure Pass Completion",
    "sc_passes_count_completed_pass_to_run_in_behind_per_match": "Pass to Run in Behind",
    "sc_passes_opportunity_take_rate": "Opportunity Take Rate",
}

FAMILY_ORDER = [
    "striker",
    "wide_attacker_winger",
    "central_midfielder",
    "full_back_wing_back",
    "centre_back",
    "goalkeeper",
]

FAMILY_LABELS = {
    "striker": "Strikers",
    "wide_attacker_winger": "Wide Attackers",
    "central_midfielder": "Central Midfielders",
    "full_back_wing_back": "Full Backs / Wing Backs",
    "centre_back": "Centre Backs",
    "goalkeeper": "Goalkeeper",
}


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


def bar(v: float, colour: str) -> str:
    w = round(v * 100)
    return f'<div class="bar-wrap"><div class="bar" style="width:{w}%;background:{colour}"></div><span class="bar-label">{w}%</span></div>'


def build_html(profiles: list, physical: dict) -> str:
    # Group by family in order
    by_family: dict[str, list] = {f: [] for f in FAMILY_ORDER}
    for p in profiles:
        fam = p.get("role_family", "")
        if fam in by_family:
            by_family[fam].append(p)

    sections = []
    for fam in FAMILY_ORDER:
        group = by_family[fam]
        if not group:
            continue
        cards = []
        for p in group:
            phys_key = p.get("physical_profile", "")
            phys = physical.get(phys_key, {})
            phys_weights = phys.get("physical_weights", {})
            gi_weights = phys.get("gi_weights", {})
            phys_sub = phys.get("physical_sub_weight", 0)
            gi_sub = phys.get("gi_sub_weight", 0)

            # Technical metrics rows
            tech_rows = "".join(
                f'<tr><td>{METRIC_LABELS.get(k, k)}</td><td>{bar(v, "#1a3a6b")}</td></tr>'
                for k, v in sorted(p["metrics"].items(), key=lambda x: -x[1])
            )

            # Physical output rows
            phys_rows = "".join(
                f'<tr><td>{PHYSICAL_LABELS.get(k, k)}</td><td>{bar(v, "#2e7d32")}</td></tr>'
                for k, v in sorted(phys_weights.items(), key=lambda x: -x[1])
            )

            # GI rows
            gi_rows = "".join(
                f'<tr><td>{GI_LABELS.get(k, k)}</td><td>{bar(v, "#7b1fa2")}</td></tr>'
                for k, v in sorted(gi_weights.items(), key=lambda x: -x[1])
            )

            profile_points_html = "".join(
                f'<span class="tag">{pt}</span>' for pt in p.get("profile_points", [])
            )

            phys_label = phys.get("label", phys_key)

            cards.append(f"""
<div class="card">
  <div class="card-header">
    <span class="role-label">{p["label"]}</span>
    <span class="phys-badge">{phys_label}</span>
  </div>
  <div class="profile-points">{profile_points_html}</div>
  <div class="two-col">
    <div class="col">
      <h4>Technical Metrics</h4>
      <table>{tech_rows}</table>
    </div>
    <div class="col">
      <h4>Physical Output <span class="sub-weight">({round(phys_sub*100)}% of physical score)</span></h4>
      <table>{phys_rows}</table>
      <h4 style="margin-top:10px">Game Intelligence <span class="sub-weight">({round(gi_sub*100)}% of physical score)</span></h4>
      <table>{gi_rows}</table>
    </div>
  </div>
</div>""")

        sections.append(f"""
<div class="family-section">
  <h2 class="family-heading">{FAMILY_LABELS[fam]}</h2>
  {''.join(cards)}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stockport County · On-Pitch Profile Metrics Reference</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10px; color: #1a1a2e; background: #fff; }}
  .cover {{ background: #1a3a6b; color: #fff; padding: 36px 40px 28px; margin-bottom: 32px; }}
  .cover h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
  .cover p {{ font-size: 11px; margin-top: 6px; opacity: 0.75; }}
  .family-section {{ padding: 0 32px; margin-bottom: 24px; page-break-inside: avoid; }}
  h2.family-heading {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
    color: #1a3a6b; border-bottom: 2px solid #1a3a6b; padding-bottom: 4px; margin-bottom: 14px; }}
  .card {{ border: 1px solid #dde; border-radius: 6px; padding: 14px 16px; margin-bottom: 16px; page-break-inside: avoid; }}
  .card-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .role-label {{ font-size: 18px; font-weight: 800; color: #1a3a6b; }}
  .phys-badge {{ font-size: 9px; font-weight: 600; background: #e8f0fe; color: #1a3a6b; padding: 2px 8px; border-radius: 10px; }}
  .profile-points {{ margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .tag {{ background: #f0f4ff; color: #1a3a6b; border: 1px solid #c5d0f0; font-size: 8.5px; padding: 2px 7px; border-radius: 10px; font-weight: 500; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .col h4 {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #555; margin-bottom: 5px; }}
  .sub-weight {{ font-weight: 400; text-transform: none; letter-spacing: 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  table tr {{ border-bottom: 1px solid #f0f0f0; }}
  table td {{ padding: 3px 0; vertical-align: middle; }}
  table td:first-child {{ width: 45%; font-size: 9px; color: #333; padding-right: 8px; }}
  table td:last-child {{ width: 55%; }}
  .bar-wrap {{ display: flex; align-items: center; gap: 5px; }}
  .bar {{ height: 8px; border-radius: 2px; min-width: 2px; }}
  .bar-label {{ font-size: 8.5px; color: #666; white-space: nowrap; }}
  @media print {{
    body {{ font-size: 9px; }}
    .family-section {{ padding: 0 20px; }}
    .card {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="cover">
  <h1>Stockport County · On-Pitch Profile Metrics Reference</h1>
  <p>Technical metric weightings (per-90, league adjusted) and physical/GI template weightings for all 17 recruitment profiles.</p>
</div>
{''.join(sections)}
</body>
</html>"""


def main() -> None:
    profiles = json.loads(PROFILES_JSON.read_text())
    physical = json.loads(PHYSICAL_JSON.read_text())

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(profiles, physical)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML written → {OUT_HTML}")

    result = subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={OUT_PDF}",
            "--print-to-pdf-no-header",
            str(OUT_HTML),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Chrome error:", result.stderr)
        raise SystemExit(1)
    print(f"PDF written  → {OUT_PDF}")


if __name__ == "__main__":
    main()
