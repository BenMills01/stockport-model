# Stockport Recruitment Model

Implementation of the March 2026 Stockport County FC recruitment model and delivery guide. The repo now includes the full code path described in the specification pack: ingestion, feature engineering, gates, model layers, scoring, governance, report rendering, and validation.

## Current Status

The codebase currently includes:

- PostgreSQL-first SQLAlchemy schema and session helpers
- API-Football, FBref, Transfermarkt, and Wyscout ingestion modules
- Feature engineering for per-90, rolling form, opposition splits, league adjustment, availability, trajectory, confidence, GBE estimation, and role classification
- Filtering, model, composite scoring, governance, validation, and HTML report output modules
- Unit coverage across config, schema, ingestion, features, gates, models, outputs, and validation

## Local Setup

1. Create a virtual environment and install the project:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
```

2. Create a local `.env` file from the checked-in `.env.example` and fill in your local values.

Required variables:

- `STOCKPORT_DATABASE_URL`
- `API_FOOTBALL_API_KEY` for live API-Football ingestion

3. Initialise the database schema:

```bash
./.venv/bin/python -m db.init_db
```

4. Seed the config-backed reference data:

```bash
./.venv/bin/python -m db.seed_reference_data
```

5. Run the test suite:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

6. Launch the local data viewer in your browser:

```bash
./.venv/bin/python -m viewer.app --open-browser
```

On macOS you can also double-click `launch_stockport_viewer.command` to open it without typing commands.

To launch the viewer in browse-only mode, disable all write actions:

```bash
./.venv/bin/python -m viewer.app --read-only --open-browser
```

On macOS you can also double-click `launch_stockport_viewer_read_only.command`.

To share the viewer with other people on the same Wi-Fi / local network, run it on all interfaces instead of `127.0.0.1`:

```bash
./.venv/bin/python -m viewer.app --host 0.0.0.0 --port 8787
```

On macOS you can also double-click `launch_stockport_viewer_lan.command`. It prints the local-network URL, for example `http://192.168.x.x:8787/`.

To combine both and share a safer browse-only version on your local network:

```bash
./.venv/bin/python -m viewer.app --host 0.0.0.0 --port 8787 --read-only
```

On macOS you can also double-click `launch_stockport_viewer_lan_read_only.command`.

To share the read-only viewer beyond your local network, you need a tunnel as well as browse-only mode. The quickest path is Cloudflare Quick Tunnel:

```bash
brew install cloudflared
./.venv/bin/python -m viewer.app --host 127.0.0.1 --port 8788 --read-only
cloudflared tunnel --url http://127.0.0.1:8788
```

Or on macOS, double-click `launch_stockport_viewer_public_read_only.command`. That launcher:

- starts a read-only viewer on `127.0.0.1:8788`
- adds temporary browser Basic Auth credentials
- opens a public Cloudflare tunnel and prints the shareable URL

Anyone with the URL and credentials can open the shared view while that terminal window stays open.

If you want a lighter beta-sharing version, you can also use a public read-only tunnel with no password:

```bash
./.venv/bin/python -m viewer.app --host 127.0.0.1 --port 8789 --read-only
cloudflared tunnel --url http://127.0.0.1:8789
```

Or on macOS, double-click `launch_stockport_viewer_public_beta.command`.

That version is easier to share, but the URL is still temporary and anyone with the link can open it.

For a narrow live ingest smoke test, you can target a single league:

```bash
./.venv/bin/python -m ingestion.run_daily_ingest --from-date 2026-03-10 --to-date 2026-03-12 --league-id 41 --skip-state
```

To backfill player bio coverage from API-Football so the model can use real DOB or current age instead of treating most players as unknown:

```bash
./.venv/bin/python -m ingestion.backfill_player_profiles
```

That defaults to the latest season per tracked league. Add `--all-seasons` if you want to widen it later.

To bootstrap historical player-match data from the existing `football_model` CSV instead of spending API quota:

```bash
./.venv/bin/python -m ingestion.import_legacy_raw_stats --tracked-only
```

Wyscout filtered imports can still be run with an explicit `player_id`, but they now also support automatic resolution from the CSV `player` / `team` fields and persist that mapping for later imports:

```python
from ingestion.wyscout_import import import_wyscout_export

import_wyscout_export(
    "data/wyscout/lewis_fiorini_central.csv",
    player_id=None,
    season="2025",
    zone="attacking_midfield_filter",
    league_id=41,
)
```

League-folder season-average workbooks can be imported in bulk too. The importer understands split `pt1` / `pt2` files, merges duplicate players within a league-season, and writes unmatched rows to `data/wyscout/unmatched` for later cleanup:

```bash
./.venv/bin/python -m ingestion.wyscout_import /Users/benmills/Downloads
```

Once raw data is in, bootstrap the live analytical layer before running briefs. This will create any missing tables, classify `player_roles` for all available seasons, and backfill `market_values` from the latest Wyscout season-average rows:

```bash
./.venv/bin/python -m governance.prepare_live_pipeline
```

The browser viewer now includes a Wyscout cleanup page at `/wyscout-review`, where you can:

- review the latest unmatched season-average rows
- accept suggested player matches or enter a manual `player_id`
- rerun the relevant Wyscout import without using SQL or shell commands
- create recruitment briefs, run longlists, and open browser-rendered reports from the dashboard

By default the viewer looks for source workbooks under `~/Downloads`. Override that with `WYSCOUT_SOURCE_DIR` if your league folders live somewhere else.

You can also force browse-only mode via environment variable:

```bash
STOCKPORT_VIEWER_READ_ONLY=true
```

Optional browser password prompt for shared views:

```bash
STOCKPORT_VIEWER_BASIC_AUTH_USER=stockport
STOCKPORT_VIEWER_BASIC_AUTH_PASSWORD=choose-a-password
```

## Permanent Hosting

The cleanest permanent setup for this project is:

- a hosted web service running the read-only viewer
- a managed PostgreSQL database containing your existing local data

This repo is now prepared for that flow with:

- [render.yaml](/Users/benmills/Stockport_Model/render.yaml) for a Render blueprint
- [Dockerfile](/Users/benmills/Stockport_Model/Dockerfile) for container-based hosting
- [scripts/render-build.sh](/Users/benmills/Stockport_Model/scripts/render-build.sh)
- [scripts/render-start.sh](/Users/benmills/Stockport_Model/scripts/render-start.sh)
- [scripts/export_local_database.sh](/Users/benmills/Stockport_Model/scripts/export_local_database.sh)
- [scripts/restore_database_dump.sh](/Users/benmills/Stockport_Model/scripts/restore_database_dump.sh)

Suggested hosted path:

1. Put this project in a Git repository and push it to GitHub.
2. Create a new Render Blueprint from that repo using `render.yaml`.
3. Let Render provision the web service and managed Postgres.
4. Export your local database:

```bash
./scripts/export_local_database.sh
```

5. Restore that dump into the hosted Postgres using the external database URL from Render:

```bash
./scripts/restore_database_dump.sh artifacts/your_dump_file.dump "YOUR_HOSTED_DATABASE_URL"
```

6. Open the Render service URL once the restore is complete.

Notes:

- The hosted viewer is intended to run with `STOCKPORT_VIEWER_READ_ONLY=true`.
- `STOCKPORT_DATABASE_URL` is automatically normalized for hosted Postgres URLs, so plain `postgresql://...` and `postgres://...` connection strings are converted to `postgresql+psycopg://...`.
- If you want a different host later, the included `Dockerfile` gives you a portable base for Fly, Railway, or another Docker-capable platform.

## Notes

- `.env` values are loaded automatically if present, but shell environment variables still take precedence.
- The codebase is structurally complete against the specification, but live operation still depends on real database credentials, external API keys, source URLs, and training data/model artifacts.
- Several modelling areas include practical fallbacks where the specification assumes richer live data than is currently available in development.
- The projection, availability, and financial model layers now include heuristic fallbacks so the composite stack can run before full trained artifacts and richer enrichment sources are in place.
- The local viewer is primarily for browsing raw/enriched database content, but it now also supports Wyscout mapping review, browser-created briefs, and one-click longlist/report runs.
