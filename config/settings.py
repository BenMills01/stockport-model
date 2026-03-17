"""Runtime settings and config file loaders."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalise_database_url(raw_url: str) -> str:
    """Ensure SQLAlchemy uses psycopg for hosted Postgres URLs."""

    url = raw_url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+psycopg" not in url.split("://", 1)[0]:
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def load_env_file(
    path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Load simple KEY=VALUE pairs from a dotenv-style file.

    Existing environment variables win over file values so shell exports remain the
    highest-priority configuration source.
    """

    target = environ if environ is not None else os.environ

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in target:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        target[key] = value


load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Central application settings loaded from environment variables."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    database_url: str = field(
        default_factory=lambda: normalise_database_url(
            os.getenv(
                "STOCKPORT_DATABASE_URL",
                "postgresql+psycopg://stockport:stockport@localhost:5432/stockport_model",
            )
        )
    )
    api_football_base_url: str = "https://v3.football.api-sports.io"
    api_football_api_key: str | None = field(
        default_factory=lambda: os.getenv("API_FOOTBALL_API_KEY")
    )
    transfermarkt_base_url: str = "https://www.transfermarkt.com"
    fbref_base_url: str = "https://fbref.com"
    http_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    wyscout_import_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WYSCOUT_IMPORT_DIR",
                str(Path(__file__).resolve().parents[1] / "data" / "wyscout"),
            )
        )
    )
    wyscout_source_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WYSCOUT_SOURCE_DIR",
                str(Path.home() / "Downloads"),
            )
        )
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data"
    )
    templates_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "templates"
    )
    daily_ingest_state_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
        / "data"
        / "daily_ingest_state.json"
    )
    api_rate_limit_seconds: int = 6
    reference_league_name: str = "Championship"
    confidence_appearance_threshold: int = 10
    league_completeness_warning_threshold: float = 0.80
    gbp_to_eur_rate: float = field(
        default_factory=lambda: float(os.getenv("GBP_TO_EUR_RATE", "1.17"))
    )
    usd_to_eur_rate: float = field(
        default_factory=lambda: float(os.getenv("USD_TO_EUR_RATE", "0.92"))
    )
    chf_to_eur_rate: float = field(
        default_factory=lambda: float(os.getenv("CHF_TO_EUR_RATE", "1.04"))
    )
    sql_echo: bool = field(
        default_factory=lambda: os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"
    )
    viewer_read_only: bool = field(
        default_factory=lambda: os.getenv("STOCKPORT_VIEWER_READ_ONLY", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    viewer_basic_auth_user: str | None = field(
        default_factory=lambda: os.getenv("STOCKPORT_VIEWER_BASIC_AUTH_USER") or None
    )
    viewer_basic_auth_password: str | None = field(
        default_factory=lambda: os.getenv("STOCKPORT_VIEWER_BASIC_AUTH_PASSWORD") or None
    )
    skillcorner_username: str | None = field(
        default_factory=lambda: os.getenv("SKILLCORNER_USERNAME") or None
    )
    skillcorner_password: str | None = field(
        default_factory=lambda: os.getenv("SKILLCORNER_PASSWORD") or None
    )
    skillcorner_base_url: str = "https://www.skillcorner.com/api"

    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    def load_json(self, filename: str) -> Any:
        with (self.config_dir / filename).open("r", encoding="utf-8") as handle:
            return json.load(handle)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
