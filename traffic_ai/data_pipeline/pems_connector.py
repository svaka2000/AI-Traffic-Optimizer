"""traffic_ai/data_pipeline/pems_connector.py

Caltrans PeMS (Performance Measurement System) data connector.

PeMS provides free per-lane volume, speed, and occupancy data at 5-minute
intervals for California freeway detectors.  See https://pems.dot.ca.gov

Usage
-----
    from traffic_ai.data_pipeline.pems_connector import PeMSConnector

    connector = PeMSConnector(station_id=400456)
    df = connector.fetch(date_from="2024-01-15", date_to="2024-01-22")
    calibration = connector.calibration_by_hour(df)

If the PeMS API key is missing or the API is unavailable, the connector
falls back to synthetic data and logs a clear warning.

Default calibration target
--------------------------
PeMS Station 400456 — I-5 near downtown San Diego, CA (Caltrans District 11).
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Schema for Phase 9A CSV adapter output
_PEMS_CSV_SCHEMA: list[str] = [
    "timestamp",
    "station_id",
    "hour_of_day",
    "day_of_week",
    "is_rush_hour",
    "total_flow_per_5min",
    "flow_per_lane_per_5min",
    "arrival_rate_per_sec",
    "occupancy",
    "avg_speed_mph",
    "pct_observed",
    "data_quality_ok",
]


def _find_col(col_lower: dict[str, str], candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive), or None."""
    for candidate in candidates:
        if candidate in col_lower:
            return col_lower[candidate]
    return None


def _safe_float(value: object, default: float) -> float:
    """Convert value to float; return default on any failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

_DEFAULT_STATION: int = 400456           # I-5 near downtown San Diego
_PEMS_API_BASE: str = "https://pems.dot.ca.gov"
_ENV_VAR_API_KEY: str = "PEMS_API_KEY"       # optional token-based auth
_ENV_VAR_USERNAME: str = "PEMS_USERNAME"     # username/password auth
_ENV_VAR_PASSWORD: str = "PEMS_PASSWORD"

# Unified schema column names required by DatasetPreprocessor
_UNIFIED_SCHEMA: list[str] = [
    "timestamp",
    "station_id",
    "lane",
    "volume",         # vehicles per 5-min interval
    "occupancy",      # 0-1 fraction of time loop is occupied
    "speed_mph",
    "arrival_rate",   # derived: vehicles per second
    "hour_of_day",
    "day_of_week",
    "is_rush_hour",
    "is_weekend",
    "rolling_mean_volume_1h",
    "rolling_mean_speed_1h",
    "queue_proxy",    # (1 - occupancy) proxy for downstream queue
    "optimal_phase",  # 0=NS, 1=EW derived label for ML training
]


# ---------------------------------------------------------------------------
# PeMSConnector
# ---------------------------------------------------------------------------

class PeMSConnector:
    """Fetch and normalise PeMS loop detector data.

    Parameters
    ----------
    station_id:
        PeMS detector station ID (default 400456 = I-5 near downtown San Diego).
    api_key:
        PeMS API key.  If ``None``, reads from the ``PEMS_API_KEY`` environment
        variable.  When absent, the connector falls back to synthetic data.
    cache_dir:
        Directory for caching raw CSV responses.  Defaults to
        ``data/raw/pems/``.
    """

    def __init__(
        self,
        station_id: int = _DEFAULT_STATION,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        cache_dir: str | Path = "data/raw/pems",
    ) -> None:
        self.station_id = station_id
        self._api_key: str | None = api_key or os.environ.get(_ENV_VAR_API_KEY)
        self._username: str | None = username or os.environ.get(_ENV_VAR_USERNAME)
        self._password: str | None = password or os.environ.get(_ENV_VAR_PASSWORD)
        self._session: Any = None   # requests.Session after login
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self._api_key and not (self._username and self._password):
            warnings.warn(
                f"PeMS credentials not found.  Set {_ENV_VAR_USERNAME!r} + "
                f"{_ENV_VAR_PASSWORD!r} (or {_ENV_VAR_API_KEY!r}) as environment "
                "variables or in a .env file.  Falling back to synthetic data.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(
        self,
        date_from: str | date,
        date_to: str | date,
        lanes: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch PeMS detector data for a date range.

        If the API key is absent or the request fails, returns a synthetic
        DataFrame calibrated to typical I-5 San Diego volume patterns.

        Parameters
        ----------
        date_from, date_to:
            Inclusive date range (``"YYYY-MM-DD"`` or :class:`datetime.date`).
        lanes:
            Specific lane numbers to return.  ``None`` = all lanes.

        Returns
        -------
        DataFrame with columns matching ``_UNIFIED_SCHEMA``.
        """
        has_credentials = bool(self._api_key or (self._username and self._password))
        if not has_credentials:
            logger.warning(
                "PeMS credentials missing — returning synthetic fallback data for "
                "station %d (%s → %s).", self.station_id, date_from, date_to
            )
            return self._synthetic_fallback(date_from, date_to)

        try:
            raw = self._fetch_from_api(date_from, date_to)
            df = self._normalise(raw, lanes=lanes)
            logger.info(
                "Fetched %d rows from PeMS station %d.", len(df), self.station_id
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PeMS API request failed (%s) — falling back to synthetic data "
                "for station %d.", exc, self.station_id
            )
            return self._synthetic_fallback(date_from, date_to)

    # ------------------------------------------------------------------
    # Phase 9A: PeMS CSV file adapter
    # ------------------------------------------------------------------

    def load_from_csv(self, path: Path) -> pd.DataFrame:
        """Parse a PeMS Station 5-Minute CSV export into AITO's unified schema.

        PeMS 5-minute station CSV files (from Data Clearinghouse at
        pems.dot.ca.gov → Data → Station 5-Minute → District 11) have this
        column layout::

            Timestamp, Station, District, Freeway, Direction, Lane Type,
            Station Length, Samples, % Observed, Total Flow,
            Avg Occupancy, Avg Speed,
            Lane 1 Flow, Lane 1 Avg Occ, Lane 1 Avg Speed, Lane 1 Observed,
            Lane 2 Flow, ...

        The method tolerates missing columns, bad rows, and messy quoting.
        Rows with malformed timestamps or missing Total Flow are skipped with
        a debug log rather than a crash.

        Parameters
        ----------
        path:
            Absolute or relative path to the PeMS station CSV.

        Returns
        -------
        DataFrame with columns:
            timestamp, station_id, hour_of_day, day_of_week, is_rush_hour,
            total_flow_per_5min, flow_per_lane_per_5min, arrival_rate_per_sec,
            occupancy, avg_speed_mph, pct_observed, data_quality_ok
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PeMS CSV not found: {path}")

        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.warning("Failed to read PeMS CSV %s: %s — returning empty frame.", path, exc)
            return pd.DataFrame(columns=_PEMS_CSV_SCHEMA)

        rows: list[dict] = []

        # --- locate key columns case-insensitively ---
        col_lower = {c.strip().lower(): c for c in raw.columns}

        ts_col      = _find_col(col_lower, ["timestamp", "time"])
        flow_col    = _find_col(col_lower, ["total flow", "totalflow"])
        occ_col     = _find_col(col_lower, ["avg occupancy", "avgoccupancy", "occupancy"])
        speed_col   = _find_col(col_lower, ["avg speed", "avgspeed", "speed"])
        pct_obs_col = _find_col(col_lower, ["% observed", "%observed", "pct observed"])
        station_col = _find_col(col_lower, ["station"])

        # count lane flow columns: "Lane N Flow" or "Lane N Avg Occ"
        lane_flow_cols = [c for k, c in col_lower.items()
                          if "lane" in k and "flow" in k]
        n_lanes = max(len(lane_flow_cols), 1)

        for _, row in raw.iterrows():
            try:
                # --- timestamp ---
                ts_raw = row.get(ts_col, "") if ts_col else ""
                ts = pd.to_datetime(ts_raw, errors="coerce")
                if pd.isna(ts):
                    logger.debug("Skipping row with unparseable timestamp: %r", ts_raw)
                    continue

                # --- station id ---
                sid_raw = row.get(station_col, self.station_id) if station_col else self.station_id
                try:
                    sid = int(float(sid_raw))
                except (ValueError, TypeError):
                    sid = self.station_id

                # --- traffic metrics ---
                total_flow = _safe_float(row.get(flow_col, 0.0) if flow_col else 0.0, 0.0)
                if total_flow < 0:
                    total_flow = 0.0
                avg_occ   = max(0.0, min(1.0, _safe_float(row.get(occ_col, 0.0) if occ_col else 0.0, 0.0)))
                avg_speed = max(0.0, _safe_float(row.get(speed_col, 30.0) if speed_col else 30.0, 30.0))
                pct_obs   = max(0.0, min(1.0, _safe_float(row.get(pct_obs_col, 1.0) if pct_obs_col else 1.0, 1.0)))

                # --- derived fields ---
                hour = ts.hour
                dow  = ts.weekday()
                is_rush = (7 <= hour <= 9) or (16 <= hour <= 18)
                flow_per_lane = total_flow / n_lanes
                arrival_rate  = flow_per_lane / 300.0   # vehicles per second per lane
                data_ok = (pct_obs >= 0.5) and (total_flow >= 0)

                rows.append({
                    "timestamp":             ts,
                    "station_id":            sid,
                    "hour_of_day":           hour,
                    "day_of_week":           dow,
                    "is_rush_hour":          is_rush,
                    "total_flow_per_5min":   int(total_flow),
                    "flow_per_lane_per_5min": flow_per_lane,
                    "arrival_rate_per_sec":  arrival_rate,
                    "occupancy":             avg_occ,
                    "avg_speed_mph":         avg_speed,
                    "pct_observed":          pct_obs,
                    "data_quality_ok":       data_ok,
                })
            except Exception as exc:  # noqa: BLE001 — never crash on a bad row
                logger.debug("Skipping malformed PeMS row: %s", exc)
                continue

        if not rows:
            logger.warning("PeMS CSV %s: no valid rows parsed.", path)
            return pd.DataFrame(columns=_PEMS_CSV_SCHEMA)

        df = pd.DataFrame(rows, columns=_PEMS_CSV_SCHEMA)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Loaded PeMS CSV %s: %d rows, %d quality-ok rows.",
            path.name, len(df), int(df["data_quality_ok"].sum()),
        )
        return df

    def compute_hourly_demand_profile(
        self, df: pd.DataFrame
    ) -> dict[int, float]:
        """Compute mean arrival rate by hour of day from parsed PeMS data.

        Filters to:
        - ``data_quality_ok == True`` rows (pct_observed >= 0.5)
        - Weekdays only (day_of_week 0–4) — planning-relevant demand

        Returns a dict mapping hour (0–23) → mean arrival_rate_per_sec.

        If fewer than 3 days of valid data exist for any hour, that hour
        falls back to the synthetic default rate (0.12 veh/sec/lane).

        Parameters
        ----------
        df:
            Output of :meth:`load_from_csv`.
        """
        _SYNTHETIC_DEFAULT = 0.12

        if df.empty or "arrival_rate_per_sec" not in df.columns:
            return {h: _SYNTHETIC_DEFAULT for h in range(24)}

        # Filter quality + weekdays
        mask = df["data_quality_ok"] & (df["day_of_week"] <= 4)
        filtered = df[mask].copy()

        if filtered.empty:
            return {h: _SYNTHETIC_DEFAULT for h in range(24)}

        # Need date column to count distinct days per hour
        filtered = filtered.copy()
        filtered["_date"] = filtered["timestamp"].dt.date

        profile: dict[int, float] = {}
        for hour in range(24):
            hourly = filtered[filtered["hour_of_day"] == hour]
            n_days = hourly["_date"].nunique()
            if n_days >= 3:
                profile[hour] = float(hourly["arrival_rate_per_sec"].mean())
            else:
                profile[hour] = _SYNTHETIC_DEFAULT

        return profile

    def auto_detect_pems_files(self, raw_dir: Path) -> list[Path]:
        """Scan raw_dir for files matching ``pems_station_*.csv`` pattern.

        Returns a sorted list of matching paths.
        """
        return sorted(Path(raw_dir).glob("pems_station_*.csv"))

    def load_best_available(
        self, raw_dir: Path
    ) -> tuple[pd.DataFrame | None, str]:
        """Try to load real PeMS data; fall back to synthetic if none found.

        Returns
        -------
        (dataframe_or_None, source_description)
            ``source_description`` is one of:
            - ``"real_pems: pems_station_400456.csv (N days, M rows)"``
            - ``"synthetic: no PeMS CSV found in data/raw/"``
        """
        csv_files = self.auto_detect_pems_files(raw_dir)
        if not csv_files:
            msg = f"synthetic: no PeMS CSV found in {raw_dir}/"
            logger.info("PeMS CSV adapter: %s — using synthetic demand profile.", msg)
            return None, msg

        # Use first (sorted) file; log which one
        chosen = csv_files[0]
        try:
            df = self.load_from_csv(chosen)
        except Exception as exc:
            msg = f"synthetic: failed to parse {chosen.name} ({exc})"
            logger.warning("PeMS CSV load failed: %s", exc)
            return None, msg

        if df.empty:
            msg = f"synthetic: {chosen.name} parsed to empty frame"
            return None, msg

        n_rows = len(df)
        n_days = int(df["timestamp"].dt.date.nunique()) if "timestamp" in df.columns else 0
        msg = f"real_pems: {chosen.name} ({n_days} days, {n_rows} rows)"
        logger.info("PeMS CSV adapter: %s", msg)
        return df, msg

    def calibration_by_hour(self, df: pd.DataFrame) -> dict[int, float]:
        """Derive per-hour mean arrival rate from PeMS data.

        Returns a dict mapping ``hour_of_day`` (0-23) →
        mean vehicle count per 5-minute interval that can be passed to
        ``MultiIntersectionNetwork(calibration_data=...)`` or used to
        calibrate the ``DemandModel``.

        Parameters
        ----------
        df:
            Output of :meth:`fetch`.

        Returns
        -------
        dict mapping hour → mean_count_per_5min_interval.
        """
        if df.empty or "hour_of_day" not in df.columns:
            return {}
        grouped = df.groupby("hour_of_day")["volume"].mean()
        return {int(h): float(v) for h, v in grouped.items()}

    # ------------------------------------------------------------------
    # API fetch (requires valid key)
    # ------------------------------------------------------------------

    def _fetch_from_api(
        self,
        date_from: str | date,
        date_to: str | date,
    ) -> pd.DataFrame:
        """Download raw PeMS clearinghouse data for the station.

        PeMS clearinghouse endpoint (type=station_5min):
            GET /clearinghouse?
                    type=station_5min
                    &district_id=11
                    &station_id={id}
                    &start_time={date_from}
                    &end_time={date_to}
                    &format=text/csv
                    &user={api_key}

        Each row contains: Timestamp, Station, District, Freeway, Direction,
        Lane Type, Station Length, Samples, % Observed, and per-lane triples
        (Volume, Occupancy, Speed).
        """
        import requests

        d_from = str(date_from)
        d_to = str(date_to)
        cache_key = self.cache_dir / f"station_{self.station_id}_{d_from}_{d_to}.csv"

        if cache_key.exists():
            logger.info("Loading cached PeMS data from %s.", cache_key)
            return pd.read_csv(cache_key)

        # Choose auth strategy: API key param OR session-based login
        session = self._get_session()
        url = _PEMS_API_BASE + "/clearinghouse"
        params: dict[str, Any] = {
            "type": "station_5min",
            "district_id": 11,  # Caltrans District 11 (San Diego)
            "station_id": self.station_id,
            "start_time": d_from,
            "end_time": d_to,
            "format": "text/csv",
        }
        if self._api_key:
            params["user"] = self._api_key
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()

        from io import StringIO
        raw = pd.read_csv(StringIO(response.text))
        raw.to_csv(cache_key, index=False)
        return raw

    def _get_session(self) -> Any:
        """Return an authenticated requests.Session (cached after first login)."""
        import requests
        if self._session is not None:
            return self._session
        session = requests.Session()
        session.headers.update({"User-Agent": "AITO-PeMS-Connector/2.0"})
        if self._username and self._password:
            self._login(session)
        self._session = session
        return session

    def _login(self, session: Any) -> None:
        """Authenticate with PeMS using username/password (session cookie flow)."""
        import requests
        login_url = _PEMS_API_BASE + "/"
        payload = {
            "username": self._username,
            "password": self._password,
            "redirect": "",
            "login": "Login",
        }
        resp = session.post(login_url, data=payload, timeout=30)
        if resp.status_code == 200 and "logout" in resp.text.lower():
            logger.info("PeMS login successful for user %s.", self._username)
        else:
            logger.warning(
                "PeMS login may have failed (status=%d). "
                "Will attempt data fetch anyway.", resp.status_code
            )

    # ------------------------------------------------------------------
    # Normalisation: map raw PeMS columns → 15-column unified schema
    # ------------------------------------------------------------------

    def _normalise(
        self, raw: pd.DataFrame, lanes: list[int] | None = None
    ) -> pd.DataFrame:
        """Normalise raw PeMS output to the 15-column unified schema.

        PeMS 5-min files have the following structure (columns vary by
        district / version, but core fields are standard):
            Timestamp, Station, District, Freeway, Direction, Lane Type,
            Station Length, Samples, % Observed,
            [Lane N Volume, Lane N Occupancy, Lane N Speed] × N_lanes
        """
        rows: list[dict[str, Any]] = []

        # Detect timestamp column (case-insensitive)
        ts_col = next(
            (c for c in raw.columns if "timestamp" in c.lower() or "time" in c.lower()),
            raw.columns[0],
        )

        # Detect per-lane volume/occupancy/speed columns
        vol_cols = [c for c in raw.columns if "lane" in c.lower() and "vol" in c.lower()]
        occ_cols = [c for c in raw.columns if "lane" in c.lower() and "occ" in c.lower()]
        spd_cols = [c for c in raw.columns if "lane" in c.lower() and ("spd" in c.lower() or "speed" in c.lower())]

        if not vol_cols:
            # Fallback: assume single-lane aggregated columns
            vol_cols = [c for c in raw.columns if "vol" in c.lower()]
            occ_cols = [c for c in raw.columns if "occ" in c.lower()]
            spd_cols = [c for c in raw.columns if "spd" in c.lower() or "speed" in c.lower()]

        n_lanes = max(len(vol_cols), 1)

        for _, row_data in raw.iterrows():
            ts = pd.to_datetime(row_data.get(ts_col, "2024-01-01"), errors="coerce")
            if pd.isna(ts):
                continue
            hour = ts.hour
            dow = ts.weekday()
            is_rush = 1 if (7 <= hour < 9) or (16 <= hour < 19) else 0
            is_weekend = 1 if dow >= 5 else 0

            for lane_idx in range(n_lanes):
                if lanes is not None and lane_idx not in lanes:
                    continue

                vol_val = float(row_data.get(vol_cols[lane_idx] if lane_idx < len(vol_cols) else vol_cols[0], 0.0) or 0.0)
                occ_val = float(row_data.get(occ_cols[lane_idx] if lane_idx < len(occ_cols) else occ_cols[0], 0.0) or 0.0)
                spd_val = float(row_data.get(spd_cols[lane_idx] if lane_idx < len(spd_cols) else spd_cols[0], 30.0) or 30.0)

                # Arrival rate: volume per 5-min interval → vehicles per second
                arrival_rate = vol_val / 300.0

                # Simple label: 0=NS, 1=EW based on hour-of-day heuristic
                # (NS gets priority during morning rush from residential areas)
                optimal_phase = 0 if (7 <= hour < 9) else 1

                rows.append({
                    "timestamp": ts,
                    "station_id": self.station_id,
                    "lane": lane_idx + 1,
                    "volume": vol_val,
                    "occupancy": max(0.0, min(1.0, occ_val)),
                    "speed_mph": max(0.0, spd_val),
                    "arrival_rate": arrival_rate,
                    "hour_of_day": hour,
                    "day_of_week": dow,
                    "is_rush_hour": is_rush,
                    "is_weekend": is_weekend,
                    "rolling_mean_volume_1h": vol_val,   # placeholder; computed below
                    "rolling_mean_speed_1h": spd_val,
                    "queue_proxy": max(0.0, 1.0 - occ_val),
                    "optimal_phase": optimal_phase,
                })

        if not rows:
            return pd.DataFrame(columns=_UNIFIED_SCHEMA)

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Compute rolling means (12 × 5-min = 1 hour)
        df["rolling_mean_volume_1h"] = (
            df.groupby("lane")["volume"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )
        df["rolling_mean_speed_1h"] = (
            df.groupby("lane")["speed_mph"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )

        return df[_UNIFIED_SCHEMA]

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    def _synthetic_fallback(
        self,
        date_from: str | date,
        date_to: str | date,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic data that mimics typical I-5 San Diego patterns.

        Uses a Gaussian-mixture demand model calibrated to approximate
        real PeMS Station 400456 volume patterns:
            - Morning peak: ~1,800 veh/hr at 8 AM
            - Evening peak: ~2,100 veh/hr at 5:30 PM
            - Off-peak:     ~600 veh/hr
        """
        rng = np.random.default_rng(seed)

        d_from = pd.to_datetime(str(date_from))
        d_to = pd.to_datetime(str(date_to))
        timestamps = pd.date_range(d_from, d_to, freq="5min")

        rows: list[dict[str, Any]] = []
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            dow = ts.dayofweek
            is_weekend = 1 if dow >= 5 else 0

            # Calibrated to PeMS 400456 (I-5 San Diego) hourly volumes
            morning = np.exp(-((hour - 8.0) ** 2) / 2.0)
            evening = np.exp(-((hour - 17.5) ** 2) / 2.0)
            base_rate = 600.0  # veh/hr off-peak
            weekend_factor = 0.70 if is_weekend else 1.0
            hourly_vol = base_rate * weekend_factor * (
                1.0 + 2.0 * morning + 2.5 * evening
            )
            vol_5min = max(0.0, float(rng.poisson(hourly_vol / 12.0)))
            occ = min(1.0, max(0.0, vol_5min / 150.0 + rng.normal(0, 0.02)))
            spd = max(5.0, 65.0 - occ * 60.0 + rng.normal(0, 3.0))
            is_rush = 1 if (7 <= ts.hour < 9) or (16 <= ts.hour < 19) else 0
            optimal_phase = 0 if (7 <= ts.hour < 9) else 1

            rows.append({
                "timestamp": ts,
                "station_id": self.station_id,
                "lane": 1,
                "volume": vol_5min,
                "occupancy": occ,
                "speed_mph": spd,
                "arrival_rate": vol_5min / 300.0,
                "hour_of_day": ts.hour,
                "day_of_week": dow,
                "is_rush_hour": is_rush,
                "is_weekend": is_weekend,
                "rolling_mean_volume_1h": vol_5min,
                "rolling_mean_speed_1h": spd,
                "queue_proxy": max(0.0, 1.0 - occ),
                "optimal_phase": optimal_phase,
            })

        if not rows:
            return pd.DataFrame(columns=_UNIFIED_SCHEMA)

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        window = 12  # 1 hour
        df["rolling_mean_volume_1h"] = df["volume"].rolling(window, min_periods=1).mean()
        df["rolling_mean_speed_1h"] = df["speed_mph"].rolling(window, min_periods=1).mean()
        return df[_UNIFIED_SCHEMA]
