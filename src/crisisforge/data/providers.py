from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class DownloadedFrame:
    """Normalized data plus enough provenance to audit the raw download."""

    frame: pd.DataFrame
    source_urls: tuple[str, ...]
    provider: str
    retrieved_at_utc: str
    payload_sha256: str


def _session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "CrisisForge/0.2 research-data-pipeline"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    lower = pd.Timestamp(start_date)
    upper = pd.Timestamp(end_date)
    result = frame.loc[frame["date"].between(lower, upper)].copy()
    result = result.sort_values("date")
    duplicate_dates = result.loc[result["date"].duplicated(keep=False), "date"]
    if not duplicate_dates.empty:
        examples = sorted(pd.DatetimeIndex(duplicate_dates.unique()).strftime("%Y-%m-%d").tolist())[
            :5
        ]
        raise RuntimeError(f"Source contains duplicate dates after filtering; examples={examples}")
    return result.reset_index(drop=True)


def _payload_digest(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def download_direct_csv(
    *,
    source_id: str,
    url: str,
    date_column: str,
    start_date: str,
    end_date: str,
    requested_columns: list[str],
    timeout_seconds: int = 45,
) -> DownloadedFrame:
    """Download an official CSV and normalize its date column."""
    response = _session().get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload_bytes = response.content
    frame = pd.read_csv(io.BytesIO(payload_bytes))
    required = {date_column, *requested_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{source_id} CSV is missing required columns: {missing}")
    frame = frame[[date_column, *requested_columns]].rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in requested_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[~frame[requested_columns].isna().all(axis=1)].copy()
    frame = _filter_dates(frame, start_date, end_date)
    return DownloadedFrame(
        frame=frame,
        source_urls=(response.url,),
        provider="direct_csv",
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def download_treasury_yield_curve(
    *,
    year_url_template: str,
    start_date: str,
    end_date: str,
    requested_columns: list[str],
    timeout_seconds: int = 45,
) -> DownloadedFrame:
    """Download yearly Treasury Atom/XML yield-curve files from Treasury.gov."""
    session = _session()
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    payloads: list[bytes] = []
    source_urls: list[str] = []
    rows: list[dict[str, Any]] = []
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    }

    for year in range(start_year, end_year + 1):
        url = year_url_template.format(year=year)
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.content
        payloads.append(payload)
        source_urls.append(response.url)
        root = ET.fromstring(payload)
        for entry in root.findall("atom:entry", namespaces):
            properties = entry.find("atom:content/m:properties", namespaces)
            if properties is None:
                continue
            date_node = properties.find("d:NEW_DATE", namespaces)
            if date_node is None or date_node.text is None:
                continue
            row: dict[str, Any] = {"date": date_node.text}
            for column in requested_columns:
                node = properties.find(f"d:{column}", namespaces)
                row[column] = None if node is None else node.text
            rows.append(row)
        polite_pause()

    if not rows:
        raise RuntimeError("Treasury XML returned no yield-curve observations")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in requested_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[~frame[requested_columns].isna().all(axis=1)].copy()
    frame = _filter_dates(frame, start_date, end_date)
    return DownloadedFrame(
        frame=frame,
        source_urls=tuple(source_urls),
        provider="treasury_xml",
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        payload_sha256=_payload_digest(payloads),
    )


def download_nyfed_effr(
    *,
    url: str,
    start_date: str,
    end_date: str,
    timeout_seconds: int = 45,
) -> DownloadedFrame:
    """Download EFFR observations from the New York Fed Markets API."""
    response = _session().get(
        url,
        params={"startDate": start_date, "endDate": end_date, "type": "rate"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload_bytes = response.content
    observations = response.json().get("refRates") or []
    if not observations:
        raise RuntimeError("New York Fed EFFR API returned no observations")
    frame = pd.DataFrame(observations)
    required = {"effectiveDate", "percentRate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"New York Fed EFFR response is missing columns: {missing}")
    frame = frame.rename(columns={"effectiveDate": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["percentRate"] = pd.to_numeric(frame["percentRate"], errors="coerce")
    frame = _filter_dates(frame[["date", "percentRate"]], start_date, end_date)
    return DownloadedFrame(
        frame=frame,
        source_urls=(response.url,),
        provider="nyfed_effr_json",
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def download_macro_source(
    source: dict[str, Any],
    start_date: str,
    end_date: str,
) -> DownloadedFrame:
    """Route a configured macro source to its direct-origin provider."""
    requested_columns = [item["source_column"] for item in source["columns"]]
    provider = source["provider"]
    if provider == "direct_csv":
        return download_direct_csv(
            source_id=source["id"],
            url=source["download_url"],
            date_column=source["date_column"],
            start_date=start_date,
            end_date=end_date,
            requested_columns=requested_columns,
        )
    if provider == "treasury_xml":
        return download_treasury_yield_curve(
            year_url_template=source["year_url_template"],
            start_date=start_date,
            end_date=end_date,
            requested_columns=requested_columns,
        )
    if provider == "nyfed_effr_json":
        return download_nyfed_effr(
            url=source["download_url"],
            start_date=start_date,
            end_date=end_date,
        )
    raise ValueError(f"Unsupported macro provider: {provider}")


def download_french_industry_returns(
    *,
    url: str,
    start_date: str,
    end_date: str,
    requested_columns: list[str],
    timeout_seconds: int = 45,
) -> DownloadedFrame:
    """Download and parse the value-weighted daily section of French industry returns."""
    response = _session().get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload_bytes = response.content
    with zipfile.ZipFile(io.BytesIO(payload_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one French CSV in ZIP, found {members}")
        text = archive.read(members[0]).decode("utf-8-sig", errors="replace")

    lines = text.splitlines()
    try:
        marker_index = next(
            index
            for index, line in enumerate(lines)
            if "Average Value Weighted Returns -- Daily" in line
        )
    except StopIteration as exc:
        raise RuntimeError("French file is missing the value-weighted daily section") from exc
    header_index = next(
        index for index in range(marker_index + 1, len(lines)) if lines[index].strip()
    )
    reader = csv.reader(lines[header_index:])
    header = [item.strip() for item in next(reader)]
    if not header or header[0]:
        raise RuntimeError(f"Unexpected French daily header: {header}")
    header[0] = "date"
    missing = sorted(set(requested_columns).difference(header))
    if missing:
        raise RuntimeError(f"French daily section is missing columns: {missing}")

    rows: list[list[str]] = []
    for row in reader:
        if not row or not re.fullmatch(r"\d{8}", row[0].strip()):
            break
        rows.append([item.strip() for item in row])
    frame = pd.DataFrame(rows, columns=header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    for column in requested_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.mask(values.isin([-99.99, -999.0]))
        frame[column] = values / 100.0
    frame = _filter_dates(frame[["date", *requested_columns]], start_date, end_date)
    return DownloadedFrame(
        frame=frame,
        source_urls=(response.url,),
        provider="french_research_zip",
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def download_return_source(
    source: dict[str, Any],
    start_date: str,
    end_date: str,
) -> DownloadedFrame:
    """Route a configured target-return source."""
    provider = source["provider"]
    if provider == "french_research_zip":
        return download_french_industry_returns(
            url=source["download_url"],
            start_date=start_date,
            end_date=end_date,
            requested_columns=[item["source_column"] for item in source["columns"]],
        )
    raise ValueError(f"Unsupported downloaded return provider: {provider}")


def write_snapshot(download: DownloadedFrame, csv_path: Path, extra: dict[str, Any]) -> None:
    """Write a raw CSV plus immutable provenance metadata."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    metadata_path = csv_path.with_suffix(".meta.json")
    metadata_tmp = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    download.frame.to_csv(csv_tmp, index=False, date_format="%Y-%m-%d")
    metadata = {
        **extra,
        "provider": download.provider,
        "source_urls": list(download.source_urls),
        "retrieved_at_utc": download.retrieved_at_utc,
        "remote_payload_sha256": download.payload_sha256,
        "local_csv_sha256": sha256_file(csv_tmp),
        "row_count": int(len(download.frame)),
        "columns": list(download.frame.columns),
        "first_date": _safe_date(download.frame["date"].min()),
        "last_date": _safe_date(download.frame["date"].max()),
    }
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(csv_tmp, csv_path)
    os.replace(metadata_tmp, metadata_path)


def read_cached_snapshot(
    csv_path: Path,
    *,
    expected_source_id: str | None = None,
    expected_provider: str | None = None,
    expected_columns: list[str] | None = None,
    expected_page_url: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load a cached snapshot only after verifying its provenance metadata."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Required cached snapshot is missing: {csv_path}")
    metadata_path = csv_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Snapshot metadata are missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for required_key in (
        "source_id",
        "provider",
        "source_urls",
        "local_csv_sha256",
        "row_count",
        "columns",
        "first_date",
        "last_date",
    ):
        if required_key not in metadata:
            raise RuntimeError(
                f"Snapshot metadata are incomplete for {csv_path}: missing {required_key}"
            )
    expected_identity = {
        "source_id": expected_source_id,
        "provider": expected_provider,
        "page_url": expected_page_url,
    }
    for key, expected_value in expected_identity.items():
        if expected_value is not None and metadata.get(key) != expected_value:
            raise RuntimeError(
                f"Cached snapshot identity mismatch for {csv_path}: "
                f"{key} expected {expected_value!r}, found {metadata.get(key)!r}"
            )
    expected_hash = metadata.get("local_csv_sha256")
    actual_hash = sha256_file(csv_path)
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError(
            f"Cached snapshot hash mismatch for {csv_path}; refresh from source before using it"
        )
    frame = pd.read_csv(csv_path)
    metadata_columns = list(metadata["columns"])
    if list(frame.columns) != metadata_columns:
        raise RuntimeError(
            f"Cached snapshot schema mismatch for {csv_path}: "
            f"expected {metadata_columns}, found {list(frame.columns)}"
        )
    if expected_columns is not None and list(frame.columns) != expected_columns:
        raise RuntimeError(
            f"Cached snapshot catalog-schema mismatch for {csv_path}: "
            f"expected {expected_columns}, found {list(frame.columns)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    expected_rows = metadata.get("row_count")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise RuntimeError(
            f"Cached snapshot row-count mismatch for {csv_path}: "
            f"expected {expected_rows}, found {len(frame)}"
        )
    if frame["date"].duplicated().any():
        raise RuntimeError(f"Cached snapshot contains duplicate dates: {csv_path}")
    if not frame["date"].is_monotonic_increasing:
        raise RuntimeError(f"Cached snapshot dates are not sorted: {csv_path}")
    actual_first = _safe_date(frame["date"].min())
    actual_last = _safe_date(frame["date"].max())
    if actual_first != metadata["first_date"] or actual_last != metadata["last_date"]:
        raise RuntimeError(
            f"Cached snapshot date-range mismatch for {csv_path}: "
            f"expected {metadata['first_date']}..{metadata['last_date']}, "
            f"found {actual_first}..{actual_last}"
        )
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be supplied together")
    if start_date is not None and end_date is not None:
        if "requested_start" not in metadata or "requested_end" not in metadata:
            raise RuntimeError(f"Snapshot metadata lack requested range for {csv_path}")
        if pd.Timestamp(metadata["requested_start"]) > pd.Timestamp(start_date):
            raise RuntimeError(
                f"Cached snapshot does not cover requested start {start_date}: {csv_path}"
            )
        if pd.Timestamp(metadata["requested_end"]) < pd.Timestamp(end_date):
            raise RuntimeError(
                f"Cached snapshot does not cover requested end {end_date}: {csv_path}"
            )
        frame = _filter_dates(frame, start_date, end_date)
        if frame.empty:
            raise RuntimeError(
                f"Cached snapshot has no observations in {start_date}..{end_date}: {csv_path}"
            )
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def polite_pause(seconds: float = 0.05) -> None:
    """Small delay between calls to public endpoints."""
    time.sleep(seconds)
