"""ngl.py -- fetch and parse Nevada Geodetic Laboratory (NGL) daily `tenv3`
station time series.

Verified endpoint (2026-10-07):
    https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14/<SITE>.tenv3
    https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/<SITE>.tenv3
Station holdings catalogue:
    https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt

tenv3 columns (header line present) include, in order:
    site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m)
    _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m)
    __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)

`east/north/up` are the local ENU coordinates in **metres**; we convert to mm.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

BASE = "https://geodesy.unr.edu/gps_timeseries/{frame}/tenv3/{frame}/{site}.tenv3"
HOLDINGS = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"


def _curl(url: str, dest: str, timeout: int) -> bool:
    """Download via curl (proven reliable on this host). Returns True on success."""
    curl = shutil.which("curl") or "curl"
    cmd = [curl, "-fsS", "--retry", "3", "--retry-delay", "2", "--max-time", str(timeout),
           "-A", "gnss-benchmark/1.0", "-o", dest, url]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.getsize(dest) > 0
    except Exception:  # noqa: BLE001
        return False


def fetch(url: str, dest: str, retries: int = 4, timeout: int = 300) -> bool:
    """Download `url` to `dest` (atomic via .part). Returns True on success.

    Timeout is generous (300 s) because the NGL host serves each connection at
    only ~18 KB/s; a 2.3 MB tenv3 file needs ~130 s, so a 120 s limit made most
    downloads time out and restart from scratch.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    part = dest + ".part"
    for k in range(retries):
        if _curl(url, part, timeout) and os.path.getsize(part) > 0:
            os.replace(part, dest)
            return True
        time.sleep(1.5 * (k + 1))
    if os.path.exists(part):
        os.remove(part)
    print(f"  FAIL {os.path.basename(dest)}")
    return False


def download_station(site: str, dest_dir: str, frame: str = "IGS14") -> str | None:
    url = BASE.format(frame=frame, site=site)
    dest = os.path.join(dest_dir, f"{site}.tenv3")
    return dest if fetch(url, dest) else None


def download_holdings(dest: str) -> str:
    fetch(HOLDINGS, dest)
    return dest


def load_holdings(path: str):
    """Return list of dicts from DataHoldings.txt (header line 1).

    Columns: Sta Lat(deg) Long(deg) Hgt(m) X(m) Y(m) Z(m) Dtbeg Dtend Dtmod
    NumSol StaOrigName -- i.e. NumSol is column 10, and Dtbeg/Dtend are
    YYYY-MM-DD dates (decimal year added as tbeg_y / tend_y).
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.readline()  # header
        for line in fh:
            p = line.split()
            if len(p) < 11:
                continue
            try:
                nb = int(float(p[10]))
            except ValueError:
                continue
            rows.append({
                "site": p[0], "lat": float(p[1]), "lon": float(p[2]), "h": float(p[3]),
                "tbeg": p[7], "tend": p[8], "n": nb,
                "tbeg_y": _date_to_year(p[7]), "tend_y": _date_to_year(p[8]),
            })
    return rows


def _date_to_year(s: str) -> float:
    """'YYYY-MM-DD' -> decimal year (fraction by day-of-year)."""
    import datetime as _dt
    y, m, d = (int(x) for x in s.split("-"))
    t0 = _dt.date(y, 1, 1)
    t1 = _dt.date(y + 1, 1, 1)
    return y + (_dt.date(y, m, d) - t0).days / (t1 - t0).days


def _colidxs(header: list[str]) -> dict:
    def find(*cands):
        for c in cands:
            for i, h in enumerate(header):
                if c in h:
                    return i
        return -1
    return {
        "t": find("yyyy.yyyy"),
        "e": find("east(m)"),
        "n": find("north(m)"),
        "u": find("up(m)"),
        "se": find("sig_e"),
        "sn": find("sig_n"),
        "su": find("sig_u"),
        "lat": find("latitude"),
        "lon": find("longitude"),
        "h": find("height(m)"),
    }


def parse_tenv3(path: str):
    """Parse one tenv3 file -> dict(t, enu_mm (n,3), sig_mm (n,3), lat, lon, h, site)."""
    import numpy as np
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().split()
        idx = _colidxs(header)
        site = os.path.basename(path).split(".")[0]
        t, e, n, u, se, sn, su, lat, lon, h = [], [], [], [], [], [], [], [], [], []
        for line in fh:
            p = line.split()
            if len(p) < len(header):
                continue
            try:
                t.append(float(p[idx["t"]]))
                e.append(float(p[idx["e"]])); n.append(float(p[idx["n"]])); u.append(float(p[idx["u"]]))
                se.append(float(p[idx["se"]])); sn.append(float(p[idx["sn"]])); su.append(float(p[idx["su"]]))
            except (ValueError, IndexError):
                continue
            lat.append(float(p[idx["lat"]])); lon.append(float(p[idx["lon"]])); h.append(float(p[idx["h"]]))
    arr = lambda a: np.asarray(a, float)  # noqa: E731
    t = arr(t)
    o = np.argsort(t)
    return {
        "site": site, "t": t[o],
        "enu_mm": np.column_stack([arr(e), arr(n), arr(u)])[o] * 1000.0,
        "sig_mm": np.column_stack([arr(se), arr(sn), arr(su)])[o] * 1000.0,
        "lat": float(lat[0]) if lat else float("nan"),
        "lon": float(lon[0]) if lon else float("nan"),
        "h": float(h[0]) if h else float("nan"),
    }
