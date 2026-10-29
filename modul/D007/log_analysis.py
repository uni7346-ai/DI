import gc, os, time, threading
import numpy as np
import pandas as pd
import polars as pl
import psutil

_proc = psutil.Process(os.getpid())
def rss_mb():
    return _proc.memory_info().rss / 1024**2

class PeakMemory:
    """with 블록 실행 중 RSS 최고치를 기록 (단위 MB)"""
    def __init__(self, interval=0.005):
        self.interval = interval
    def __enter__(self):
        gc.collect()
        self.base = rss_mb(); self.peak = self.base
        self._stop = threading.Event()
        def watch():
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_mb())
                time.sleep(self.interval)
        self._t = threading.Thread(target=watch, daemon=True); self._t.start()
        return self
    def __exit__(self, *exc):
        self._stop.set(); self._t.join()
        self.peak = max(self.peak, rss_mb())
        self.delta = self.peak - self.base   # 피크 증가량

def analyze_pandas_optimized(csv_path):
    dtype_map = {
        "user_id": "int32",
        "page": "category", "device": "category",
        "status_code": "int16", "response_ms": "int16", "bytes_sent": "int32",
    }
    df = pd.read_csv(csv_path, dtype=dtype_map, parse_dates=["ts"])

    # (1) 페이지별 응답 시간 분포
    by_page = (df.groupby("page", observed=True)["response_ms"]
                 .agg(["mean", "median", "std"]).round(2).sort_index())

    # (2) 디바이스별 트래픽 점유율 (bytes_sent 기준)
    dev_total = df.groupby("device", observed=True)["bytes_sent"].sum().astype("int64")
    dev_share = (dev_total / dev_total.sum() * 100).round(1).rename("share_pct")

    # (3) 에러(400+) 페이지별 건수
    err_count = (df[df["status_code"] >= 400]
                 .groupby("page", observed=True).size()
                 .rename("n_errors").sort_values(ascending=False))

    return {"by_page": by_page, "dev_share": dev_share, "err_count": err_count}

def _stats_from_counts(vals, cnts):
    """정렬된 값·빈도 → 평균, 중앙값, 표본표준편차(ddof=1)"""
    n = cnts.sum()
    mean = (vals * cnts).sum() / n
    var = ((vals - mean) ** 2 * cnts).sum() / (n - 1)
    cum = np.cumsum(cnts)
    lo = vals[np.searchsorted(cum, (n - 1) // 2 + 1)]   # 0-index 기준 (n-1)//2 번째
    hi = vals[np.searchsorted(cum, n // 2 + 1)]         # 0-index 기준 n//2 번째
    return mean, (lo + hi) / 2, np.sqrt(var)

def analyze_chunked(csv_path, chunksize=50_000):
    ms_hist = None                     # (page, response_ms) → 건수
    dev_bytes = pd.Series(dtype="int64")
    err_by_page = pd.Series(dtype="int64")

    reader = pd.read_csv(csv_path, chunksize=chunksize,
                         dtype={"user_id": "int32", "page": "category", "device": "category",
                                "status_code": "int16", "response_ms": "int16",
                                "bytes_sent": "int32"},
                         parse_dates=["ts"])
    for chunk in reader:
        h = chunk.groupby(["page", "response_ms"], observed=True).size()
        ms_hist = h if ms_hist is None else ms_hist.add(h, fill_value=0)

        d = chunk.groupby("device", observed=True)["bytes_sent"].sum().astype("int64")
        dev_bytes = dev_bytes.add(d, fill_value=0)

        e = chunk[chunk["status_code"] >= 400].groupby("page", observed=True).size()
        err_by_page = err_by_page.add(e, fill_value=0)

    # reduce
    rows = {}
    for page, s in ms_hist.groupby(level="page", observed=True):
        s = s.droplevel("page").sort_index()
        rows[page] = _stats_from_counts(s.index.to_numpy(np.float64), s.to_numpy(np.float64))
    by_page = (pd.DataFrame(rows, index=["mean", "median", "std"]).T
                 .rename_axis("page").round(2).sort_index())

    dev_share = (dev_bytes / dev_bytes.sum() * 100).round(1).rename("share_pct")
    err_count = err_by_page.astype("int64").sort_values(ascending=False).rename("n_errors")
    return {"by_page": by_page, "dev_share": dev_share, "err_count": err_count}

def analyze_polars_lazy(csv_path):
    lf = pl.scan_csv(csv_path, try_parse_dates=True)

    by_page_lf = lf.group_by("page").agg(
        pl.col("response_ms").mean().alias("mean"),
        pl.col("response_ms").median().alias("median"),
        pl.col("response_ms").std().alias("std"),          # 기본 ddof=1 (pandas와 동일)
    ).sort("page")

    dev_lf = lf.group_by("device").agg(pl.col("bytes_sent").sum().alias("total_bytes"))

    err_lf = (lf.filter(pl.col("status_code") >= 400)       # predicate pushdown
                .group_by("page").agg(pl.len().alias("n_errors"))
                .sort("n_errors", descending=True))

    by_page_df, dev_df, err_df = pl.collect_all([by_page_lf, dev_lf, err_lf])

    by_page = by_page_df.to_pandas().set_index("page").round(2)
    dev_tot = dev_df.to_pandas().set_index("device")["total_bytes"]
    dev_share = (dev_tot / dev_tot.sum() * 100).round(1).rename("share_pct")
    err_count = err_df.to_pandas().set_index("page")["n_errors"]
    return {"by_page": by_page, "dev_share": dev_share, "err_count": err_count}
