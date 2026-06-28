"""
GeckoTerminal Solana Rug Pull Data Collection Pipeline
======================================================
Collects pool data, OHLCV candles, trade history, and token metadata
from the GeckoTerminal free API for Solana meme coins.

No API key required — GeckoTerminal public API is free.
Rate limit: ~30 requests/minute → we enforce 2s delay between calls.
"""

import requests
import time
import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.geckoterminal.com/api/v2"
NETWORK   = "solana"
HEADERS   = {"Accept": "application/json;version=20230302"}
DELAY     = 3.5          # seconds between requests (safe under 30 rpm)
MAX_RETRY = 3            # retries on transient failures


# ── Low-level HTTP ───────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    """GET with retry + rate-limit-aware backoff."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"Rate limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                log.warning(f"Not found: {url}")
                return None
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp.json()
        except requests.RequestException as e:
            log.error(f"Attempt {attempt}/{MAX_RETRY} failed: {e}")
            time.sleep(5 * attempt)
    return None


# ── GeckoTerminal API wrappers ───────────────────────────────────────────────

def get_pool_info(pool_address: str) -> Optional[dict]:
    """
    Pool overview: liquidity, volume, price, tx counts, buy/sell counts.
    Endpoint: GET /networks/solana/pools/{pool_address}
    """
    data = _get(f"/networks/{NETWORK}/pools/{pool_address}")
    if not data:
        return None
    return data.get("data", {}).get("attributes", {})


def get_token_info(token_address: str) -> Optional[dict]:
    """
    Token metadata: name, symbol, social links, decimals, age.
    Endpoint: GET /networks/solana/tokens/{token_address}
    """
    data = _get(f"/networks/{NETWORK}/tokens/{token_address}")
    if not data:
        return None
    return data.get("data", {}).get("attributes", {})


def get_ohlcv(pool_address: str, timeframe: str = "minute", aggregate: int = 5,
              limit: int = 100) -> Optional[list]:
    """
    OHLCV candle data for temporal feature engineering.

    timeframe options : 'minute', 'hour', 'day'
    aggregate         : candle size (e.g. 5 = 5-minute candles)
    limit             : number of candles to fetch (max 1000)

    Returns list of [timestamp, open, high, low, close, volume]
    """
    data = _get(
        f"/networks/{NETWORK}/pools/{pool_address}/ohlcv/{timeframe}",
        params={"aggregate": aggregate, "limit": limit, "currency": "usd"},
    )
    if not data:
        return None
    return data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])


def get_trades(pool_address: str, trade_type: str = "all") -> Optional[list]:
    """
    Recent individual trades — used for tx size, wallet count, buy/sell ratio.
    trade_type: 'buy', 'sell', 'all'
    Returns up to 300 recent trades.
    """
    data = _get(
        f"/networks/{NETWORK}/pools/{pool_address}/trades",
        params={"trade_volume_in_usd_greater_than": 0},
    )
    if not data:
        return None
    trades = data.get("data", [])
    if trade_type == "all":
        return trades
    return [t for t in trades if t.get("attributes", {}).get("kind") == trade_type]


def get_trending_pools(page: int = 1) -> list[dict]:
    data = _get(f"/networks/{NETWORK}/trending_pools", params={"page": page})
    if not data:
        return []
    results = []
    for item in data.get("data", []):
        attr = item.get("attributes", {})
        # extract real token address from relationships
        rels = item.get("relationships", {})
        base_token = rels.get("base_token", {}).get("data", {}).get("id", "")
        # id format is "solana_<token_address>" — strip the prefix
        token_addr = base_token.replace(f"{NETWORK}_", "") if base_token else ""
        attr["_token_address"] = token_addr
        attr["id"] = item.get("id", "")
        results.append(attr)
    return results


def get_new_pools(page: int = 1) -> list[dict]:
    data = _get(f"/networks/{NETWORK}/new_pools", params={"page": page})
    if not data:
        return []
    results = []
    for item in data.get("data", []):
        attr = item.get("attributes", {})
        # extract real token address from relationships
        rels = item.get("relationships", {})
        base_token = rels.get("base_token", {}).get("data", {}).get("id", "")
        # id format is "solana_<token_address>" — strip the prefix
        token_addr = base_token.replace(f"{NETWORK}_", "") if base_token else ""
        attr["_token_address"] = token_addr
        attr["id"] = item.get("id", "")
        results.append(attr)
    return results


# ── Feature engineering from raw API data ───────────────────────────────────

def engineer_features(pool: dict, token: dict, ohlcv: list, trades: list) -> dict:
    """
    Turn raw API responses into ML-ready features.
    All features computed at a single snapshot in time (no lookahead).
    """
    now_ts = datetime.now(timezone.utc).timestamp()

    # ── Liquidity features ──────────────────────────────────────────────────
    liq_usd       = float(pool.get("reserve_in_usd", 0) or 0)
    fdv_usd       = float(pool.get("fdv_usd", 0) or 1)          # avoid div/0
    market_cap    = float(pool.get("market_cap_usd", 0) or fdv_usd)

    liq_change_5m  = float((pool.get("price_change_percentage", {}) or {}).get("m5",  0) or 0)
    liq_change_1h  = float((pool.get("price_change_percentage", {}) or {}).get("h1",  0) or 0)
    liq_change_24h = float((pool.get("price_change_percentage", {}) or {}).get("h24", 0) or 0)

    # ── Volume features ─────────────────────────────────────────────────────
    vol_5m  = float((pool.get("volume_usd", {}) or {}).get("m5",  0) or 0)
    vol_1h  = float((pool.get("volume_usd", {}) or {}).get("h1",  0) or 0)
    vol_6h  = float((pool.get("volume_usd", {}) or {}).get("h6",  0) or 0)
    vol_24h = float((pool.get("volume_usd", {}) or {}).get("h24", 0) or 0)

    vol_spike_ratio = (vol_1h / (vol_24h / 24 + 1e-6))   # 1h vs hourly avg

    # ── Price features ──────────────────────────────────────────────────────
    price_usd = float(pool.get("base_token_price_usd", 0) or 0)
    price_change = pool.get("price_change_percentage", {}) or {}
    p5m  = float(price_change.get("m5",  0) or 0)
    p1h  = float(price_change.get("h1",  0) or 0)
    p6h  = float(price_change.get("h6",  0) or 0)
    p24h = float(price_change.get("h24", 0) or 0)

    # ── Transaction count features ──────────────────────────────────────────
    tx_counts = pool.get("transactions", {}) or {}

    def _tx(window, kind):
        return int((tx_counts.get(window, {}) or {}).get(kind, 0) or 0)

    buys_5m   = _tx("m5",  "buys");   sells_5m  = _tx("m5",  "sells")
    buys_1h   = _tx("h1",  "buys");   sells_1h  = _tx("h1",  "sells")
    buys_24h  = _tx("h24", "buys");   sells_24h = _tx("h24", "sells")
    buyers_5m = _tx("m5",  "buyers"); sellers_5m = _tx("m5", "sellers")
    buyers_1h = _tx("h1",  "buyers"); sellers_1h = _tx("h1", "sellers")

    buy_sell_ratio_1h  = buys_1h  / (sells_1h  + 1e-6)
    buy_sell_ratio_24h = buys_24h / (sells_24h + 1e-6)
    unique_ratio_1h    = buyers_1h / (sellers_1h + 1e-6)

    # ── Trade-level features (from /trades endpoint) ────────────────────────
    trade_sizes_usd = []
    buy_vols, sell_vols = [], []

    for t in (trades or []):
        attr = t.get("attributes", {})
        vol  = float(attr.get("volume_in_usd", 0) or 0)
        kind = attr.get("kind", "")
        trade_sizes_usd.append(vol)
        if kind == "buy":
            buy_vols.append(vol)
        elif kind == "sell":
            sell_vols.append(vol)

    avg_trade_size = sum(trade_sizes_usd) / (len(trade_sizes_usd) + 1e-6)
    max_trade_size = max(trade_sizes_usd, default=0)
    max_trade_pct_pool = max_trade_size / (liq_usd + 1e-6)
    buy_vol_total  = sum(buy_vols)
    sell_vol_total = sum(sell_vols)
    buy_sell_vol_ratio = buy_vol_total / (sell_vol_total + 1e-6)
    whale_flag = int(max_trade_size > 10_000)

    # ── OHLCV / temporal features ───────────────────────────────────────────
    consec_red = 0
    candle_body_wick_ratios = []

    if ohlcv:
        for candle in ohlcv[-20:]:                  # last 20 candles
            ts, o, h, l, c, v = (float(x) for x in candle)
            body  = abs(c - o)
            wick  = (h - l) + 1e-9
            candle_body_wick_ratios.append(body / wick)

        # consecutive red candles (close < open) from most recent
        for candle in reversed(ohlcv[-10:]):
            _, o, h, l, c, v = (float(x) for x in candle)
            if c < o:
                consec_red += 1
            else:
                break

        # volume momentum: last 5 candles vs previous 5
        vols = [float(c[5]) for c in ohlcv]
        recent_avg = sum(vols[-5:]) / 5 if len(vols) >= 5 else 0
        prior_avg  = sum(vols[-10:-5]) / 5 if len(vols) >= 10 else recent_avg
        vol_momentum = recent_avg / (prior_avg + 1e-6)
    else:
        vol_momentum = 1.0

    avg_body_wick = (sum(candle_body_wick_ratios) / len(candle_body_wick_ratios)
                     if candle_body_wick_ratios else 0.5)

    # ── Token metadata / social features ───────────────────────────────────
    created_at_str = pool.get("pool_created_at") or token.get("pool_created_at", "")
    try:
        created_ts = datetime.fromisoformat(
            created_at_str.replace("Z", "+00:00")
        ).timestamp()
        pool_age_hours = (now_ts - created_ts) / 3600
    except Exception:
        pool_age_hours = -1

    websites   = token.get("websites", []) or []
    socials    = token.get("discord_url") or token.get("telegram_handle") or ""
    twitter    = token.get("twitter_handle", "") or ""

    has_website  = int(len(websites) > 0)
    has_twitter  = int(bool(twitter))
    has_social   = int(bool(socials))

    # ── Assemble feature dict ───────────────────────────────────────────────
    features = {
        # identifiers
        "pool_address":           pool.get("address", ""),
        "token_name":             pool.get("name", ""),
        "snapshot_ts":            int(now_ts),

        # liquidity
        "liquidity_usd":          liq_usd,
        "liq_to_fdv_ratio":       liq_usd / (fdv_usd + 1e-6),
        "liq_to_mcap_ratio":      liq_usd / (market_cap + 1e-6),
        "price_change_5m":        p5m,
        "price_change_1h":        p1h,
        "price_change_6h":        p6h,
        "price_change_24h":       p24h,

        # volume
        "vol_5m":                 vol_5m,
        "vol_1h":                 vol_1h,
        "vol_6h":                 vol_6h,
        "vol_24h":                vol_24h,
        "vol_spike_ratio":        vol_spike_ratio,
        "vol_momentum":           vol_momentum,

        # transactions
        "buys_5m":                buys_5m,
        "sells_5m":               sells_5m,
        "buys_1h":                buys_1h,
        "sells_1h":               sells_1h,
        "buys_24h":               buys_24h,
        "sells_24h":              sells_24h,
        "buy_sell_ratio_1h":      buy_sell_ratio_1h,
        "buy_sell_ratio_24h":     buy_sell_ratio_24h,
        "unique_buyer_ratio_1h":  unique_ratio_1h,

        # trade-level
        "avg_trade_size_usd":     avg_trade_size,
        "max_trade_size_usd":     max_trade_size,
        "max_trade_pct_pool":     max_trade_pct_pool,
        "buy_sell_vol_ratio":     buy_sell_vol_ratio,
        "whale_flag":             whale_flag,

        # candlestick / temporal
        "consec_red_candles":     consec_red,
        "avg_body_wick_ratio":    avg_body_wick,

        # metadata
        "pool_age_hours":         round(pool_age_hours, 2),
        "has_website":            has_website,
        "has_twitter":            has_twitter,
        "has_social":             has_social,

        # raw json blobs (for future re-engineering)
        "_raw_pool":              json.dumps(pool),
        "_raw_token":             json.dumps(token),
    }
    return features


# ── SQLite storage ───────────────────────────────────────────────────────────

def init_db(db_path: str = "rug_data.db") -> sqlite3.Connection:
    """Create tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            pool_address     TEXT NOT NULL,
            token_name       TEXT,
            snapshot_ts      INTEGER,

            liquidity_usd    REAL,
            liq_to_fdv_ratio REAL,
            liq_to_mcap_ratio REAL,
            price_change_5m  REAL,
            price_change_1h  REAL,
            price_change_6h  REAL,
            price_change_24h REAL,

            vol_5m           REAL,
            vol_1h           REAL,
            vol_6h           REAL,
            vol_24h          REAL,
            vol_spike_ratio  REAL,
            vol_momentum     REAL,

            buys_5m          INTEGER,
            sells_5m         INTEGER,
            buys_1h          INTEGER,
            sells_1h         INTEGER,
            buys_24h         INTEGER,
            sells_24h        INTEGER,
            buy_sell_ratio_1h  REAL,
            buy_sell_ratio_24h REAL,
            unique_buyer_ratio_1h REAL,

            avg_trade_size_usd REAL,
            max_trade_size_usd REAL,
            max_trade_pct_pool REAL,
            buy_sell_vol_ratio REAL,
            whale_flag         INTEGER,

            consec_red_candles INTEGER,
            avg_body_wick_ratio REAL,

            pool_age_hours   REAL,
            has_website      INTEGER,
            has_twitter      INTEGER,
            has_social       INTEGER,

            label            INTEGER DEFAULT NULL,   -- 1=rug, 0=legit, NULL=unlabeled

            _raw_pool        TEXT,
            _raw_token       TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pool_ts
        ON snapshots (pool_address, snapshot_ts)
    """)
    conn.commit()
    return conn


def save_snapshot(conn: sqlite3.Connection, features: dict):
    """Insert a feature snapshot row."""
    cols = [c for c in features if not c.startswith("_") or c in ("_raw_pool", "_raw_token")]
    placeholders = ", ".join("?" * len(cols))
    col_names    = ", ".join(cols)
    values       = [features[c] for c in cols]
    conn.execute(
        f"INSERT INTO snapshots ({col_names}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


# ── Auto-labeler (outcome-based) ─────────────────────────────────────────────

def label_rugs(conn: sqlite3.Connection,
               liquidity_drop_pct: float = 0.80,
               within_hours: float = 72.0):
    """
    Auto-label pools as rugs if their liquidity dropped by `liquidity_drop_pct`
    within `within_hours` of their first snapshot.

    Labels:
        1 = rug (liquidity drained >= 80% within 72h)
        0 = likely legit (liquidity held up)
        NULL = not enough data yet
    """
    query = """
        SELECT pool_address,
               MIN(snapshot_ts) AS first_ts,
               MAX(liquidity_usd) AS peak_liq,
               MIN(liquidity_usd) AS min_liq
        FROM snapshots
        WHERE label IS NULL
        GROUP BY pool_address
        HAVING (MAX(snapshot_ts) - MIN(snapshot_ts)) >= ?
    """
    min_observation_window = within_hours * 3600
    rows = conn.execute(query, (min_observation_window,)).fetchall()

    for pool_address, first_ts, peak_liq, min_liq in rows:
        if peak_liq and peak_liq > 0:
            drop = (peak_liq - min_liq) / peak_liq
            label = 1 if drop >= liquidity_drop_pct else 0
            conn.execute(
                "UPDATE snapshots SET label = ? WHERE pool_address = ?",
                (label, pool_address),
            )
    conn.commit()
    log.info(f"Auto-labeled {len(rows)} pools.")


# ── Main collection loop ──────────────────────────────────────────────────────

def collect_pool(pool_address: str, token_address: str,
                 conn: sqlite3.Connection):
    """Collect one full snapshot for a pool and store it."""
    log.info(f"Collecting pool {pool_address[:12]}…")

    pool   = get_pool_info(pool_address)
    if not pool:
        log.warning("  ↳ Pool info missing, skipping.")
        return

    token  = get_token_info(token_address) or {}
    ohlcv  = get_ohlcv(pool_address, timeframe="minute", aggregate=5, limit=60) or []
    trades = get_trades(pool_address) or []

    features = engineer_features(pool, token, ohlcv, trades)
    save_snapshot(conn, features)
    log.info(f"  ↳ Saved. Liq=${features['liquidity_usd']:,.0f}  "
             f"Vol1h=${features['vol_1h']:,.0f}  "
             f"Age={features['pool_age_hours']:.1f}h")


def run_discovery_loop(conn: sqlite3.Connection,
                       pages: int = 3,
                       include_new: bool = True):
    """
    Discover pools from trending + new listings, then snapshot each one.
    Call this on a cron or loop to build your dataset over time.
    """
    seen = set()

    # Gather from trending
    for page in range(1, pages + 1):
        log.info(f"Fetching trending pools page {page}…")
        for pool_attr in get_trending_pools(page):
            pool_addr  = pool_attr.get("address", "")
            # relationships aren't in attributes — we derive token from pool name
            # For full token address you'd parse pool relationships; use address as fallback
            token_addr = pool_attr.get("_token_address") or pool_addr
            if pool_addr and pool_addr not in seen:
                seen.add(pool_addr)
                collect_pool(pool_addr, token_addr, conn)

    # Gather from new pools
    if include_new:
        for page in range(1, pages + 1):
            log.info(f"Fetching new pools page {page}…")
            for pool_attr in get_new_pools(page):
                pool_addr  = pool_attr.get("address", "")
                token_addr = pool_attr.get("base_token_address", pool_addr)
                if pool_addr and pool_addr not in seen:
                    seen.add(pool_addr)
                    collect_pool(pool_addr, token_addr, conn)

    log.info(f"Discovery round complete. Collected {len(seen)} pools.")


def collect_watchlist(pool_token_pairs: list[tuple[str, str]],
                      conn: sqlite3.Connection):
    """
    Snapshot a fixed watchlist of pool addresses.
    Use this for known meme coins you're monitoring in real time.

    pool_token_pairs: list of (pool_address, token_address) tuples
    """
    for pool_addr, token_addr in pool_token_pairs:
        collect_pool(pool_addr, token_addr, conn)


# ── Export to CSV ─────────────────────────────────────────────────────────────

def export_csv(conn: sqlite3.Connection, out_path: str = "rug_dataset.csv"):
    """Export labeled snapshots to CSV for model training."""
    import csv
    rows = conn.execute(
        "SELECT * FROM snapshots WHERE label IS NOT NULL ORDER BY snapshot_ts"
    ).fetchall()
    cols = [desc[0] for desc in conn.execute("SELECT * FROM snapshots LIMIT 0").description]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    log.info(f"Exported {len(rows)} labeled rows → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeckoTerminal Solana rug pull collector")
    parser.add_argument("--mode", choices=["discover", "watchlist", "label", "export"],
                        default="discover",
                        help="discover=find new pools, watchlist=snapshot fixed list, "
                             "label=auto-label rugs, export=dump CSV")
    parser.add_argument("--db",    default="rug_data.db",    help="SQLite database path")
    parser.add_argument("--pages", type=int, default=2,       help="Pages to fetch in discovery mode")
    parser.add_argument("--pools", nargs="*",                 help="Pool addresses for watchlist mode")
    parser.add_argument("--tokens", nargs="*",                help="Token addresses for watchlist mode")
    parser.add_argument("--csv",   default="rug_dataset.csv", help="Output CSV path for export")
    args = parser.parse_args()

    conn = init_db(args.db)

    if args.mode == "discover":
        run_discovery_loop(conn, pages=args.pages)

    elif args.mode == "watchlist":
        if not args.pools:
            print("Pass --pools <addr1> <addr2> ...")
        else:
            tokens = args.tokens or args.pools   # fallback if no token addrs given
            pairs  = list(zip(args.pools, tokens))
            collect_watchlist(pairs, conn)

    elif args.mode == "label":
        label_rugs(conn)

    elif args.mode == "export":
        export_csv(conn, args.csv)

    conn.close()
