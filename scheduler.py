"""
Continuous monitoring scheduler for the rug pull pipeline.
Runs discovery + watchlist snapshots on a loop with configurable intervals.

Usage:
    python scheduler.py                     # discovery every 30min
    python scheduler.py --interval 600      # discovery every 10min
    python scheduler.py --watchlist-file pools.txt
"""

import time
import logging
import argparse
from pathlib import Path
from collector import init_db, run_discovery_loop, collect_watchlist, label_rugs, export_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_watchlist(path: str) -> list[tuple[str, str]]:
    """
    Load a watchlist file. Each line: pool_address,token_address
    Lines starting with # are comments.
    """
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
        else:
            pairs.append((parts[0].strip(), parts[0].strip()))  # fallback
    return pairs


def run(db: str, interval: int, pages: int, watchlist_file: str | None,
        auto_label_every: int, export_every: int):

    conn = init_db(db)
    watchlist = load_watchlist(watchlist_file) if watchlist_file else []
    cycle = 0

    log.info(f"Scheduler started. Interval={interval}s, Pages={pages}, "
             f"Watchlist={len(watchlist)} pools")

    try:
        while True:
            cycle += 1
            log.info(f"─── Cycle {cycle} ──────────────────────")

            # Discovery round
            run_discovery_loop(conn, pages=pages)

            # Watchlist round
            if watchlist:
                log.info(f"Snapshotting {len(watchlist)} watchlist pools…")
                collect_watchlist(watchlist, conn)

            # Auto-label every N cycles
            if cycle % auto_label_every == 0:
                log.info("Running auto-labeler…")
                label_rugs(conn)

            # Export CSV every N cycles
            if cycle % export_every == 0:
                log.info("Exporting dataset CSV…")
                export_csv(conn, "rug_dataset.csv")

            log.info(f"Cycle {cycle} done. Sleeping {interval}s…")
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Interrupted — exporting final dataset…")
        label_rugs(conn)
        export_csv(conn, "rug_dataset.csv")
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",              default="rug_data.db")
    parser.add_argument("--interval",        type=int, default=1800,  help="Seconds between cycles")
    parser.add_argument("--pages",           type=int, default=2,     help="Discovery pages per cycle")
    parser.add_argument("--watchlist-file",  default=None,            help="Path to pool watchlist CSV")
    parser.add_argument("--auto-label-every",type=int, default=4,     help="Label every N cycles")
    parser.add_argument("--export-every",    type=int, default=8,     help="Export CSV every N cycles")
    args = parser.parse_args()

    run(
        db=args.db,
        interval=args.interval,
        pages=args.pages,
        watchlist_file=args.watchlist_file,
        auto_label_every=args.auto_label_every,
        export_every=args.export_every,
    )
