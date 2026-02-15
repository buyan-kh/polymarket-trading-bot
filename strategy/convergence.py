#!/usr/bin/env python3
"""
Convergence Strategy Backtest

Normal mode: Near expiry, if one side is at 0.85+, buy it — likely winner,
settles at ~1.00.

Reverse mode (--reverse): Buy the UNDERDOG (cheap side) near expiry. Loses
small most of the time but wins huge on upsets (buy at 0.05, settle at 0.99).

Usage:
    python strategy/convergence.py data/btc-5m-2026-02-14.jsonl
    python strategy/convergence.py data/btc-5m-2026-02-14.jsonl --reverse
    python strategy/convergence.py data/btc-5m-2026-02-14.jsonl --reverse --threshold 0.90 --time 90
    python strategy/convergence.py data/btc-5m-2026-02-14.jsonl --sweep
"""

import sys
import time as time_mod
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    StrategyBacktester, MarketState, base_argparser,
    print_data_summary, run_sweep,
)


class ConvergenceBacktester(StrategyBacktester):
    name = "Convergence"

    def __init__(self, threshold: float = 0.85, time_window: float = 60.0,
                 reverse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.time_window = time_window
        self.reverse = reverse
        self.name = "Convergence (reverse)" if reverse else "Convergence"
        self._traded_markets: set = set()
        self._last_snapshots: dict = {}  # slug -> last record

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        self._last_snapshots[state.slug] = record

        # Only trade near expiry
        if state.time_remaining > self.time_window or state.time_remaining <= 0:
            return

        # Only one trade per market
        if state.slug in self._traded_markets:
            return

        if self.reverse:
            # Buy the underdog — the cheap side when leader is confident
            for side in ["up", "down"]:
                data = record.get(side)
                if not data or not data.get("mid"):
                    continue
                if data["mid"] >= self.threshold:
                    # Leader found — buy the OTHER side
                    other = "down" if side == "up" else "up"
                    other_data = record.get(other)
                    if not other_data or not other_data.get("ask"):
                        continue
                    pos = self.buy(other, record,
                                   tag=f"rev@{other_data.get('mid', 0):.2f}")
                    if pos:
                        self._traded_markets.add(state.slug)
                    return
        else:
            # Normal: buy the leader
            for side in ["up", "down"]:
                data = record.get(side)
                if not data or not data.get("mid"):
                    continue
                if data["mid"] >= self.threshold:
                    pos = self.buy(side, record, tag=f"conv@{data['mid']:.2f}")
                    if pos:
                        self._traded_markets.add(state.slug)
                    return

    def on_market_change(self, old: str, new: str, t: float) -> None:
        # Settle open positions instead of force-closing
        for pos in list(self.positions):
            # Determine winner from last snapshot
            last = self._last_snapshots.get(old)
            if last:
                up_mid = last.get("up", {}).get("mid", 0) if last.get("up") else 0
                down_mid = last.get("down", {}).get("mid", 0) if last.get("down") else 0
                winner = "up" if up_mid > down_mid else "down"
                settle_price = 0.99 if pos.side == winner else 0.01
            else:
                settle_price = pos.entry_price
            self.settle(pos, settle_price, t)


def main():
    parser = base_argparser("Backtest convergence strategy (buy leader near expiry)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Min mid price to enter (default: 0.85)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Seconds before expiry to start looking (default: 60)")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse: buy the underdog instead of the leader")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee)

    if args.sweep:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        times = [30, 45, 60, 90, 120]
        reverses = [False, True]
        sweep_params = [{"threshold": th, "time_window": tw, "reverse": rev}
                        for th, tw, rev in product(thresholds, times, reverses)]
        run_sweep(ConvergenceBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = ConvergenceBacktester(
            threshold=args.threshold, time_window=args.time,
            reverse=args.reverse, **base_kwargs)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        mode = "REVERSE" if args.reverse else "normal"
        bt.print_results(results,
                         f"{mode} threshold={args.threshold} time={args.time}s",
                         verbose=not args.quiet)
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
