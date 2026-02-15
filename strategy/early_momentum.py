#!/usr/bin/env python3
"""
Early Momentum Strategy Backtest

Logic: In the first 30-60s of a market, detect which side pulls ahead
and ride it with TP/SL.

Usage:
    python strategy/early_momentum.py data/btc-5m-2026-02-14.jsonl
    python strategy/early_momentum.py data/btc-5m-2026-02-14.jsonl --window 45 --threshold 0.58
    python strategy/early_momentum.py data/btc-5m-2026-02-14.jsonl --sweep
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


class EarlyMomentumBacktester(StrategyBacktester):
    name = "Early Momentum"

    def __init__(self, window: float = 60.0, threshold: float = 0.55,
                 tp: float = 0.10, sl: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.threshold = threshold
        self.tp = tp
        self.sl = sl
        self._traded_markets: set = set()

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        # Check TP/SL on open positions
        self.check_tp_sl(record)

        # Only enter in early window, one trade per market
        if state.elapsed > self.window:
            return
        if state.slug in self._traded_markets:
            return

        # Look for the leader
        for side in ["up", "down"]:
            data = record.get(side)
            if not data or not data.get("mid"):
                continue
            if data["mid"] >= self.threshold:
                pos = self.buy(side, record, tag=f"mom@{state.elapsed:.0f}s",
                               tp=self.tp, sl=self.sl)
                if pos:
                    self._traded_markets.add(state.slug)
                return


def main():
    parser = base_argparser("Backtest early momentum strategy")
    parser.add_argument("--window", type=float, default=60.0,
                        help="Entry window in seconds (default: 60)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Min mid price to enter (default: 0.55)")
    parser.add_argument("--tp", type=float, default=0.10,
                        help="Take profit (default: 0.10)")
    parser.add_argument("--sl", type=float, default=0.05,
                        help="Stop loss (default: 0.05)")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee)

    if args.sweep:
        windows = [30, 45, 60, 90]
        thresholds = [0.52, 0.55, 0.58, 0.60]
        tps = [0.05, 0.10, 0.15]
        sls = [0.03, 0.05, 0.08]
        sweep_params = [{"window": w, "threshold": th, "tp": tp, "sl": sl}
                        for w, th, tp, sl in product(windows, thresholds, tps, sls)]
        run_sweep(EarlyMomentumBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = EarlyMomentumBacktester(
            window=args.window, threshold=args.threshold,
            tp=args.tp, sl=args.sl, **base_kwargs)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        bt.print_results(
            results,
            f"window={args.window}s threshold={args.threshold} tp={args.tp} sl={args.sl}",
            verbose=not args.quiet)
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
