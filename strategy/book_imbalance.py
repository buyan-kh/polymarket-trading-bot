#!/usr/bin/env python3
"""
Book Imbalance Strategy Backtest

Logic: Tight spread = confident market makers. When one side has a much
tighter spread than the other, buy the tight-spread side.

Usage:
    python strategy/book_imbalance.py data/btc-5m-2026-02-14.jsonl
    python strategy/book_imbalance.py data/btc-5m-2026-02-14.jsonl --max-tight 0.01 --min-wide 0.05
    python strategy/book_imbalance.py data/btc-5m-2026-02-14.jsonl --sweep
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


class BookImbalanceBacktester(StrategyBacktester):
    name = "Book Imbalance"

    def __init__(self, max_tight: float = 0.02, min_wide: float = 0.04,
                 tp: float = 0.08, sl: float = 0.05,
                 cooldown: float = 10.0, skip_end: float = 30.0,
                 reverse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_tight = max_tight
        self.min_wide = min_wide
        self.tp = tp
        self.sl = sl
        self.cooldown = cooldown
        self.skip_end = skip_end
        self.reverse = reverse
        self._last_trade_time: float = 0.0

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        # Check TP/SL first
        self.check_tp_sl(record)

        # Skip end of market
        if state.time_remaining < self.skip_end:
            return

        # Don't stack positions
        if self.positions:
            return

        # Cooldown
        t = record["t"]
        if t - self._last_trade_time < self.cooldown:
            return

        up = record.get("up")
        down = record.get("down")
        if not up or not down:
            return
        if not up.get("spread") or not down.get("spread"):
            return

        up_spread = up["spread"]
        down_spread = down["spread"]

        # Look for imbalance: one tight, other wide
        tight_side = None
        if up_spread <= self.max_tight and down_spread >= self.min_wide:
            tight_side = "up"
        elif down_spread <= self.max_tight and up_spread >= self.min_wide:
            tight_side = "down"

        if tight_side:
            # Reverse: buy the wide-spread side instead
            buy_side = ("down" if tight_side == "up" else "up") if self.reverse else tight_side
            pos = self.buy(buy_side, record,
                           tag=f"imb u={up_spread:.2f} d={down_spread:.2f}",
                           tp=self.tp, sl=self.sl)
            if pos:
                self._last_trade_time = t


def main():
    parser = base_argparser("Backtest book imbalance strategy (spread-based)")
    parser.add_argument("--max-tight", type=float, default=0.02,
                        help="Max spread for tight side (default: 0.02)")
    parser.add_argument("--min-wide", type=float, default=0.04,
                        help="Min spread for wide side (default: 0.04)")
    parser.add_argument("--tp", type=float, default=0.08,
                        help="Take profit (default: 0.08)")
    parser.add_argument("--sl", type=float, default=0.05,
                        help="Stop loss (default: 0.05)")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse: buy the wide-spread side instead of tight")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee)

    if args.sweep:
        tights = [0.01, 0.02]
        wides = [0.03, 0.04, 0.05]
        tps = [0.05, 0.08, 0.10]
        sls = [0.03, 0.05, 0.08]
        reverses = [False, True]
        sweep_params = [{"max_tight": mt, "min_wide": mw, "tp": tp, "sl": sl, "reverse": rev}
                        for mt, mw, tp, sl, rev in product(tights, wides, tps, sls, reverses)]
        run_sweep(BookImbalanceBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = BookImbalanceBacktester(
            max_tight=args.max_tight, min_wide=args.min_wide,
            tp=args.tp, sl=args.sl, reverse=args.reverse, **base_kwargs)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        bt.print_results(
            results,
            f"tight<={args.max_tight} wide>={args.min_wide} tp={args.tp} sl={args.sl}",
            verbose=not args.quiet)
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
