#!/usr/bin/env python3
"""
Lead-Lag Strategy Backtest

Logic: UP + DOWN should sum to ~1.0. When the sum deviates, one side moved
but the other hasn't caught up yet. Buy the lagging (underpriced) side.

Usage:
    python strategy/lead_lag.py data/btc-5m-2026-02-14.jsonl
    python strategy/lead_lag.py data/btc-5m-2026-02-14.jsonl --deviation 0.05 --tp 0.03
    python strategy/lead_lag.py data/btc-5m-2026-02-14.jsonl --sweep
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


class LeadLagBacktester(StrategyBacktester):
    name = "Lead-Lag"

    def __init__(self, deviation: float = 0.03, tp: float = 0.03,
                 sl: float = 0.03, skip_start: float = 10.0,
                 skip_end: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.deviation = deviation
        self.tp = tp
        self.sl = sl
        self.skip_start = skip_start
        self.skip_end = skip_end

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        # Check TP/SL first
        self.check_tp_sl(record)

        # Skip noisy periods
        if state.elapsed < self.skip_start:
            return
        if state.time_remaining < self.skip_end:
            return

        # Don't stack positions in same market
        if self.positions:
            return

        up = record.get("up")
        down = record.get("down")
        if not up or not down:
            return
        if not up.get("mid") or not down.get("mid"):
            return

        up_mid = up["mid"]
        down_mid = down["mid"]
        price_sum = up_mid + down_mid

        # Check for deviation below 1.0 (sum too low -> one side is underpriced)
        if price_sum < (1.0 - self.deviation):
            # Buy the side that's more underpriced
            # Expected: each side should be (1.0 - other_mid)
            up_expected = 1.0 - down_mid
            down_expected = 1.0 - up_mid
            up_gap = up_expected - up_mid   # positive = underpriced
            down_gap = down_expected - down_mid

            if up_gap > down_gap:
                side = "up"
                gap = up_gap
            else:
                side = "down"
                gap = down_gap

            pos = self.buy(side, record,
                           tag=f"lag sum={price_sum:.3f} gap={gap:.3f}",
                           tp=self.tp, sl=self.sl)

        # Also check sum > 1.0 (one side is overpriced, other is "lagging up")
        elif price_sum > (1.0 + self.deviation):
            # Sum is too high - one side is overpriced
            # We can't short, but the overpriced side may drop
            # Buy the side that's closer to fair value (less overpriced)
            up_expected = 1.0 - down_mid
            down_expected = 1.0 - up_mid
            up_gap = up_mid - up_expected   # positive = overpriced
            down_gap = down_mid - down_expected

            # Buy the LESS overpriced side (it hasn't caught up yet)
            if up_gap < down_gap:
                side = "up"
            else:
                side = "down"

            pos = self.buy(side, record,
                           tag=f"lag sum={price_sum:.3f}",
                           tp=self.tp, sl=self.sl)


def main():
    parser = base_argparser("Backtest lead-lag strategy (price sum deviation)")
    parser.add_argument("--deviation", type=float, default=0.03,
                        help="Sum deviation threshold (default: 0.03)")
    parser.add_argument("--tp", type=float, default=0.03,
                        help="Take profit (default: 0.03)")
    parser.add_argument("--sl", type=float, default=0.03,
                        help="Stop loss (default: 0.03)")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee)

    if args.sweep:
        devs = [0.02, 0.03, 0.05]
        tps = [0.02, 0.03, 0.05]
        sls = [0.02, 0.03, 0.05]
        sweep_params = [{"deviation": d, "tp": tp, "sl": sl}
                        for d, tp, sl in product(devs, tps, sls)]
        run_sweep(LeadLagBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = LeadLagBacktester(
            deviation=args.deviation, tp=args.tp, sl=args.sl, **base_kwargs)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        bt.print_results(
            results,
            f"deviation={args.deviation} tp={args.tp} sl={args.sl}",
            verbose=not args.quiet)
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
