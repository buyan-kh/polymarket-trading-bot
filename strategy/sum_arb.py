#!/usr/bin/env python3
"""
Sum Arbitrage Strategy Backtest

Logic: If up.ask + down.ask is cheap enough that buying both sides guarantees
profit after fees (one side settles at ~1.00, other at ~0.00). This is a
risk-free arb when the sum is low enough.

Usage:
    python strategy/sum_arb.py data/btc-5m-2026-02-14.jsonl
    python strategy/sum_arb.py data/btc-5m-2026-02-14.jsonl --min-profit 0.01
    python strategy/sum_arb.py data/btc-5m-2026-02-14.jsonl --sweep
"""

import sys
import time as time_mod
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    StrategyBacktester, MarketState, BTPosition, base_argparser,
    print_data_summary, run_sweep,
)


class SumArbBacktester(StrategyBacktester):
    name = "Sum Arb"

    def __init__(self, min_profit: float = 0.005, **kwargs):
        super().__init__(**kwargs)
        self.min_profit = min_profit
        self._traded_markets: set = set()
        self._last_snapshots: dict = {}

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        self._last_snapshots[state.slug] = record

        # Only one arb per market
        if state.slug in self._traded_markets:
            return

        up = record.get("up")
        down = record.get("down")
        if not up or not down:
            return
        if not up.get("ask") or not down.get("ask"):
            return

        up_ask = up["ask"]
        down_ask = down["ask"]

        if up_ask <= 0 or down_ask <= 0 or up_ask >= 1 or down_ask >= 1:
            return

        # Cost to buy both sides (including entry fee)
        from common import taker_fee_rate
        total_cost = (up_ask + down_ask)
        up_fee = self.flat_fee if self.flat_fee is not None else taker_fee_rate(up_ask)
        down_fee = self.flat_fee if self.flat_fee is not None else taker_fee_rate(down_ask)
        cost_with_fee = up_ask / (1 - up_fee) + down_ask / (1 - down_fee)

        # Settlement: winner at 1.00, loser at 0.00 -> gross = 1.00
        # Fee on exit at 1.00 is near zero
        exit_fee = self.flat_fee if self.flat_fee is not None else taker_fee_rate(1.00)
        settlement_net = 1.00 * (1 - exit_fee)

        profit_per_unit = settlement_net - cost_with_fee

        if profit_per_unit < self.min_profit:
            return

        # Buy both sides
        pos_up = self.buy("up", record, tag=f"arb_up@{up_ask:.2f}")
        pos_down = self.buy("down", record, tag=f"arb_dn@{down_ask:.2f}")

        if pos_up or pos_down:
            self._traded_markets.add(state.slug)

    def on_market_change(self, old: str, new: str, t: float) -> None:
        # Settle: determine winner from last snapshot
        last = self._last_snapshots.get(old)
        for pos in list(self.positions):
            if last:
                up_mid = last.get("up", {}).get("mid", 0) if last.get("up") else 0
                down_mid = last.get("down", {}).get("mid", 0) if last.get("down") else 0
                winner = "up" if up_mid > down_mid else "down"
                settle_price = 1.00 if pos.side == winner else 0.00
            else:
                settle_price = pos.entry_price
            self.settle(pos, settle_price, t)


def main():
    parser = base_argparser("Backtest sum arbitrage strategy (buy both sides cheap)")
    parser.add_argument("--min-profit", type=float, default=0.005,
                        help="Min profit per unit to enter (default: 0.005)")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee)

    if args.sweep:
        profits = [0.001, 0.005, 0.01]
        fees = [0.01, 0.015, 0.02]
        sweep_params = [{"min_profit": mp, "fee": fe}
                        for mp, fe in product(profits, fees)]
        run_sweep(SumArbBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = SumArbBacktester(min_profit=args.min_profit, **base_kwargs)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        fee_str = f"{args.fee:.1%}" if args.fee is not None else "curve"
        bt.print_results(results,
                         f"min_profit={args.min_profit} fee={fee_str}",
                         verbose=not args.quiet)
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
