#!/usr/bin/env python3
"""
Fair Value Strategy Backtest

Uses Black-Scholes binary option pricing to compute theoretical fair value
from BTC spot price, strike (price at market open), time remaining, and
realized volatility. Trades when Polymarket price deviates from fair value
by more than fees + spread.

This is the first strategy that anchors to external price data (Binance)
rather than trading purely on Polymarket order book patterns.

Fair value formula (cash-or-nothing binary call):
    fv_up = N(d2) where d2 = [ln(S/K) + (-sigma^2/2)*T] / (sigma * sqrt(T))

Usage:
    python strategy/fair_value.py data/btc-5m-2026-02-14.jsonl
    python strategy/fair_value.py data/btc-5m-2026-02-14.jsonl --edge 0.03
    python strategy/fair_value.py data/btc-5m-2026-02-14.jsonl --vol-window 120
    python strategy/fair_value.py data/btc-5m-2026-02-14.jsonl --sweep
"""

import sys
import json
import time as time_mod
from pathlib import Path
from itertools import product
from typing import Optional, Dict

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    StrategyBacktester, MarketState, taker_fee_rate,
    base_argparser, print_data_summary, run_sweep,
)
from lib.binance_feed import (
    fetch_klines_range, build_price_lookup, get_price_at,
    compute_rolling_vol,
)


# ── Binary option fair value ────────────────────────────────────────

def binary_call_fv(S: float, K: float, T: float, sigma: float) -> float:
    """
    Fair value of a cash-or-nothing binary call option.

    Returns probability that S > K at expiry under risk-neutral measure.
    For a 5-minute crypto market, this is essentially the real-world probability.

    Args:
        S: Current spot price
        K: Strike price (price at market open)
        T: Time to expiry in years
        sigma: Annualized volatility

    Returns:
        Fair value between 0 and 1
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)

    d2 = (np.log(S / K) + (-sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d2))


def kelly_size(p_true: float, ask_price: float, fee_rate: float,
               fraction: float = 0.25) -> float:
    """
    Kelly criterion sizing for a binary bet.

    Args:
        p_true: True probability of winning (from fair value model)
        ask_price: Price you'd pay (Polymarket ask)
        fee_rate: Fee rate on entry
        fraction: Kelly fraction (< 1 for fractional Kelly)

    Returns:
        Fraction of bankroll to bet (0 if no edge)
    """
    cost = ask_price * (1 + fee_rate)
    if cost >= 1.0 or cost <= 0:
        return 0.0
    b = (1.0 - cost) / cost  # net odds
    q = 1 - p_true
    f = (b * p_true - q) / b
    return max(0.0, f * fraction)


# ── Strategy ────────────────────────────────────────────────────────

class FairValueBacktester(StrategyBacktester):
    name = "FairValue"

    def __init__(
        self,
        edge_threshold: float = 0.05,
        vol_window: int = 300,
        kelly_frac: float = 0.25,
        tp: float = 0.06,
        sl: float = 0.04,
        min_time_remaining: float = 30.0,
        max_time_remaining: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_threshold = edge_threshold
        self.vol_window = vol_window
        self.kelly_frac = kelly_frac
        self.tp = tp
        self.sl = sl
        self.min_time_remaining = min_time_remaining
        self.max_time_remaining = max_time_remaining

        # State populated by pre-processing
        self._btc_prices: dict = {}          # timestamp -> BTC close price
        self._vol_lookup: dict = {}           # timestamp -> realized vol
        self._strike_prices: dict = {}        # slug -> strike price (BTC at market open)
        self._traded_markets: set = set()     # slugs we've traded in
        self._last_snapshots: dict = {}       # slug -> last record
        self._data_loaded: bool = False

    def _load_binance_data(self, data_path: str) -> None:
        """Load Binance price data - from embedded JSONL or REST API fallback."""
        if self._data_loaded:
            return

        # Scan JSONL to find time range, slugs, and embedded BTC prices
        first_t = None
        last_t = None
        slugs = set()
        embedded_prices = []  # (timestamp, price) from "btc" field

        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                t = record.get("t", 0)
                if first_t is None or t < first_t:
                    first_t = t
                if last_t is None or t > last_t:
                    last_t = t
                slug = record.get("slug", "")
                if slug:
                    slugs.add(slug)
                # Collect embedded BTC prices
                btc = record.get("btc")
                if btc:
                    embedded_prices.append((int(t), float(btc)))

        if not first_t or not last_t:
            return

        if embedded_prices:
            # Use embedded prices from recorder (real 1s data)
            print(f"  Using embedded BTC prices: {len(embedded_prices)} data points")
            self._btc_prices = {ts: p for ts, p in embedded_prices}

            # Build kline-like tuples for vol computation: (ts, o, h, l, c, v)
            klines = [(ts, p, p, p, p, 0.0) for ts, p in embedded_prices]
            klines.sort(key=lambda x: x[0])

            # Deduplicate by timestamp (keep last price per second)
            deduped = {}
            for k in klines:
                deduped[k[0]] = k
            klines = sorted(deduped.values(), key=lambda x: x[0])

            print(f"  Computing rolling vol (window={self.vol_window}s) from {len(klines)} bars...")
            self._vol_lookup = compute_rolling_vol(klines, self.vol_window)
        else:
            # Fallback: fetch from Binance REST API (old data files)
            buffer = max(self.vol_window + 60, 600)
            start_ts = first_t - buffer
            end_ts = last_t + 10

            print(f"  No embedded BTC prices — fetching from Binance REST API...")
            print(f"  Fetching Binance 1s klines: {int(end_ts - start_ts)}s range...")
            klines = fetch_klines_range("BTCUSDT", start_ts, end_ts, interval="1s")
            print(f"  Got {len(klines)} klines")

            self._btc_prices = build_price_lookup(klines)

            print(f"  Computing rolling vol (window={self.vol_window}s)...")
            self._vol_lookup = compute_rolling_vol(klines, self.vol_window)

        # Extract strike prices from slugs
        for slug in slugs:
            parts = slug.split("-")
            if parts[-1].isdigit():
                market_open_ts = int(parts[-1])
                strike = get_price_at(self._btc_prices, market_open_ts)
                if strike:
                    self._strike_prices[slug] = strike

        print(f"  Strikes: {len(self._strike_prices)} markets")
        self._data_loaded = True

    def run(self, data_path: str) -> dict:
        """Override run to pre-load Binance data first."""
        self._load_binance_data(data_path)
        return super().run(data_path)

    def _get_vol(self, t: float) -> float:
        """Get realized vol at timestamp, with fallback."""
        ts = int(t)
        if ts in self._vol_lookup:
            return self._vol_lookup[ts]
        # Try nearby
        for offset in range(1, 6):
            if ts - offset in self._vol_lookup:
                return self._vol_lookup[ts - offset]
            if ts + offset in self._vol_lookup:
                return self._vol_lookup[ts + offset]
        return 0.0

    def on_snapshot(self, record: dict, state: MarketState) -> None:
        self._last_snapshots[state.slug] = record

        # Check TP/SL on existing positions
        self.check_tp_sl(record)

        # Time filters
        if state.time_remaining < self.min_time_remaining:
            return
        if self.max_time_remaining > 0 and state.time_remaining > self.max_time_remaining:
            return

        # Only one trade per market
        if state.slug in self._traded_markets:
            return

        # Get BTC spot price
        S = get_price_at(self._btc_prices, record["t"])
        if not S:
            return

        # Get strike
        K = self._strike_prices.get(state.slug)
        if not K:
            return

        # Time to expiry in years
        T = state.time_remaining / (365.25 * 24 * 3600)

        # Realized vol
        sigma = self._get_vol(record["t"])
        if sigma <= 0:
            return

        # Compute fair values
        fv_up = binary_call_fv(S, K, T, sigma)
        fv_down = 1.0 - fv_up

        # Check both sides for edge
        for side, fv in [("up", fv_up), ("down", fv_down)]:
            data = record.get(side)
            if not data or not data.get("ask"):
                continue

            ask = data["ask"]
            if ask <= 0 or ask >= 1.0:
                continue

            fee = taker_fee_rate(ask)
            edge = fv - ask - (ask * fee)

            if edge >= self.edge_threshold:
                # Kelly sizing
                k_frac = kelly_size(fv, ask, fee, self.kelly_frac)
                if k_frac <= 0:
                    continue

                # Override bet_fraction with Kelly size
                old_bf = self.bet_fraction
                self.bet_fraction = min(k_frac, 0.25)  # cap at 25%

                pos = self.buy(
                    side, record,
                    tp=self.tp, sl=self.sl,
                    tag=f"fv={fv:.3f} edge={edge:.3f} σ={sigma:.1%}",
                )

                self.bet_fraction = old_bf

                if pos:
                    self._traded_markets.add(state.slug)
                    return

    def on_market_change(self, old: str, new: str, t: float) -> None:
        """Settle positions at actual settlement price."""
        for pos in list(self.positions):
            last = self._last_snapshots.get(old)
            if last:
                # Determine winner from last BTC price vs strike
                btc_price = get_price_at(self._btc_prices, t)
                strike = self._strike_prices.get(old)

                if btc_price and strike:
                    winner = "up" if btc_price > strike else "down"
                else:
                    # Fallback: use Polymarket prices
                    up_mid = last.get("up", {}).get("mid", 0) if last.get("up") else 0
                    down_mid = last.get("down", {}).get("mid", 0) if last.get("down") else 0
                    winner = "up" if up_mid > down_mid else "down"

                # Settlement: winner gets ~1.00, loser gets ~0.00
                settle_price = 1.00 if pos.side == winner else 0.00
            else:
                settle_price = pos.entry_price

            self.settle(pos, settle_price, t)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = base_argparser(
        "Backtest fair value strategy (Black-Scholes binary pricing vs Polymarket)"
    )
    parser.add_argument("--edge", type=float, default=0.05,
                        help="Min edge to enter (default: 0.05)")
    parser.add_argument("--vol-window", type=int, default=300,
                        help="Realized vol window in seconds (default: 300)")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25)")
    parser.add_argument("--tp", type=float, default=0.06,
                        help="Take profit offset (default: 0.06)")
    parser.add_argument("--sl", type=float, default=0.04,
                        help="Stop loss offset (default: 0.04)")
    parser.add_argument("--min-time", type=float, default=30.0,
                        help="Min seconds remaining to trade (default: 30)")
    parser.add_argument("--max-time", type=float, default=0.0,
                        help="Max seconds remaining to trade (default: 0 = no limit)")
    args = parser.parse_args()

    print_data_summary(args.data)

    base_kwargs = dict(balance=args.balance, bet_fraction=args.bet, fee=args.fee,
                       max_bet=args.max_bet)

    if args.sweep:
        edges = [0.02, 0.03, 0.05, 0.07, 0.10]
        vol_windows = [60, 120, 300, 600]
        kellys = [0.10, 0.25, 0.50]
        sweep_params = [
            {
                "edge_threshold": e,
                "vol_window": vw,
                "kelly_frac": k,
                "tp": args.tp,
                "sl": args.sl,
                "min_time_remaining": args.min_time,
                "max_time_remaining": args.max_time,
            }
            for e, vw, k in product(edges, vol_windows, kellys)
        ]
        print(f"\n  Sweeping {len(sweep_params)} parameter combinations...")
        run_sweep(FairValueBacktester, sweep_params, args.data, base_kwargs)
    else:
        bt = FairValueBacktester(
            edge_threshold=args.edge,
            vol_window=args.vol_window,
            kelly_frac=args.kelly,
            tp=args.tp,
            sl=args.sl,
            min_time_remaining=args.min_time,
            max_time_remaining=args.max_time,
            **base_kwargs,
        )
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        bt.print_results(
            results,
            f"edge={args.edge} vol={args.vol_window}s kelly={args.kelly}",
            verbose=not args.quiet,
        )
        print(f"  Ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
