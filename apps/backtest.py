#!/usr/bin/env python3
"""
Backtester - Replay recorded data through the flash crash strategy.

Reads JSONL data from the recorder and simulates trading with
configurable parameters. Uses realistic bid/ask pricing and fees.

Usage:
    python apps/backtest.py data/btc-5m-2026-02-14.jsonl
    python apps/backtest.py data/btc-5m-2026-02-14.jsonl --drop 0.05
    python apps/backtest.py data/btc-5m-2026-02-14.jsonl --sweep

Examples:
    # Single run with specific params
    python apps/backtest.py data/btc-5m-2026-02-14.jsonl --drop 0.05 --tp 0.05 --sl 0.03

    # Sweep drop thresholds to find the best one
    python apps/backtest.py data/btc-5m-2026-02-14.jsonl --sweep
"""

import sys
import json
import argparse
import time as time_mod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from common import taker_fee_rate


@dataclass
class BTPosition:
    """Backtesting position."""
    side: str
    entry_price: float  # ask price at entry
    size: float  # shares
    cost: float  # usdc spent (including fee)
    entry_time: float
    tp_price: float
    sl_price: float


@dataclass
class BTTrade:
    """Completed trade."""
    n: int
    side: str
    entry: float
    exit: float
    pnl: float
    hold: float
    result: str
    balance: float
    market: str


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    balance: float = 10.0
    bet_fraction: float = 0.10
    drop_threshold: float = 0.30
    lookback_seconds: int = 10
    take_profit: float = 0.10
    stop_loss: float = 0.05
    fee: Optional[float] = None
    no_trade_seconds: int = 30
    duration_minutes: int = 5


@dataclass
class PricePoint:
    timestamp: float
    price: float


class Backtester:
    """Replays recorded data through flash crash strategy logic."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.balance
        self.start_balance = config.balance

        # State
        self.position: Optional[BTPosition] = None
        self.trades: List[BTTrade] = []
        self.current_market: str = ""
        self.market_end: float = 0.0

        # Price history (for flash crash detection)
        self._history: Dict[str, deque] = {
            "up": deque(maxlen=200),
            "down": deque(maxlen=200),
        }

        # Stats
        self.crashes_detected = 0
        self.crashes_skipped_notrade = 0
        self.crashes_skipped_position = 0

    def run(self, data_path: str) -> dict:
        """Run backtest on recorded data file."""
        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                event = record.get("event")

                if event == "session_start":
                    self.config.duration_minutes = record.get("duration", 5)
                    continue
                elif event == "session_end":
                    continue
                elif event == "market_change":
                    self._on_market_change(record)
                    continue

                # Regular price snapshot
                self._process_snapshot(record)

        # Close any remaining position
        if self.position:
            self._force_close(self._last_time)

        return self._get_results()

    def _on_market_change(self, record: dict) -> None:
        """Handle market change."""
        # Close open position at last known bid
        if self.position:
            self._force_close(record["t"])

        self.current_market = record.get("new", "")
        self._history["up"].clear()
        self._history["down"].clear()

    def _process_snapshot(self, record: dict) -> None:
        """Process a single price snapshot."""
        t = record["t"]
        slug = record.get("slug", "")
        self._last_time = t

        # Track market
        if slug != self.current_market:
            self.current_market = slug

        # Parse end time
        end_str = record.get("end", "")
        if end_str:
            from datetime import datetime, timezone
            try:
                end_str = end_str.replace("Z", "+00:00")
                self.market_end = datetime.fromisoformat(end_str).timestamp()
            except Exception:
                pass

        # Record prices (mid for crash detection)
        for side in ["up", "down"]:
            data = record.get(side)
            if data and data.get("mid"):
                self._history[side].append(PricePoint(t, data["mid"]))

        # Check TP/SL on open position
        if self.position:
            self._check_exit(record, t)

        # Detect flash crash (only if no position)
        if not self.position:
            self._detect_and_trade(record, t)

    def _time_remaining(self, t: float) -> float:
        """Get seconds remaining in current market."""
        if self.market_end <= 0:
            return 999
        return max(0, self.market_end - t)

    def _in_no_trade_zone(self, t: float) -> bool:
        """Check if in no-trade zone."""
        return self._time_remaining(t) <= self.config.no_trade_seconds

    def _detect_crash(self, side: str, t: float) -> Optional[float]:
        """Detect flash crash on a side. Returns drop amount or None."""
        history = self._history[side]
        if len(history) < 2:
            return None

        current = history[-1].price
        cutoff = t - self.config.lookback_seconds

        # Find oldest price in lookback window
        old_price = None
        for p in history:
            if p.timestamp >= cutoff:
                old_price = p.price
                break

        if old_price is None:
            return None

        drop = old_price - current
        if drop >= self.config.drop_threshold:
            return drop
        return None

    def _detect_and_trade(self, record: dict, t: float) -> None:
        """Detect flash crash and execute trade if found."""
        if self._in_no_trade_zone(t):
            return

        for side in ["up", "down"]:
            drop = self._detect_crash(side, t)
            if drop is None:
                continue

            self.crashes_detected += 1

            # Clear history (anti-cascade)
            self._history[side].clear()

            # Get ask price for entry
            data = record.get(side)
            if not data or not data.get("ask") or data["ask"] >= 1:
                continue

            ask = data["ask"]
            usdc_amount = self.balance * self.config.bet_fraction
            fee_rate = self.config.fee if self.config.fee is not None else taker_fee_rate(ask)
            usdc_after_fee = usdc_amount * (1 - fee_rate)
            size = usdc_after_fee / ask

            self.position = BTPosition(
                side=side,
                entry_price=ask,
                size=size,
                cost=usdc_amount,
                entry_time=t,
                tp_price=ask + self.config.take_profit,
                sl_price=ask - self.config.stop_loss,
            )
            return  # Only one trade at a time

    def _check_exit(self, record: dict, t: float) -> None:
        """Check TP/SL on open position."""
        pos = self.position
        data = record.get(pos.side)
        if not data or not data.get("bid"):
            return

        bid = data["bid"]

        if bid >= pos.tp_price:
            self._close_position(bid, t, "W")
        elif bid <= pos.sl_price:
            self._close_position(bid, t, "L")

    def _close_position(self, exit_bid: float, t: float, result: str) -> None:
        """Close position at bid price with fee."""
        pos = self.position
        gross = pos.size * exit_bid
        fee_rate = self.config.fee if self.config.fee is not None else taker_fee_rate(exit_bid)
        fee = gross * fee_rate
        net_proceeds = gross - fee
        pnl = net_proceeds - pos.cost

        self.balance += pnl
        hold = t - pos.entry_time

        self.trades.append(BTTrade(
            n=len(self.trades) + 1,
            side=pos.side.upper(),
            entry=pos.entry_price,
            exit=exit_bid,
            pnl=pnl,
            hold=hold,
            result=result,
            balance=self.balance,
            market=self.current_market,
        ))

        self.position = None

    def _force_close(self, t: float) -> None:
        """Force close at entry price (market expired)."""
        pos = self.position
        exit_gross = pos.size * pos.entry_price
        fee_rate = self.config.fee if self.config.fee is not None else taker_fee_rate(pos.entry_price)
        exit_fee = exit_gross * fee_rate
        pnl = exit_gross - exit_fee - pos.cost

        self.balance += pnl
        hold = t - pos.entry_time

        self.trades.append(BTTrade(
            n=len(self.trades) + 1,
            side=pos.side.upper(),
            entry=pos.entry_price,
            exit=pos.entry_price,
            pnl=pnl,
            hold=hold,
            result="X",  # expired
            balance=self.balance,
            market=self.current_market,
        ))
        self.position = None

    def _get_results(self) -> dict:
        """Compile results."""
        wins = sum(1 for t in self.trades if t.result == "W")
        losses = sum(1 for t in self.trades if t.result == "L")
        expired = sum(1 for t in self.trades if t.result == "X")
        total_pnl = self.balance - self.start_balance
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        return {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "expired": expired,
            "total_pnl": total_pnl,
            "return_pct": total_pnl / self.start_balance * 100,
            "win_rate": win_rate,
            "final_balance": self.balance,
            "start_balance": self.start_balance,
            "crashes_detected": self.crashes_detected,
            "trade_log": self.trades,
        }


def print_results(results: dict, config: BacktestConfig, verbose: bool = True) -> None:
    """Print backtest results."""
    if verbose and results["trade_log"]:
        print(f"\n  {'#':>3}  {'Side':4}  {'Entry':>8}  {'Exit':>8}  {'PnL':>9}  {'Hold':>6}  Result  {'Balance':>10}")
        print(f"  {'---':>3}  {'----':4}  {'--------':>8}  {'--------':>8}  {'---------':>9}  {'------':>6}  ------  {'----------':>10}")
        for t in results["trade_log"]:
            hold = f"{t.hold:.0f}s" if t.hold < 60 else f"{t.hold / 60:.1f}m"
            print(f"  {t.n:>3}  {t.side:4}  {t.entry:>8.4f}  {t.exit:>8.4f}  ${t.pnl:>+8.2f}  {hold:>6}  {'  ' + t.result:>6}  ${t.balance:>9.2f}")

    fee_str = f"{config.fee:.0%}" if config.fee is not None else "curve"
    print(f"\n  Results (drop={config.drop_threshold:.2f} tp={config.take_profit:.2f} sl={config.stop_loss:.2f} fee={fee_str}):")
    print(f"    Trades: {results['trades']} (W:{results['wins']} L:{results['losses']} X:{results['expired']})")
    print(f"    Win rate: {results['win_rate']:.1f}%")
    print(f"    PnL: ${results['total_pnl']:+.2f} ({results['return_pct']:+.1f}%)")
    print(f"    Balance: ${results['start_balance']:.2f} -> ${results['final_balance']:.2f}")
    print(f"    Crashes detected: {results['crashes_detected']}")


def run_sweep(data_path: str, base_config: BacktestConfig) -> None:
    """Sweep parameters and show comparison table."""
    drops = [0.30]
    tps = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    sls = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]

    all_results = []

    for drop in drops:
        for tp in tps:
            for sl in sls:
                cfg = BacktestConfig(
                    balance=base_config.balance,
                    bet_fraction=base_config.bet_fraction,
                    drop_threshold=drop,
                    lookback_seconds=base_config.lookback_seconds,
                    take_profit=tp,
                    stop_loss=sl,
                    fee=base_config.fee,
                    no_trade_seconds=base_config.no_trade_seconds,
                )
                bt = Backtester(cfg)
                r = bt.run(data_path)
                all_results.append((cfg, r))

    # Sort by return (worst first)
    all_results.sort(key=lambda x: x[1]["return_pct"])

    # Print table
    print(f"\n  {'Drop':>6}  {'TP':>6}  {'SL':>6}  {'Trades':>6}  {'W':>3}  {'L':>3}  {'WinR':>6}  {'PnL':>9}  {'Return':>8}  {'Balance':>9}")
    print(f"  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}  {'---':>3}  {'---':>3}  {'------':>6}  {'---------':>9}  {'--------':>8}  {'---------':>9}")

    for cfg, r in all_results[:30]:  # Worst 30
        print(
            f"  {cfg.drop_threshold:>6.2f}  {cfg.take_profit:>6.2f}  {cfg.stop_loss:>6.2f}  "
            f"{r['trades']:>6}  {r['wins']:>3}  {r['losses']:>3}  "
            f"{r['win_rate']:>5.1f}%  ${r['total_pnl']:>+8.2f}  "
            f"{r['return_pct']:>+7.1f}%  ${r['final_balance']:>8.2f}"
        )

    print(f"\n  Showing worst 30 of {len(all_results)} parameter combinations")


def main():
    parser = argparse.ArgumentParser(description="Backtest flash crash strategy")
    parser.add_argument("data", help="Path to recorded JSONL data file")
    parser.add_argument("--balance", type=float, default=10.0)
    parser.add_argument("--bet", type=float, default=0.10, help="Bet fraction (default: 0.10)")
    parser.add_argument("--drop", type=float, default=0.30, help="Drop threshold (default: 0.30)")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--tp", type=float, default=0.10, help="Take profit (default: 0.10)")
    parser.add_argument("--sl", type=float, default=0.05, help="Stop loss (default: 0.05)")
    parser.add_argument("--fee", type=float, default=None, help="Flat fee override (default: Polymarket fee curve)")
    parser.add_argument("--no-trade", type=int, default=30, help="No-trade zone seconds (default: 30)")
    parser.add_argument("--sweep", action="store_true", help="Sweep all parameter combinations")
    parser.add_argument("--quiet", action="store_true", help="Hide individual trade log")

    args = parser.parse_args()

    config = BacktestConfig(
        balance=args.balance,
        bet_fraction=args.bet,
        drop_threshold=args.drop,
        lookback_seconds=args.lookback,
        take_profit=args.tp,
        stop_loss=args.sl,
        fee=args.fee,
        no_trade_seconds=args.no_trade,
    )

    # Count data
    with open(args.data) as f:
        lines = f.readlines()
    snapshots = sum(1 for l in lines if '"event"' not in l)
    changes = sum(1 for l in lines if '"market_change"' in l)
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    hours = (last["t"] - first["t"]) / 3600

    print(f"  Data: {args.data}")
    print(f"  Snapshots: {snapshots:,} | Markets: {changes} | Duration: {hours:.1f} hours")

    if args.sweep:
        run_sweep(args.data, config)
    else:
        bt = Backtester(config)
        start = time_mod.time()
        results = bt.run(args.data)
        elapsed = time_mod.time() - start
        print_results(results, config, verbose=not args.quiet)
        print(f"  Backtest ran in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
