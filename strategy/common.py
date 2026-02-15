"""
Shared backtest infrastructure for strategy backtesting.

Provides base classes and utilities used by all strategy scripts.
Extract from apps/backtest.py with support for multiple positions
and abstract on_snapshot() dispatch.
"""

import json
import argparse
import time as time_mod
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


@dataclass
class BTPosition:
    """Backtesting position."""
    side: str
    entry_price: float  # ask price at entry
    size: float         # shares
    cost: float         # usdc spent (including fee)
    entry_time: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    tag: str = ""       # strategy-specific label


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
    tag: str = ""


@dataclass
class MarketState:
    """State passed to on_snapshot for each record."""
    slug: str
    end_ts: float
    time_remaining: float
    elapsed: float          # seconds since first snapshot in this market
    snapshot_count: int     # how many snapshots seen in this market


class StrategyBacktester(ABC):
    """Base class for strategy backtests."""

    name: str = "strategy"

    def __init__(self, balance: float = 10.0, bet_fraction: float = 0.10,
                 fee: float = 0.02, **kwargs):
        self.balance = balance
        self.start_balance = balance
        self.bet_fraction = bet_fraction
        self.fee = fee

        # State
        self.positions: List[BTPosition] = []
        self.trades: List[BTTrade] = []
        self.current_market: str = ""
        self.market_end_ts: float = 0.0
        self._market_start_t: float = 0.0
        self._market_snapshot_count: int = 0
        self._last_record: Optional[dict] = None
        self._last_time: float = 0.0
        self._duration_minutes: int = 5

    # ── Core loop ──────────────────────────────────────────────

    def run(self, data_path: str) -> dict:
        """Run backtest on recorded JSONL data."""
        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                event = record.get("event")

                if event == "session_start":
                    self._duration_minutes = record.get("duration", 5)
                    continue
                elif event == "session_end":
                    continue
                elif event == "market_change":
                    self._handle_market_change(record)
                    continue

                # Regular snapshot
                self._process_record(record)

        # Close remaining positions at end of data
        for pos in list(self.positions):
            self._force_close(pos, self._last_time)

        return self._compile_results()

    def _process_record(self, record: dict) -> None:
        t = record["t"]
        slug = record.get("slug", "")
        self._last_time = t
        self._last_record = record

        # Track market
        if slug != self.current_market:
            self.current_market = slug
            self._market_start_t = t
            self._market_snapshot_count = 0

        # Parse end time
        end_str = record.get("end", "")
        if end_str:
            try:
                end_str = end_str.replace("Z", "+00:00")
                self.market_end_ts = datetime.fromisoformat(end_str).timestamp()
            except Exception:
                pass

        self._market_snapshot_count += 1

        time_remaining = max(0, self.market_end_ts - t) if self.market_end_ts > 0 else 999
        elapsed = t - self._market_start_t

        state = MarketState(
            slug=slug,
            end_ts=self.market_end_ts,
            time_remaining=time_remaining,
            elapsed=elapsed,
            snapshot_count=self._market_snapshot_count,
        )

        self.on_snapshot(record, state)

    def _handle_market_change(self, record: dict) -> None:
        old = record.get("old", "")
        new = record.get("new", "")
        self.on_market_change(old, new, record["t"])
        self.current_market = new
        self._market_start_t = record["t"]
        self._market_snapshot_count = 0
        self.market_end_ts = 0.0

    # ── Abstract method ────────────────────────────────────────

    @abstractmethod
    def on_snapshot(self, record: dict, state: MarketState) -> None:
        """Process a single price snapshot. Implement in subclass."""

    def on_market_change(self, old: str, new: str, t: float) -> None:
        """Handle market change. Default: force close all positions."""
        for pos in list(self.positions):
            self._force_close(pos, t)

    # ── Trading helpers ────────────────────────────────────────

    def buy(self, side: str, record: dict, tag: str = "",
            tp: Optional[float] = None, sl: Optional[float] = None) -> Optional[BTPosition]:
        """Open a position at ask price with fee."""
        data = record.get(side)
        if not data or not data.get("ask") or data["ask"] >= 1.0:
            return None

        ask = data["ask"]
        if ask <= 0:
            return None

        usdc_amount = self.balance * self.bet_fraction
        if usdc_amount < 0.01:
            return None

        usdc_after_fee = usdc_amount * (1 - self.fee)
        size = usdc_after_fee / ask

        pos = BTPosition(
            side=side,
            entry_price=ask,
            size=size,
            cost=usdc_amount,
            entry_time=record["t"],
            tp_price=ask + tp if tp else None,
            sl_price=ask - sl if sl else None,
            tag=tag,
        )
        self.balance -= usdc_amount
        self.positions.append(pos)
        return pos

    def sell(self, pos: BTPosition, record: dict, result: str = "S") -> BTTrade:
        """Close a position at bid price with fee."""
        data = record.get(pos.side)
        bid = data["bid"] if data and data.get("bid") else pos.entry_price
        return self._close(pos, bid, record["t"], result)

    def settle(self, pos: BTPosition, price: float, t: float) -> BTTrade:
        """Close a position at settlement price."""
        return self._close(pos, price, t, "E")  # E = expiry/settlement

    def check_tp_sl(self, record: dict) -> None:
        """Check TP/SL on all open positions."""
        for pos in list(self.positions):
            data = record.get(pos.side)
            if not data or not data.get("bid"):
                continue
            bid = data["bid"]
            if pos.tp_price and bid >= pos.tp_price:
                self._close(pos, bid, record["t"], "W")
            elif pos.sl_price and bid <= pos.sl_price:
                self._close(pos, bid, record["t"], "L")

    def _close(self, pos: BTPosition, exit_price: float, t: float, result: str) -> BTTrade:
        """Internal close logic."""
        gross = pos.size * exit_price
        fee = gross * self.fee
        net_proceeds = gross - fee
        pnl = net_proceeds - pos.cost

        self.balance += pos.cost + pnl  # return cost + pnl
        hold = t - pos.entry_time

        trade = BTTrade(
            n=len(self.trades) + 1,
            side=pos.side.upper(),
            entry=pos.entry_price,
            exit=exit_price,
            pnl=pnl,
            hold=hold,
            result=result,
            balance=self.balance,
            market=self.current_market,
            tag=pos.tag,
        )
        self.trades.append(trade)
        if pos in self.positions:
            self.positions.remove(pos)
        return trade

    def _force_close(self, pos: BTPosition, t: float) -> None:
        """Force close at entry price (market expired / data ended)."""
        self._close(pos, pos.entry_price, t, "X")

    # ── Results ────────────────────────────────────────────────

    def _compile_results(self) -> dict:
        wins = sum(1 for t in self.trades if t.result == "W")
        losses = sum(1 for t in self.trades if t.result == "L")
        settled = sum(1 for t in self.trades if t.result == "E")
        expired = sum(1 for t in self.trades if t.result == "X")
        sells = sum(1 for t in self.trades if t.result == "S")
        total_pnl = self.balance - self.start_balance
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        total_pnl_trades = sum(1 for t in self.trades if t.result in ("W", "L", "S"))
        profitable = sum(1 for t in self.trades if t.pnl > 0)
        profit_rate = profitable / len(self.trades) * 100 if self.trades else 0

        return {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "settled": settled,
            "expired": expired,
            "sells": sells,
            "total_pnl": total_pnl,
            "return_pct": total_pnl / self.start_balance * 100,
            "win_rate": win_rate,
            "profit_rate": profit_rate,
            "final_balance": self.balance,
            "start_balance": self.start_balance,
            "trade_log": self.trades,
        }

    def print_results(self, results: dict, params: str = "", verbose: bool = True) -> None:
        """Print trade log and summary."""
        if verbose and results["trade_log"]:
            print(f"\n  {'#':>3}  {'Side':4}  {'Entry':>8}  {'Exit':>8}  "
                  f"{'PnL':>9}  {'Hold':>6}  Res  {'Balance':>10}  {'Tag'}")
            print(f"  {'---':>3}  {'----':4}  {'--------':>8}  {'--------':>8}  "
                  f"{'---------':>9}  {'------':>6}  ---  {'----------':>10}  {'---'}")
            for t in results["trade_log"]:
                hold = f"{t.hold:.0f}s" if t.hold < 60 else f"{t.hold / 60:.1f}m"
                print(f"  {t.n:>3}  {t.side:4}  {t.entry:>8.4f}  {t.exit:>8.4f}  "
                      f"${t.pnl:>+8.4f}  {hold:>6}  {'  ' + t.result}  "
                      f"${t.balance:>9.4f}  {t.tag}")

        print(f"\n  {self.name} Results{' (' + params + ')' if params else ''}:")
        print(f"    Trades: {results['trades']} "
              f"(W:{results['wins']} L:{results['losses']} "
              f"E:{results['settled']} X:{results['expired']} S:{results['sells']})")
        print(f"    Win rate: {results['win_rate']:.1f}% | "
              f"Profitable: {results['profit_rate']:.1f}%")
        print(f"    PnL: ${results['total_pnl']:+.4f} ({results['return_pct']:+.1f}%)")
        print(f"    Balance: ${results['start_balance']:.2f} -> ${results['final_balance']:.4f}")


# ── CLI helpers ────────────────────────────────────────────────

def base_argparser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("data", help="Path to recorded JSONL data file")
    parser.add_argument("--balance", type=float, default=10.0)
    parser.add_argument("--bet", type=float, default=0.10, help="Bet fraction (default: 0.10)")
    parser.add_argument("--fee", type=float, default=0.02, help="Fee per side (default: 0.02)")
    parser.add_argument("--sweep", action="store_true", help="Sweep parameter combinations")
    parser.add_argument("--quiet", action="store_true", help="Hide trade log")
    return parser


def print_data_summary(data_path: str) -> None:
    """Print summary of the data file."""
    with open(data_path) as f:
        lines = f.readlines()
    snapshots = sum(1 for l in lines if '"event"' not in l)
    changes = sum(1 for l in lines if '"market_change"' in l)
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    hours = (last["t"] - first["t"]) / 3600
    print(f"  Data: {data_path}")
    print(f"  Snapshots: {snapshots:,} | Markets: {changes} | Duration: {hours:.1f} hours")


def run_sweep(strategy_cls, sweep_params: List[Dict[str, Any]],
              data_path: str, base_kwargs: dict, verbose: bool = False) -> None:
    """Run parameter sweep and print comparison table.

    sweep_params: list of dicts, each dict is one parameter combination
                  to pass as kwargs to strategy_cls constructor.
    """
    all_results = []
    for params in sweep_params:
        kwargs = {**base_kwargs, **params}
        bt = strategy_cls(**kwargs)
        r = bt.run(data_path)
        all_results.append((params, r))

    # Sort by return (best first)
    all_results.sort(key=lambda x: x[1]["return_pct"], reverse=True)

    # Build header from param names
    param_names = list(sweep_params[0].keys()) if sweep_params else []
    header_parts = [f"{p:>8}" for p in param_names]
    header = "  ".join(header_parts)
    separator = "  ".join(["-" * 8] * len(param_names))

    print(f"\n  {header}  {'Trades':>6}  {'W':>3}  {'L':>3}  {'E':>3}  "
          f"{'ProfR':>6}  {'PnL':>10}  {'Return':>8}")
    print(f"  {separator}  {'------':>6}  {'---':>3}  {'---':>3}  {'---':>3}  "
          f"{'------':>6}  {'----------':>10}  {'--------':>8}")

    for params, r in all_results:
        vals = "  ".join([f"{params[p]:>8.3f}" if isinstance(params[p], float)
                         else f"{params[p]:>8}" for p in param_names])
        print(f"  {vals}  {r['trades']:>6}  {r['wins']:>3}  {r['losses']:>3}  "
              f"{r['settled']:>3}  {r['profit_rate']:>5.1f}%  "
              f"${r['total_pnl']:>+9.4f}  {r['return_pct']:>+7.1f}%")

    print(f"\n  {len(all_results)} parameter combinations (sorted best to worst)")
