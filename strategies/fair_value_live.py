"""
Fair Value Live Strategy - Binary Option Pricing for Live Trading

Uses Black-Scholes binary option pricing to compute theoretical fair value
from live BTC spot price (Binance WebSocket), strike (price at market open),
time remaining, and realized volatility. Trades when Polymarket price deviates
from fair value by more than fees + spread.

Strategy Logic:
1. Connect to Binance WebSocket for live BTC spot price
2. Track strike price (BTC price when each market opens)
3. Compute realized vol from recent Binance 1-second bars
4. Compute fair value: fv_up = N(d2) using Black-Scholes binary formula
5. Buy when edge (fv - ask - fees) exceeds threshold
6. Exit via TP/SL or market settlement

Usage:
    from strategies.fair_value_live import FairValueStrategy, FairValueConfig

    strategy = FairValueStrategy(bot, config)
    await strategy.run()
"""

import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from lib.console import Colors, format_countdown
from lib.binance_feed import BinancePriceFeed, realized_vol_from_prices
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


# ── Binary option math ──────────────────────────────────────────────

def binary_call_fv(S: float, K: float, T: float, sigma: float) -> float:
    """Fair value of a cash-or-nothing binary call (P(S > K at expiry))."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)
    d2 = (np.log(S / K) + (-sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d2))


def kelly_size(p_true: float, ask_price: float, fee_rate: float,
               fraction: float = 0.25) -> float:
    """Kelly criterion sizing for a binary bet."""
    cost = ask_price * (1 + fee_rate)
    if cost >= 1.0 or cost <= 0:
        return 0.0
    b = (1.0 - cost) / cost
    q = 1 - p_true
    f = (b * p_true - q) / b
    return max(0.0, f * fraction)


# ── Config ──────────────────────────────────────────────────────────

@dataclass
class FairValueConfig(StrategyConfig):
    """Fair value strategy configuration."""

    edge_threshold: float = 0.05   # Min edge to enter
    vol_window: int = 300          # Seconds for realized vol
    kelly_frac: float = 0.25       # Kelly fraction
    min_time_remaining: float = 30.0   # Don't trade with < N seconds left
    max_time_remaining: float = 0.0    # Don't trade in first N seconds (0 = no limit)


# ── Strategy ────────────────────────────────────────────────────────

class FairValueStrategy(BaseStrategy):
    """
    Fair Value Trading Strategy.

    Uses Black-Scholes binary option pricing with live Binance BTC price
    to compute theoretical fair value and trade mispricing.
    """

    def __init__(self, bot: TradingBot, config: FairValueConfig):
        super().__init__(bot, config)
        self.fv_config = config

        # Binance feed
        self._binance = BinancePriceFeed(symbol="btcusdt")

        # Strike prices per market slug
        self._strike_prices: Dict[str, float] = {}

        # Fair value state (updated each tick)
        self._fv_up: float = 0.5
        self._fv_down: float = 0.5
        self._current_edge_up: float = 0.0
        self._current_edge_down: float = 0.0
        self._current_sigma: float = 0.0
        self._current_btc: float = 0.0
        self._current_strike: float = 0.0

        # Track if we've traded this market
        self._traded_this_market: bool = False

    async def start(self) -> bool:
        """Start strategy and Binance feed."""
        # Start Binance WebSocket
        self.log("Connecting to Binance price feed...", "info")
        if not await self._binance.connect():
            self.log("Failed to connect to Binance WebSocket", "error")
            return False

        # Wait for first price
        if not await self._binance.wait_for_price(timeout=10.0):
            self.log("Timeout waiting for Binance price", "error")
            await self._binance.disconnect()
            return False

        btc_price = self._binance.get_price()
        self.log(f"Binance connected: BTC = ${btc_price:,.2f}", "success")

        # Start base strategy (Polymarket WS)
        if not await super().start():
            await self._binance.disconnect()
            return False

        # Set initial strike for current market
        self._set_strike_for_current_market()

        return True

    async def stop(self) -> None:
        """Stop strategy and Binance feed."""
        await self._binance.disconnect()
        await super().stop()

    def _set_strike_for_current_market(self) -> None:
        """Set strike price for current market from current BTC price."""
        market = self.current_market
        if not market:
            return

        slug = market.slug
        if slug not in self._strike_prices:
            # Use current BTC price as strike (approximation)
            # Ideally we'd use the price at market_open_ts from the slug
            btc_price = self._binance.get_price()
            if btc_price > 0:
                # Try to get market open timestamp from slug
                slug_ts = market.slug_timestamp()
                if slug_ts:
                    # Market has been running for some time already
                    # Use current price as approximation of strike
                    # (In a production system, we'd fetch historical price at slug_ts)
                    pass
                self._strike_prices[slug] = btc_price
                self.log(f"Strike set: ${btc_price:,.2f} for {slug}", "info")

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update."""
        pass  # Price recording done in base class

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Main strategy tick - compute fair value and check for edge."""
        market = self.current_market
        if not market:
            return

        # Get BTC spot price
        S = self._binance.get_price()
        if S <= 0:
            return
        self._current_btc = S

        # Get strike
        K = self._strike_prices.get(market.slug, 0)
        if K <= 0:
            return
        self._current_strike = K

        # Time remaining
        mins, secs = market.get_countdown()
        if mins < 0:
            return
        remaining_seconds = mins * 60 + secs

        # Time filters
        if remaining_seconds < self.fv_config.min_time_remaining:
            return
        if (self.fv_config.max_time_remaining > 0 and
                remaining_seconds > self.fv_config.max_time_remaining):
            return

        T = remaining_seconds / (365.25 * 24 * 3600)

        # Realized vol
        sigma = self._binance.get_realized_vol(self.fv_config.vol_window)
        self._current_sigma = sigma
        if sigma <= 0:
            return

        # Compute fair values
        self._fv_up = binary_call_fv(S, K, T, sigma)
        self._fv_down = 1.0 - self._fv_up

        # Compute edges
        up_ask = self.market.get_best_ask("up")
        down_ask = self.market.get_best_ask("down")

        from strategy.common import taker_fee_rate

        if up_ask > 0 and up_ask < 1.0:
            fee = taker_fee_rate(up_ask)
            self._current_edge_up = self._fv_up - up_ask - (up_ask * fee)
        else:
            self._current_edge_up = 0.0

        if down_ask > 0 and down_ask < 1.0:
            fee = taker_fee_rate(down_ask)
            self._current_edge_down = self._fv_down - down_ask - (down_ask * fee)
        else:
            self._current_edge_down = 0.0

        # Check if we can trade
        if not self.positions.can_open_position:
            return
        if self._traded_this_market:
            return

        # Try to enter on the side with best edge
        sides = []
        if self._current_edge_up >= self.fv_config.edge_threshold:
            sides.append(("up", self._fv_up, self._current_edge_up, up_ask))
        if self._current_edge_down >= self.fv_config.edge_threshold:
            sides.append(("down", self._fv_down, self._current_edge_down, down_ask))

        if not sides:
            return

        # Pick the side with the highest edge
        sides.sort(key=lambda x: x[2], reverse=True)
        side, fv, edge, ask = sides[0]

        fee = taker_fee_rate(ask)
        k = kelly_size(fv, ask, fee, self.fv_config.kelly_frac)
        if k <= 0:
            return

        self.log(
            f"FAIR VALUE SIGNAL: {side.upper()} fv={fv:.3f} ask={ask:.4f} "
            f"edge={edge:.3f} sigma={sigma:.1%} kelly={k:.3f}",
            "trade",
        )

        current_price = prices.get(side, 0)
        if current_price > 0:
            success = await self.execute_buy(side, current_price)
            if success:
                self._traded_this_market = True

    def _close_positions_on_market_change(self, old_slug: str = "") -> None:
        """Settle positions at binary outcome (0 or 1) based on BTC vs strike."""
        positions = self.positions.get_all_positions()
        if not positions:
            return

        btc_price = self._binance.get_price()
        strike = self._strike_prices.get(old_slug, 0)

        if btc_price > 0 and strike > 0:
            winner = "up" if btc_price > strike else "down"
        else:
            winner = None

        for position in positions:
            if winner:
                settle_price = 1.00 if position.side == winner else 0.00
            else:
                # Can't determine winner, fall back to last price
                settle_price = self.prices.get_current_price(position.side)
                if settle_price <= 0:
                    settle_price = position.entry_price

            pnl = position.get_pnl(settle_price)
            if self.config.paper:
                self._paper_balance += pnl
            elif self._live_balance > 0:
                self._live_balance += position.size * settle_price

            result = "WON" if settle_price == 1.0 else "LOST"
            self.log(
                f"SETTLED: {position.side.upper()} @ {settle_price:.2f} ({result}) "
                f"PnL: ${pnl:+.2f}",
                "success" if pnl >= 0 else "warning"
            )
            self.positions.close_position(position.id, realized_pnl=pnl)
            self._record_trade(position, settle_price, pnl)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - update strike price."""
        self._traded_this_market = False
        self._set_strike_for_current_market()

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display with fair value overlay."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        binance_status = f"{Colors.GREEN}BN{Colors.RESET}" if self._binance.is_connected else f"{Colors.RED}BN{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        header = (
            f"{Colors.CYAN}[{self.config.coin}]{Colors.RESET} [{ws_status}|{binance_status}] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} "
            f"(W:{stats['winning_trades']} L:{stats['losing_trades']}) | "
            f"PnL: ${stats['total_pnl']:+.2f}"
        )
        if self.config.paper:
            header += f" | {Colors.YELLOW}[PAPER] ${self._paper_balance:.2f}{Colors.RESET}"
        lines.append(header)
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # BTC + Fair Value panel
        btc_str = f"${self._current_btc:,.2f}" if self._current_btc > 0 else "--"
        strike_str = f"${self._current_strike:,.2f}" if self._current_strike > 0 else "--"
        sigma_str = f"{self._current_sigma:.1%}" if self._current_sigma > 0 else "--"

        lines.append(
            f"  BTC: {Colors.BOLD}{btc_str}{Colors.RESET}  "
            f"Strike: {strike_str}  "
            f"Sigma: {sigma_str}  "
            f"Vol window: {self.fv_config.vol_window}s"
        )

        # Fair value and edge
        fv_up_str = f"{self._fv_up:.4f}"
        fv_down_str = f"{self._fv_down:.4f}"

        up_edge_color = Colors.GREEN if self._current_edge_up >= self.fv_config.edge_threshold else Colors.DIM
        down_edge_color = Colors.GREEN if self._current_edge_down >= self.fv_config.edge_threshold else Colors.DIM

        lines.append(
            f"  FV UP: {Colors.GREEN}{fv_up_str}{Colors.RESET}  "
            f"Edge: {up_edge_color}{self._current_edge_up:+.4f}{Colors.RESET}  |  "
            f"FV DOWN: {Colors.RED}{fv_down_str}{Colors.RESET}  "
            f"Edge: {down_edge_color}{self._current_edge_down:+.4f}{Colors.RESET}  "
            f"(threshold: {self.fv_config.edge_threshold:.2f})"
        )
        lines.append("-" * 80)

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|{Colors.RED}{'DOWN':^39}{Colors.RESET}")
        lines.append(f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}")
        lines.append("-" * 80)

        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            up_bid = f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}" if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            up_ask = f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}" if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            down_bid = f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}" if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            down_ask = f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}" if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            lines.append(f"{up_bid} | {up_ask}|{down_bid} | {down_ask}")

        lines.append("-" * 80)

        # Summary
        up_mid = up_ob.mid_price if up_ob else prices.get("up", 0)
        down_mid = down_ob.mid_price if down_ob else prices.get("down", 0)
        up_spread = self.market.get_spread("up")
        down_spread = self.market.get_spread("down")

        lines.append(
            f"Mid: {Colors.GREEN}{up_mid:.4f}{Colors.RESET}  "
            f"Spread: {up_spread:.4f}  "
            f"FV: {Colors.GREEN}{self._fv_up:.4f}{Colors.RESET}           |"
            f"Mid: {Colors.RED}{down_mid:.4f}{Colors.RESET}  "
            f"Spread: {down_spread:.4f}  "
            f"FV: {Colors.RED}{self._fv_down:.4f}{Colors.RESET}"
        )

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()
        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"Size: ${pos.size:.2f} | PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open positions){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 80)
            lines.append(f"{Colors.BOLD}Recent Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        # Render
        output = "\033[H\033[J" + "\n".join(lines)
        print(output, flush=True)

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"
        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)
