"""
Strategy Base Class - Foundation for Trading Strategies

Provides:
- Base class for all trading strategies
- Common lifecycle methods (start, stop, run)
- Integration with lib components (MarketManager, PriceTracker, PositionManager)
- Logging and status display utilities

Usage:
    from strategies.base import BaseStrategy, StrategyConfig

    class MyStrategy(BaseStrategy):
        async def on_book_update(self, snapshot):
            # Handle orderbook updates
            pass

        async def on_tick(self, prices):
            # Called each strategy tick
            pass
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List

from lib.console import LogBuffer, log
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker
from lib.position_manager import PositionManager, Position
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class StrategyConfig:
    """Base strategy configuration."""

    coin: str = "ETH"
    size: float = 0.0  # Fixed USDC size per trade (0 = use bet_fraction instead)
    max_positions: int = 1
    take_profit: float = 0.10
    stop_loss: float = 0.05

    # Paper trading
    paper: bool = False  # Enable paper trading (no real orders)
    paper_balance: float = 10.0  # Starting paper balance in USDC
    bet_fraction: float = 0.10  # Fraction of balance to bet each trade (used in paper + live)
    paper_fee: float = 0.02  # Simulated fee per trade (2%)

    # Live balance tracking
    live_balance: float = 0.0  # Starting USDC balance for live % sizing

    # Market settings
    market_duration: int = 15  # Market duration in minutes (5 or 15)
    no_trade_seconds: int = 0  # Don't open new positions in last N seconds (0 = auto)
    market_check_interval: float = 30.0
    auto_switch_market: bool = True

    # Price tracking
    price_lookback_seconds: int = 10
    price_history_size: int = 100

    # Display settings
    update_interval: float = 0.1
    order_refresh_interval: float = 30.0  # Seconds between order refreshes


class BaseStrategy(ABC):
    """
    Base class for trading strategies.

    Provides common infrastructure:
    - MarketManager for WebSocket and market discovery
    - PriceTracker for price history
    - PositionManager for positions and TP/SL
    - Logging and status display
    """

    def __init__(self, bot: TradingBot, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            bot: TradingBot instance for order execution
            config: Strategy configuration
        """
        self.bot = bot
        self.config = config

        # Core components
        self.market = MarketManager(
            coin=config.coin,
            market_check_interval=config.market_check_interval,
            auto_switch_market=config.auto_switch_market,
            duration_minutes=config.market_duration,
        )

        self.prices = PriceTracker(
            lookback_seconds=config.price_lookback_seconds,
            max_history=config.price_history_size,
        )

        self.positions = PositionManager(
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            max_positions=config.max_positions,
        )

        # State
        self.running = False
        self._status_mode = False

        # Paper trading
        self._paper_balance = config.paper_balance if config.paper else 0.0

        # Live balance tracking (for % sizing)
        self._live_balance = config.live_balance
        self._paper_order_counter = 0

        # Trade log (all completed trades)
        self._trade_log: List[dict] = []

        # Logging
        self._log_buffer = LogBuffer(max_size=5)

        # Open orders cache (refreshed in background)
        self._cached_orders: List[dict] = []
        self._last_order_refresh: float = 0
        self._order_refresh_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.market.is_connected

    @property
    def current_market(self) -> Optional[MarketInfo]:
        """Get current market info."""
        return self.market.current_market

    @property
    def token_ids(self) -> Dict[str, str]:
        """Get current token IDs."""
        return self.market.token_ids

    @property
    def open_orders(self) -> List[dict]:
        """Get cached open orders."""
        return self._cached_orders

    def _refresh_orders_sync(self) -> List[dict]:
        """Refresh open orders synchronously (called via to_thread)."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.bot.get_open_orders())
            finally:
                loop.close()
        except Exception:
            return []

    async def _do_order_refresh(self) -> None:
        """Background task to refresh orders without blocking."""
        try:
            orders = await asyncio.to_thread(self._refresh_orders_sync)
            self._cached_orders = orders
        except Exception:
            pass
        finally:
            self._order_refresh_task = None

    def _maybe_refresh_orders(self) -> None:
        """Schedule order refresh if interval has passed (fire-and-forget)."""
        now = time.time()
        if now - self._last_order_refresh > self.config.order_refresh_interval:
            # Don't start new refresh if one is already running
            if self._order_refresh_task is not None and not self._order_refresh_task.done():
                return
            self._last_order_refresh = now
            # Fire and forget - doesn't block main loop
            self._order_refresh_task = asyncio.create_task(self._do_order_refresh())

    def log(self, msg: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            msg: Message to log
            level: Log level (info, success, warning, error, trade)
        """
        if self._status_mode:
            self._log_buffer.add(msg, level)
        else:
            log(msg, level)

    async def start(self) -> bool:
        """
        Start the strategy.

        Returns:
            True if started successfully
        """
        self.running = True

        # Fetch live balance from Polymarket if not paper and no fixed size
        if not self.config.paper and self.config.size <= 0 and self._live_balance <= 0:
            self.log("Fetching USDC balance from Polymarket...", "info")
            balance = await self.bot.get_balance()
            if balance > 0:
                self._live_balance = balance
                self.log(f"Balance: ${balance:.2f} USDC", "success")
            else:
                self.log("Could not fetch balance. Set --size or --balance.", "error")
                return False

        # Register callbacks on market manager
        @self.market.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):  # pyright: ignore[reportUnusedFunction]
            # Record price
            for side, token_id in self.token_ids.items():
                if token_id == snapshot.asset_id:
                    self.prices.record(side, snapshot.mid_price)
                    break

            # Delegate to subclass
            await self.on_book_update(snapshot)

        @self.market.on_market_change
        def handle_market_change(old_slug: str, new_slug: str):  # pyright: ignore[reportUnusedFunction]
            self.log(f"Market changed: {old_slug} -> {new_slug}", "warning")
            # Close any open positions — old tokens are invalid in the new market
            self._close_positions_on_market_change(old_slug)
            self.prices.clear()
            self.on_market_change(old_slug, new_slug)

        @self.market.on_connect
        def handle_connect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket connected", "success")
            self.on_connect()

        @self.market.on_disconnect
        def handle_disconnect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket disconnected", "warning")
            self.on_disconnect()

        # Start market manager
        if not await self.market.start():
            self.running = False
            return False

        # Wait for initial data
        if not await self.market.wait_for_data(timeout=5.0):
            self.log("Timeout waiting for market data", "warning")

        return True

    async def stop(self) -> None:
        """Stop the strategy."""
        self.running = False

        # Cancel order refresh task if running
        if self._order_refresh_task is not None:
            self._order_refresh_task.cancel()
            try:
                await self._order_refresh_task
            except asyncio.CancelledError:
                pass
            self._order_refresh_task = None

        await self.market.stop()

    async def run(self) -> None:
        """Main strategy loop."""
        try:
            if not await self.start():
                self.log("Failed to start strategy", "error")
                return

            self._status_mode = True

            while self.running:
                # Get current prices
                prices = self._get_current_prices()

                # Call tick handler
                await self.on_tick(prices)

                # Check position exits
                await self._check_exits(prices)

                # Refresh orders in background (fire-and-forget)
                self._maybe_refresh_orders()

                # Update display
                self.render_status(prices)

                await asyncio.sleep(self.config.update_interval)

        except KeyboardInterrupt:
            self.log("Strategy stopped by user")
        finally:
            await self.stop()
            self._print_summary()

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices from market manager.

        In paper mode, returns bid prices (what you'd actually sell at)
        so TP/SL checks are realistic. In live mode, returns mid prices.
        """
        prices = {}
        for side in ["up", "down"]:
            if self.config.paper:
                # Use bid — that's the real exit price
                price = self.market.get_best_bid(side)
            else:
                price = self.market.get_mid_price(side)
            if price > 0:
                prices[side] = price
        return prices

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Check and execute exits for all positions."""
        exits = self.positions.check_all_exits(prices)

        for position, exit_type, pnl in exits:
            if exit_type == "take_profit":
                self.log(
                    f"TAKE PROFIT: {position.side.upper()} PnL: +${pnl:.2f}",
                    "success"
                )
            elif exit_type == "stop_loss":
                self.log(
                    f"STOP LOSS: {position.side.upper()} PnL: ${pnl:.2f}",
                    "warning"
                )

            # Execute sell
            await self.execute_sell(position, prices.get(position.side, 0))

    async def execute_buy(self, side: str, current_price: float) -> bool:
        """
        Execute market buy order.

        Args:
            side: "up" or "down"
            current_price: Current market price

        Returns:
            True if order placed successfully
        """
        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        # No-trade zone: don't open positions near market expiry
        if self._in_no_trade_zone():
            self.log("Skipped: market ending soon (no-trade zone)", "warning")
            return False

        # Calculate size and fill price
        if self.config.paper:
            # Use the ask price — that's what you'd actually pay
            fill_price = self.market.get_best_ask(side)
            if fill_price <= 0 or fill_price >= 1:
                fill_price = current_price  # fallback
            usdc_amount = self._paper_balance * self.config.bet_fraction
            # Deduct fee from the amount we can spend
            usdc_after_fee = usdc_amount * (1 - self.config.paper_fee)
            size = usdc_after_fee / fill_price
        else:
            fill_price = current_price
            if self.config.size > 0:
                usdc_amount = self.config.size
            else:
                usdc_amount = self._live_balance * self.config.bet_fraction
            size = usdc_amount / current_price

        buy_price = min(current_price + 0.02, 0.99)

        if self.config.paper:
            # Paper trade — simulate fill at ask
            fee_paid = usdc_amount * self.config.paper_fee
            self._paper_order_counter += 1
            order_id = f"paper-{self._paper_order_counter}"
            self.log(
                f"[PAPER] BUY {side.upper()} @ {fill_price:.4f} (ask) "
                f"size={size:.2f} (${usdc_amount:.2f} - ${fee_paid:.2f} fee) | "
                f"Balance: ${self._paper_balance:.2f}",
                "trade"
            )
            self.positions.open_position(
                side=side,
                token_id=token_id,
                entry_price=fill_price,
                size=size,
                order_id=order_id,
            )
            self._log_trade_stats()
            return True
        else:
            self.log(f"BUY {side.upper()} @ {current_price:.4f} size={size:.2f} (${usdc_amount:.2f})", "trade")

            result = await self.bot.place_order(
                token_id=token_id,
                price=buy_price,
                size=size,
                side="BUY"
            )

            if result.success:
                self.log(f"Order placed: {result.order_id}", "success")
                self.positions.open_position(
                    side=side,
                    token_id=token_id,
                    entry_price=current_price,
                    size=size,
                    order_id=result.order_id,
                )
                if self._live_balance > 0:
                    self._live_balance -= usdc_amount
                return True
            else:
                self.log(f"Order failed: {result.message}", "error")
                return False

    async def execute_sell(self, position: Position, current_price: float) -> bool:
        """
        Execute sell order to close position.

        Args:
            position: Position to close
            current_price: Current price

        Returns:
            True if order placed
        """
        if self.config.paper:
            # Use bid price — that's what you'd actually get
            fill_price = self.market.get_best_bid(position.side)
            if fill_price <= 0:
                fill_price = current_price  # fallback
        else:
            fill_price = current_price

        sell_price = max(current_price - 0.02, 0.01)
        pnl = position.get_pnl(fill_price)

        if self.config.paper:
            # Deduct sell fee
            gross_proceeds = position.size * fill_price
            fee = gross_proceeds * self.config.paper_fee
            pnl -= fee

            self._paper_balance += pnl
            self.log(
                f"[PAPER] SELL {position.side.upper()} @ {fill_price:.4f} (bid) "
                f"fee: ${fee:.2f} | PnL: ${pnl:+.2f} | Balance: ${self._paper_balance:.2f}",
                "success" if pnl >= 0 else "warning"
            )
            self.positions.close_position(position.id, realized_pnl=pnl)
            self._record_trade(position, fill_price, pnl)
            self._log_trade_stats()
            return True
        else:
            result = await self.bot.place_order(
                token_id=position.token_id,
                price=sell_price,
                size=position.size,
                side="SELL"
            )

            if result.success:
                self.log(f"Sell order: {result.order_id} PnL: ${pnl:+.2f}", "success")
                self.positions.close_position(position.id, realized_pnl=pnl)
                self._record_trade(position, current_price, pnl)
                if self._live_balance > 0:
                    self._live_balance += position.size * current_price
                return True
            else:
                self.log(f"Sell failed: {result.message}", "error")
                return False

    def _in_no_trade_zone(self) -> bool:
        """Check if we're too close to market expiry to open new positions."""
        market = self.market.current_market
        if not market:
            return False

        mins, secs = market.get_countdown()
        if mins < 0:
            return False

        remaining = mins * 60 + secs
        cutoff = self.config.no_trade_seconds
        if cutoff == 0:
            # Auto: 30s for 5m markets, 120s for 15m markets
            if self.config.market_duration <= 5:
                cutoff = 30
            else:
                cutoff = 120

        return remaining <= cutoff

    def _close_positions_on_market_change(self, old_slug: str = "") -> None:
        """Close all open positions when market changes.

        Uses the last known price for each position's side.
        Positions can't carry over — the token IDs change with each market.
        Subclasses can override for binary settlement (0 or 1).
        """
        positions = self.positions.get_all_positions()
        if not positions:
            return

        for position in positions:
            # Use last recorded price for this side
            last_price = self.prices.get_current_price(position.side)
            if last_price <= 0:
                last_price = position.entry_price  # fallback: flat close

            pnl = position.get_pnl(last_price)
            if self.config.paper:
                self._paper_balance += pnl
            elif self._live_balance > 0:
                self._live_balance += position.size * last_price

            self.log(
                f"MARKET EXPIRED: closed {position.side.upper()} @ {last_price:.4f} "
                f"PnL: ${pnl:+.2f}",
                "warning"
            )
            self.positions.close_position(position.id, realized_pnl=pnl)
            self._record_trade(position, last_price, pnl)

    def _record_trade(self, position: Position, exit_price: float, pnl: float) -> None:
        """Record a completed trade in the trade log."""
        hold_time = position.get_hold_time()
        self._trade_log.append({
            "n": len(self._trade_log) + 1,
            "side": position.side.upper(),
            "entry": position.entry_price,
            "exit": exit_price,
            "size": position.size,
            "pnl": pnl,
            "result": "W" if pnl >= 0 else "L",
            "hold": hold_time,
            "balance": self._paper_balance if self.config.paper else None,
        })

    def _log_trade_stats(self) -> None:
        """Log current trade statistics."""
        stats = self.positions.get_stats()
        msg = (
            f"Trades: {stats['trades_closed']} | "
            f"W: {stats['winning_trades']} L: {stats['losing_trades']} | "
            f"PnL: ${stats['total_pnl']:+.2f}"
        )
        if self.config.paper:
            msg += f" | Balance: ${self._paper_balance:.2f}"
        self.log(msg, "info")

    def _print_summary(self) -> None:
        """Print session summary with full trade log."""
        self._status_mode = False
        print()
        stats = self.positions.get_stats()

        # Trade log table
        if self._trade_log:
            self.log("Trade Log:")
            bal_hdr = "    Balance" if self.config.paper else ""
            bal_sep = "  ----------" if self.config.paper else ""
            self.log(f"  {'#':>3}  {'Side':4}  {'Entry':>8}  {'Exit':>8}  {'PnL':>9}  {'Hold':>6}  Result{bal_hdr}")
            self.log(f"  {'---':>3}  {'----':4}  {'--------':>8}  {'--------':>8}  {'---------':>9}  {'------':>6}  ------{bal_sep}")
            for t in self._trade_log:
                hold_str = f"{t['hold']:.0f}s" if t['hold'] < 60 else f"{t['hold'] / 60:.1f}m"
                line = (
                    f"  {t['n']:>3}  {t['side']:4}  "
                    f"{t['entry']:>8.4f}  {t['exit']:>8.4f}  "
                    f"${t['pnl']:>+8.2f}  {hold_str:>6}  "
                    f"{'  ' + t['result']:>6}"
                )
                if self.config.paper and t['balance'] is not None:
                    line += f"  ${t['balance']:>9.2f}"
                self.log(line)
            self.log("")

        # Summary
        self.log("Session Summary:")
        if self.config.paper:
            self.log(f"  Starting Balance: ${self.config.paper_balance:.2f}")
            self.log(f"  Final Balance: ${self._paper_balance:.2f}")
            net = self._paper_balance - self.config.paper_balance
            self.log(f"  Net Return: ${net:+.2f} ({net / self.config.paper_balance * 100:+.1f}%)")
        self.log(f"  Trades: {stats['trades_closed']}")
        self.log(f"  Wins: {stats['winning_trades']} | Losses: {stats['losing_trades']}")
        self.log(f"  Total PnL: ${stats['total_pnl']:+.2f}")
        self.log(f"  Win rate: {stats['win_rate']:.1f}%")

    # Abstract methods to implement in subclasses

    @abstractmethod
    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """
        Handle orderbook update.

        Called when new orderbook data is received.

        Args:
            snapshot: OrderbookSnapshot from WebSocket
        """
        pass

    @abstractmethod
    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Handle strategy tick.

        Called on each iteration of the main loop.

        Args:
            prices: Current prices {side: price}
        """
        pass

    @abstractmethod
    def render_status(self, prices: Dict[str, float]) -> None:
        """
        Render status display.

        Called on each tick to update the display.

        Args:
            prices: Current prices
        """
        pass

    # Optional hooks (override as needed)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Called when market changes."""
        pass

    def on_connect(self) -> None:
        """Called when WebSocket connects."""
        pass

    def on_disconnect(self) -> None:
        """Called when WebSocket disconnects."""
        pass
