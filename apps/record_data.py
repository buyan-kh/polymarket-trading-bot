#!/usr/bin/env python3
"""
Market Data Recorder - Record orderbook data for backtesting.

Connects to Polymarket WebSocket, records bid/ask/mid prices for
every book update, and saves to a JSONL file. Auto-switches markets.

Usage:
    python apps/record_data.py --coin BTC --duration 5
    python apps/record_data.py --coin BTC --duration 15 --output data/btc-15m.jsonl

Data format (one JSON object per line):
    {"t": 1771063800.5, "slug": "btc-updown-5m-...", "end": "2026-...",
     "up": {"mid": 0.55, "bid": 0.54, "ask": 0.56, "spread": 0.02},
     "down": {"mid": 0.45, "bid": 0.44, "ask": 0.46, "spread": 0.02}}

    {"t": 1771064100.0, "event": "market_change",
     "old": "btc-updown-5m-...", "new": "btc-updown-5m-..."}
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Suppress noisy logs
logging.getLogger("src.websocket_client").setLevel(logging.WARNING)
logging.getLogger("src.bot").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.market_manager import MarketManager
from src.websocket_client import OrderbookSnapshot


class DataRecorder:
    """Records orderbook data to a JSONL file."""

    def __init__(self, coin: str, duration: int, output_path: str, sample_interval: float = 0.5):
        self.coin = coin
        self.duration = duration
        self.output_path = output_path
        self.sample_interval = sample_interval

        self.manager = MarketManager(
            coin=coin,
            duration_minutes=duration,
        )

        self._file = None
        self._record_count = 0
        self._start_time = time.time()
        self._last_sample = 0.0

    def _write_record(self, record: dict) -> None:
        """Write a single JSON record to file."""
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._record_count += 1

    async def start(self) -> bool:
        """Start recording."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # Open file for appending
        self._file = open(self.output_path, "a")

        # Write session start marker
        self._write_record({
            "t": time.time(),
            "event": "session_start",
            "coin": self.coin,
            "duration": self.duration,
            "sample_interval": self.sample_interval,
        })

        # Register callbacks
        @self.manager.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):
            now = time.time()
            if now - self._last_sample < self.sample_interval:
                return
            self._last_sample = now
            self._write_snapshot(now)

        @self.manager.on_market_change
        def handle_change(old_slug: str, new_slug: str):
            self._write_record({
                "t": time.time(),
                "event": "market_change",
                "old": old_slug,
                "new": new_slug,
            })

        # Start market manager
        if not await self.manager.start():
            print("Failed to start market manager")
            return False

        if not await self.manager.wait_for_data(timeout=10.0):
            print("Timeout waiting for market data")

        return True

    def _write_snapshot(self, now: float) -> None:
        """Write current orderbook state."""
        market = self.manager.current_market
        if not market:
            return

        record = {
            "t": now,
            "slug": market.slug,
            "end": market.end_date,
        }

        for side in ["up", "down"]:
            ob = self.manager.get_orderbook(side)
            if ob:
                record[side] = {
                    "mid": round(ob.mid_price, 4),
                    "bid": round(ob.best_bid, 4),
                    "ask": round(ob.best_ask, 4),
                    "spread": round(ob.best_ask - ob.best_bid, 4),
                }
            else:
                record[side] = None

        self._write_record(record)

    async def run(self) -> None:
        """Main recording loop."""
        if not await self.start():
            return

        try:
            while True:
                await asyncio.sleep(1)
                self._print_status()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    def _print_status(self) -> None:
        """Print recording status."""
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        market = self.manager.current_market
        slug = market.slug if market else "---"
        countdown = market.get_countdown_str() if market else "--:--"

        size_kb = os.path.getsize(self.output_path) / 1024

        status = (
            f"\r  Recording: {self.coin} {self.duration}m | "
            f"Market: {slug} (ends {countdown}) | "
            f"Records: {self._record_count:,} | "
            f"Size: {size_kb:.0f}KB | "
            f"Time: {hours:02d}:{mins:02d}:{secs:02d}"
        )
        print(status, end="", flush=True)

    async def stop(self) -> None:
        """Stop recording."""
        if self._file:
            self._write_record({
                "t": time.time(),
                "event": "session_end",
                "records": self._record_count,
            })
            self._file.close()

        await self.manager.stop()

        elapsed = time.time() - self._start_time
        size_kb = os.path.getsize(self.output_path) / 1024
        print(f"\n\nRecording stopped.")
        print(f"  Records: {self._record_count:,}")
        print(f"  File: {self.output_path} ({size_kb:.0f}KB)")
        print(f"  Duration: {elapsed / 3600:.1f} hours")


def main():
    parser = argparse.ArgumentParser(description="Record Polymarket orderbook data")
    parser.add_argument("--coin", default="BTC", choices=["BTC", "ETH", "SOL", "XRP"])
    parser.add_argument("--duration", type=int, default=5, choices=[5, 15])
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--sample-rate", type=float, default=0.5, help="Seconds between samples (default: 0.5)")

    args = parser.parse_args()

    if args.output:
        output = args.output
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        output = f"data/{args.coin.lower()}-{args.duration}m-{date}.jsonl"

    print(f"Market Data Recorder")
    print(f"  Coin: {args.coin}")
    print(f"  Duration: {args.duration}m")
    print(f"  Sample rate: every {args.sample_rate}s")
    print(f"  Output: {output}")
    print(f"  Press Ctrl+C to stop\n")

    recorder = DataRecorder(
        coin=args.coin,
        duration=args.duration,
        output_path=output,
        sample_interval=args.sample_rate,
    )

    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
