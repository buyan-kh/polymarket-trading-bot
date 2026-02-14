#!/usr/bin/env python3
"""
Flash Crash Strategy Runner

Entry point for running the flash crash strategy.

Usage:
    python apps/run_flash_crash.py --coin ETH
    python apps/run_flash_crash.py --coin BTC --size 10
    python apps/run_flash_crash.py --coin BTC --drop 0.25
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Suppress noisy logs
logging.getLogger("src.websocket_client").setLevel(logging.WARNING)
logging.getLogger("src.bot").setLevel(logging.WARNING)

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.console import Colors
from src.bot import TradingBot
from src.config import Config
from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flash Crash Strategy for Polymarket 15-minute markets"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default="ETH",
        choices=["BTC", "ETH", "SOL", "XRP"],
        help="Coin to trade (default: ETH)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        choices=[5, 15],
        help="Market duration in minutes (default: 15, only BTC supports 5)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=5.0,
        help="Trade size in USDC (default: 5.0)"
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.30,
        help="Drop threshold as absolute probability change (default: 0.30)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window in seconds (default: 10)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.10,
        help="Take profit in dollars (default: 0.10)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss in dollars (default: 0.05)"
    )
    parser.add_argument(
        "--paper",
        type=float,
        default=None,
        metavar="BALANCE",
        help="Paper trading mode with starting balance in USDC (e.g. --paper 10)"
    )
    parser.add_argument(
        "--bet-fraction",
        type=float,
        default=0.10,
        help="Fraction of balance to bet each trade (default: 0.10)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("src.websocket_client").setLevel(logging.DEBUG)

    # Check environment
    is_paper = args.paper is not None
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    safe_address = os.environ.get("POLY_SAFE_ADDRESS")

    if not is_paper and (not private_key or not safe_address):
        print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS must be set{Colors.RESET}")
        print("Set them in .env file or export as environment variables")
        print(f"  Tip: Use {Colors.CYAN}--paper 10{Colors.RESET} for paper trading without credentials")
        sys.exit(1)

    # Create bot
    if is_paper and not (private_key and safe_address):
        # Paper mode with dummy credentials — bot is never used for real orders
        dummy_key = "0x" + "a" * 64
        dummy_addr = "0x" + "b" * 40
        config = Config.from_env()
        config.safe_address = dummy_addr
        bot = TradingBot(config=config, private_key=dummy_key)
    else:
        config = Config.from_env()
        bot = TradingBot(config=config, private_key=private_key)

        if not bot.is_initialized():
            print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
            sys.exit(1)

    # Create strategy config
    strategy_config = FlashCrashConfig(
        coin=args.coin.upper(),
        size=args.size,
        market_duration=args.duration,
        drop_threshold=args.drop,
        price_lookback_seconds=args.lookback,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        paper=is_paper,
        paper_balance=args.paper if is_paper else 10.0,
        bet_fraction=args.bet_fraction,
    )

    # Print configuration
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Flash Crash Strategy - {strategy_config.coin} {strategy_config.market_duration}-Minute Markets{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    if strategy_config.paper:
        print(f"  {Colors.YELLOW}*** PAPER TRADING MODE - No real orders ***{Colors.RESET}")
    print(f"Configuration:")
    print(f"  Coin: {strategy_config.coin}")
    print(f"  Duration: {strategy_config.market_duration}m")
    if strategy_config.paper:
        print(f"  Starting Balance: ${strategy_config.paper_balance:.2f}")
        print(f"  Bet: {strategy_config.bet_fraction * 100:.0f}% of balance per trade")
    else:
        print(f"  Size: ${strategy_config.size:.2f}")
    print(f"  Drop threshold: {strategy_config.drop_threshold:.2f}")
    print(f"  Lookback: {strategy_config.price_lookback_seconds}s")
    print(f"  Take profit: +${strategy_config.take_profit:.2f}")
    print(f"  Stop loss: -${strategy_config.stop_loss:.2f}")
    print()

    # Create and run strategy
    strategy = FlashCrashStrategy(bot=bot, config=strategy_config)

    try:
        asyncio.run(strategy.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
