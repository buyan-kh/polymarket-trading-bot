"""
Signer Module - EIP-712 Order Signing

Provides EIP-712 signature functionality for Polymarket orders
and authentication messages.

Uses the same signing approach as the official Polymarket
python-order-utils library (poly_eip712_structs + keccak + _sign_hash).

Example:
    from src.signer import OrderSigner

    signer = OrderSigner(private_key)
    signature = signer.sign_order(
        token_id="123...",
        price=0.65,
        size=10,
        side="BUY",
        maker="0x..."
    )
"""

import time
from random import random
from datetime import datetime, timezone
from math import floor, ceil
from decimal import Decimal
from typing import Optional, Dict, Any
from dataclasses import dataclass
from eth_account import Account
from eth_utils import to_checksum_address, keccak
from poly_eip712_structs import EIP712Struct, Address, String, Uint, make_domain


# USDC has 6 decimal places
USDC_DECIMALS = 6

# Exchange contract addresses (Polygon mainnet)
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# --- EIP-712 Struct Definitions (matching official python-order-utils) ---

class Order(EIP712Struct):
    """EIP-712 Order struct for CTF Exchange. Name must be 'Order' for correct EIP-712 type hash."""
    salt = Uint(256)
    maker = Address()
    signer = Address()
    taker = Address()
    tokenId = Uint(256)
    makerAmount = Uint(256)
    takerAmount = Uint(256)
    expiration = Uint(256)
    nonce = Uint(256)
    feeRateBps = Uint(256)
    side = Uint(8)
    signatureType = Uint(8)


class ClobAuth(EIP712Struct):
    """EIP-712 ClobAuth struct for L1 authentication."""
    address = Address()
    timestamp = String()
    nonce = Uint()
    message = String()


# --- Rounding helpers (matching official py-clob-client) ---

def _round_down(x: float, sig_digits: int) -> float:
    return floor(x * (10 ** sig_digits)) / (10 ** sig_digits)


def _round_normal(x: float, sig_digits: int) -> float:
    return round(x * (10 ** sig_digits)) / (10 ** sig_digits)


def _round_up(x: float, sig_digits: int) -> float:
    return ceil(x * (10 ** sig_digits)) / (10 ** sig_digits)


def _to_token_decimals(x: float) -> int:
    f = (10 ** 6) * x
    if _decimal_places(f) > 0:
        f = _round_normal(f, 0)
    return int(f)


def _decimal_places(x: float) -> int:
    return abs(Decimal(x.__str__()).as_tuple().exponent)


def _generate_seed() -> int:
    """Pseudo random seed (matches official generate_seed)."""
    now = datetime.now().replace(tzinfo=timezone.utc).timestamp()
    return round(now * random())


# Rounding config per tick size
ROUNDING_CONFIG = {
    "0.1": {"price": 1, "size": 2, "amount": 3},
    "0.01": {"price": 2, "size": 2, "amount": 4},
    "0.001": {"price": 3, "size": 2, "amount": 5},
    "0.0001": {"price": 4, "size": 2, "amount": 6},
}


def get_order_amounts(side: str, size: float, price: float, tick_size: str = "0.01"):
    """
    Calculate maker/taker amounts matching official py-clob-client logic.

    Returns:
        (side_int, maker_amount, taker_amount)
    """
    rc = ROUNDING_CONFIG[tick_size]

    raw_price = _round_normal(price, rc["price"])

    if side == "BUY":
        raw_taker_amt = _round_down(size, rc["size"])
        raw_maker_amt = raw_taker_amt * raw_price
        if _decimal_places(raw_maker_amt) > rc["amount"]:
            raw_maker_amt = _round_up(raw_maker_amt, rc["amount"] + 4)
            if _decimal_places(raw_maker_amt) > rc["amount"]:
                raw_maker_amt = _round_down(raw_maker_amt, rc["amount"])
        return 0, _to_token_decimals(raw_maker_amt), _to_token_decimals(raw_taker_amt)

    elif side == "SELL":
        raw_maker_amt = _round_down(size, rc["size"])
        raw_taker_amt = raw_maker_amt * raw_price
        if _decimal_places(raw_taker_amt) > rc["amount"]:
            raw_taker_amt = _round_up(raw_taker_amt, rc["amount"] + 4)
            if _decimal_places(raw_taker_amt) > rc["amount"]:
                raw_taker_amt = _round_down(raw_taker_amt, rc["amount"])
        return 1, _to_token_decimals(raw_maker_amt), _to_token_decimals(raw_taker_amt)

    else:
        raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")


@dataclass
class OrderData:
    """
    Represents a Polymarket order's parameters.

    Attributes:
        token_id: The ERC-1155 token ID for the market outcome
        price: Price per share (0-1, e.g., 0.65 = 65%)
        size: Number of shares
        side: Order side ('BUY' or 'SELL')
        maker: The maker's wallet address (Safe/Proxy)
        nonce: Order nonce (default 0, used for onchain cancellations)
        fee_rate_bps: Fee rate in basis points (usually 0)
        signature_type: Signature type (0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE)
        neg_risk: Whether market uses neg risk exchange
        tick_size: Market tick size for rounding
    """
    token_id: str
    price: float
    size: float
    side: str
    maker: str
    nonce: int = 0
    fee_rate_bps: int = 0
    signature_type: int = 2
    neg_risk: bool = False
    tick_size: str = "0.01"

    def __post_init__(self):
        """Validate and normalize order parameters."""
        self.side = self.side.upper()
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")

        if not 0 < self.price <= 1:
            raise ValueError(f"Invalid price: {self.price}")

        if self.size <= 0:
            raise ValueError(f"Invalid size: {self.size}")

        # Calculate amounts using official rounding logic
        self.side_value, self.maker_amount, self.taker_amount = get_order_amounts(
            self.side, self.size, self.price, self.tick_size
        )


class SignerError(Exception):
    """Base exception for signer operations."""
    pass


class OrderSigner:
    """
    Signs Polymarket orders using EIP-712.

    Uses the same approach as the official python-order-utils library:
    - poly_eip712_structs for EIP-712 struct definitions
    - keccak hash of signable_bytes for struct hash
    - Account._sign_hash for signing
    """

    def __init__(self, private_key: str, chain_id: int = 137):
        """
        Initialize signer with a private key.

        Args:
            private_key: Private key (with or without 0x prefix)
            chain_id: Chain ID (137 for Polygon mainnet)

        Raises:
            ValueError: If private key is invalid
        """
        if private_key.startswith("0x"):
            private_key = private_key[2:]

        self._private_key = f"0x{private_key}"

        try:
            self.wallet = Account.from_key(self._private_key)
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")

        self.address = self.wallet.address
        self.chain_id = chain_id

        # Build EIP-712 domains (matching official libraries exactly)
        self._auth_domain = make_domain(
            name="ClobAuthDomain",
            version="1",
            chainId=chain_id,
        )

        self._order_domain = make_domain(
            name="Polymarket CTF Exchange",
            version="1",
            chainId=str(chain_id),
            verifyingContract=CTF_EXCHANGE,
        )

        self._neg_risk_order_domain = make_domain(
            name="Polymarket CTF Exchange",
            version="1",
            chainId=str(chain_id),
            verifyingContract=NEG_RISK_CTF_EXCHANGE,
        )

    def _sign_hash(self, struct_hash: str) -> str:
        """
        Sign an EIP-712 struct hash (matching official Signer.sign).

        Args:
            struct_hash: Hex string of the hash to sign (with 0x prefix)

        Returns:
            Hex-encoded signature with 0x prefix
        """
        sig = Account._sign_hash(struct_hash, self._private_key)
        return "0x" + sig.signature.hex()

    def _create_struct_hash(self, struct: EIP712Struct, domain) -> str:
        """
        Create EIP-712 struct hash (matching official BaseBuilder._create_struct_hash).

        Args:
            struct: EIP-712 struct instance
            domain: EIP-712 domain separator

        Returns:
            Hex string of the keccak hash with 0x prefix
        """
        return "0x" + keccak(struct.signable_bytes(domain=domain)).hex()

    @classmethod
    def from_encrypted(
        cls,
        encrypted_data: dict,
        password: str
    ) -> "OrderSigner":
        from .crypto import KeyManager, InvalidPasswordError
        manager = KeyManager()
        private_key = manager.decrypt(encrypted_data, password)
        return cls(private_key)

    def sign_auth_message(
        self,
        timestamp: Optional[str] = None,
        nonce: int = 0
    ) -> str:
        """
        Sign an authentication message for L1 authentication.
        Matches official sign_clob_auth_message exactly.

        Args:
            timestamp: Message timestamp (defaults to current time)
            nonce: Message nonce (usually 0)

        Returns:
            Hex-encoded signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        # Build ClobAuth struct (matching official ClobAuth EIP712Struct)
        clob_auth_msg = ClobAuth(
            address=self.address,
            timestamp=timestamp,
            nonce=nonce,
            message="This message attests that I control the given wallet",
        )

        # Hash and sign (matching official approach)
        struct_hash = self._create_struct_hash(clob_auth_msg, self._auth_domain)
        return self._sign_hash(struct_hash)

    def sign_order(self, order: OrderData) -> Dict[str, Any]:
        """
        Sign a Polymarket order.
        Matches official OrderBuilder.build_signed_order exactly.

        Args:
            order: Order instance to sign

        Returns:
            Dictionary containing order and signature

        Raises:
            SignerError: If signing fails
        """
        try:
            salt = _generate_seed()

            # Select domain based on neg_risk
            domain = self._neg_risk_order_domain if order.neg_risk else self._order_domain

            # Build Order EIP712Struct (name must be 'Order' for correct type hash)
            order_struct = Order(
                salt=salt,
                maker=to_checksum_address(order.maker),
                signer=self.address,
                taker=to_checksum_address(ZERO_ADDRESS),
                tokenId=int(order.token_id),
                makerAmount=order.maker_amount,
                takerAmount=order.taker_amount,
                expiration=0,
                nonce=order.nonce,
                feeRateBps=order.fee_rate_bps,
                side=order.side_value,
                signatureType=order.signature_type,
            )

            # Hash and sign (matching official approach)
            struct_hash = self._create_struct_hash(order_struct, domain)
            sig_hex = self._sign_hash(struct_hash)

            # Build API body matching official SignedOrder.dict() format:
            # - salt: int
            # - signatureType: int
            # - side: "BUY"/"SELL" string
            # - tokenId, makerAmount, takerAmount, expiration, nonce, feeRateBps: string
            # - addresses: checksum strings
            return {
                "order": {
                    "salt": salt,
                    "maker": to_checksum_address(order.maker),
                    "signer": self.address,
                    "taker": ZERO_ADDRESS,
                    "tokenId": str(order.token_id),
                    "makerAmount": str(order.maker_amount),
                    "takerAmount": str(order.taker_amount),
                    "expiration": "0",
                    "nonce": str(order.nonce),
                    "feeRateBps": str(order.fee_rate_bps),
                    "side": order.side,  # "BUY" or "SELL"
                    "signatureType": order.signature_type,
                    "signature": sig_hex,
                },
                "signature": sig_hex,
                "signer": self.address,
            }

        except Exception as e:
            raise SignerError(f"Failed to sign order: {e}")

    def sign_order_dict(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        maker: str,
        nonce: int = 0,
        fee_rate_bps: int = 0,
        signature_type: int = 2,
        neg_risk: bool = False,
        tick_size: str = "0.01",
    ) -> Dict[str, Any]:
        """
        Sign an order from dictionary parameters.
        """
        order = OrderData(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            maker=maker,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
            signature_type=signature_type,
            neg_risk=neg_risk,
            tick_size=tick_size,
        )
        return self.sign_order(order)

    def sign_message(self, message: str) -> str:
        """
        Sign a plain text message (for API key derivation).
        """
        from eth_account.messages import encode_defunct
        signable = encode_defunct(text=message)
        signed = self.wallet.sign_message(signable)
        return "0x" + signed.signature.hex()


# Alias for backwards compatibility
WalletSigner = OrderSigner
