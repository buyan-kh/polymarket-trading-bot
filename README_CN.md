# Polymarket 交易机器人

[English](README.md) | 简体中文

一个新手友好的 Python 交易机器人，支持 Polymarket 无 Gas 交易和实时 WebSocket 数据。

## 特性

- **简单易用**：几行代码即可开始交易
- **零 Gas 费用**：使用 Builder Program 凭证免除 Gas 费
- **实时 WebSocket**：通过 WebSocket 实时获取订单簿更新
- **15分钟市场**：内置支持 BTC/ETH/SOL/XRP 15分钟涨跌市场
- **闪崩策略**：预置波动率交易策略
- **终端界面**：实时订单簿显示，原地更新
- **安全存储**：私钥使用 PBKDF2 + Fernet 加密保护
- **完整测试**：89 个单元测试覆盖所有功能

## 快速开始（5 分钟）

### 第一步：安装

```bash
git clone https://github.com/your-username/polymarket-trading-bot.git
cd polymarket-trading-bot
pip install -r requirements.txt
```

### 第二步：配置

```bash
# 设置你的凭证
export POLY_PRIVATE_KEY=你的MetaMask私钥
export POLY_SAFE_ADDRESS=0x你的Polymarket钱包地址
```

> **如何找到 Safe 地址？** 访问 [polymarket.com/settings](https://polymarket.com/settings)，复制你的钱包地址。

### 第三步：运行

```bash
# 运行快速入门示例
python examples/quickstart.py

# 或运行闪崩策略
python strategies/flash_crash_strategy.py --coin BTC
```

就这么简单！你已经准备好开始交易了。

## 交易策略

### 闪崩策略 (Flash Crash Strategy)

监控15分钟涨跌市场的概率突然下跌，并自动执行交易。

```bash
# 使用默认设置运行（0.30 下跌阈值）
python strategies/flash_crash_strategy.py --coin BTC

# 自定义设置
python strategies/flash_crash_strategy.py --coin ETH --drop 0.25 --size 10

# 可用选项
--coin      BTC, ETH, SOL, XRP（默认：ETH）
--drop      下跌阈值，绝对变化值（默认：0.30）
--size      交易金额，USDC（默认：5.0）
--lookback  检测窗口，秒（默认：10）
--take-profit  止盈金额（默认：0.10）
--stop-loss    止损金额（默认：0.05）
```

**策略逻辑：**
1. 自动发现当前15分钟市场
2. 通过 WebSocket 实时监控订单簿价格
3. 当概率在10秒内下跌0.30+时，买入崩盘的一方
4. 在 +$0.10（止盈）或 -$0.05（止损）时退出

### 实时订单簿界面

在终端中查看实时订单簿数据：

```bash
python strategies/orderbook_tui.py --coin BTC --levels 5
```

## 代码示例

### 最简单的例子

```python
from src import create_bot_from_env
import asyncio

async def main():
    # 从环境变量创建机器人
    bot = create_bot_from_env()

    # 获取你的挂单
    orders = await bot.get_open_orders()
    print(f"你有 {len(orders)} 个挂单")

asyncio.run(main())
```

### 下单交易

```python
from src import TradingBot, Config
import asyncio

async def trade():
    # 创建配置
    config = Config(safe_address="0x你的Safe地址")

    # 使用私钥初始化机器人
    bot = TradingBot(config=config, private_key="0x你的私钥")

    # 下一个买单
    result = await bot.place_order(
        token_id="12345...",   # 市场代币 ID
        price=0.65,            # 价格（0.65 = 65% 概率）
        size=10.0,             # 股数
        side="BUY"             # 或 "SELL"
    )

    if result.success:
        print(f"下单成功！订单号：{result.order_id}")
    else:
        print(f"下单失败：{result.message}")

asyncio.run(trade())
```

### 实时 WebSocket 数据

```python
from src.websocket_client import MarketWebSocket, OrderbookSnapshot
import asyncio

async def main():
    ws = MarketWebSocket()

    @ws.on_book
    async def on_book_update(snapshot: OrderbookSnapshot):
        print(f"中间价：{snapshot.mid_price:.4f}")
        print(f"最高买价：{snapshot.best_bid:.4f}")
        print(f"最低卖价：{snapshot.best_ask:.4f}")

    await ws.subscribe(["token_id_1", "token_id_2"])
    await ws.run()

asyncio.run(main())
```

### 获取15分钟市场信息

```python
from src.gamma_client import GammaClient

gamma = GammaClient()

# 获取当前 BTC 15分钟市场
market = gamma.get_market_info("BTC")
print(f"市场：{market['question']}")
print(f"Up代币：{market['token_ids']['up']}")
print(f"Down代币：{market['token_ids']['down']}")
print(f"结束时间：{market['end_date']}")
```

### 撤销订单

```python
# 撤销指定订单
await bot.cancel_order("订单ID")

# 撤销所有订单
await bot.cancel_all_orders()

# 撤销特定市场的订单
await bot.cancel_market_orders(market="condition_id", asset_id="token_id")
```

## 项目结构

```
polymarket-trading-bot/
├── src/                      # 核心库
│   ├── bot.py               # TradingBot - 主接口
│   ├── config.py            # 配置管理
│   ├── client.py            # API 客户端（CLOB、Relayer）
│   ├── signer.py            # 订单签名（EIP-712）
│   ├── crypto.py            # 私钥加密
│   ├── utils.py             # 辅助函数
│   ├── gamma_client.py      # 15分钟市场发现
│   └── websocket_client.py  # 实时 WebSocket 客户端
│
├── strategies/               # 交易策略
│   ├── flash_crash_strategy.py  # 波动率交易策略
│   └── orderbook_tui.py     # 实时订单簿显示
│
├── examples/                 # 示例代码
│   ├── quickstart.py        # 从这里开始！
│   ├── basic_trading.py     # 常用操作
│   └── strategy_example.py  # 自定义策略
│
├── scripts/                  # 工具脚本
│   ├── setup.py             # 交互式设置
│   ├── run_bot.py           # 运行机器人
│   └── full_test.py         # 集成测试
│
└── tests/                    # 单元测试
```

## 配置选项

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `POLY_PRIVATE_KEY` | 是 | 你的钱包私钥 |
| `POLY_SAFE_ADDRESS` | 是 | 你的 Polymarket Safe 地址 |
| `POLY_BUILDER_API_KEY` | 无 Gas 需要 | Builder Program API 密钥 |
| `POLY_BUILDER_API_SECRET` | 无 Gas 需要 | Builder Program 密钥 |
| `POLY_BUILDER_API_PASSPHRASE` | 无 Gas 需要 | Builder Program 口令 |

### 配置文件（另一种方式）

创建 `config.yaml`：

```yaml
safe_address: "0x你的Safe地址"

# 无 Gas 交易（可选）
builder:
  api_key: "你的api_key"
  api_secret: "你的api_secret"
  api_passphrase: "你的passphrase"
```

然后加载它：

```python
bot = TradingBot(config_path="config.yaml", private_key="0x...")
```

## 无 Gas 交易

要免除 Gas 费用：

1. 申请 [Builder Program](https://polymarket.com/settings?tab=builder)
2. 设置环境变量：

```bash
export POLY_BUILDER_API_KEY=你的密钥
export POLY_BUILDER_API_SECRET=你的密钥
export POLY_BUILDER_API_PASSPHRASE=你的口令
```

当凭证存在时，机器人会自动使用无 Gas 模式。

## API 参考

### TradingBot 方法

| 方法 | 说明 |
|------|------|
| `place_order(token_id, price, size, side)` | 下限价单 |
| `cancel_order(order_id)` | 撤销指定订单 |
| `cancel_all_orders()` | 撤销所有挂单 |
| `cancel_market_orders(market, asset_id)` | 撤销特定市场的订单 |
| `get_open_orders()` | 获取挂单列表 |
| `get_trades(limit=100)` | 获取交易历史 |
| `get_order_book(token_id)` | 获取市场订单簿 |
| `get_market_price(token_id)` | 获取当前市场价格 |
| `is_initialized()` | 检查机器人是否就绪 |

### MarketWebSocket 方法

| 方法 | 说明 |
|------|------|
| `subscribe(asset_ids, replace=False)` | 订阅市场数据 |
| `run(auto_reconnect=True)` | 启动 WebSocket 连接 |
| `disconnect()` | 关闭连接 |
| `get_orderbook(asset_id)` | 获取缓存的订单簿 |
| `get_mid_price(asset_id)` | 获取中间价 |

### GammaClient 方法

| 方法 | 说明 |
|------|------|
| `get_current_15m_market(coin)` | 获取当前15分钟市场 |
| `get_market_info(coin)` | 获取市场信息含代币ID |
| `get_all_15m_markets()` | 列出所有15分钟市场 |

## 安全性

你的私钥受到以下保护：

1. **PBKDF2** 密钥派生（480,000 次迭代）
2. **Fernet** 对称加密
3. 文件权限设置为 `0600`（仅所有者可读）

最佳实践：
- 永远不要将 `.env` 文件提交到 git
- 使用专门的钱包进行交易
- 妥善保管你的加密密钥文件

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并显示覆盖率
pytest tests/ -v --cov=src
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `POLY_PRIVATE_KEY not set` | 运行 `export POLY_PRIVATE_KEY=你的私钥` |
| `POLY_SAFE_ADDRESS not set` | 从 polymarket.com/settings 获取 |
| `Invalid private key` | 检查私钥是否为 64 个十六进制字符 |
| `Order failed` | 检查是否有足够的余额 |
| `WebSocket not connecting` | 检查网络/防火墙设置 |

## 新手学习路径

1. 首先阅读 `examples/quickstart.py` - 最简单的示例
2. 然后看 `examples/basic_trading.py` - 常用操作
3. 研究 `src/bot.py` - 理解核心类
4. 运行 `strategies/flash_crash_strategy.py` - 实战策略
5. 最后看 `examples/strategy_example.py` - 自定义策略

## 贡献

1. Fork 本仓库
2. 创建功能分支
3. 为新代码编写测试
4. 运行 `pytest tests/ -v`
5. 提交 Pull Request

## 许可证

MIT 许可证 - 详见 LICENSE 文件。
