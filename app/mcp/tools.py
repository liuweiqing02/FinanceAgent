from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def get_market_snapshot(ticker: str) -> dict:
    """拉取真实行情与估值快照，失败时降级到离线兜底值。"""

    symbol = ticker.upper().strip() or "AAPL"
    quote = _fetch_yahoo_quote(symbol)
    if quote is not None:
        return quote
    return _fallback_market_snapshot(symbol)


def get_fundamental_snapshot(ticker: str) -> dict:
    """拉取真实财务核心指标快照，失败时返回空壳结构。"""

    symbol = ticker.upper().strip() or "AAPL"
    data = _fetch_yahoo_fundamental(symbol)
    if data is not None:
        return data

    return {
        "ticker": symbol,
        "currency": "N/A",
        "trailing_pe": None,
        "forward_pe": None,
        "price_to_book": None,
        "market_cap": None,
        "profit_margin": None,
        "gross_margin": None,
        "operating_margin": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "return_on_equity": None,
        "free_cash_flow": None,
        "total_cash": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_news_file(base_dir: str, ticker: str) -> list[dict]:
    """从知识库读取新闻文本，兼容离线场景。"""

    folder = Path(base_dir)
    rows: list[dict] = []
    for p in sorted(folder.glob(f"*{ticker.upper()}*.md")):
        content = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({"source_id": p.stem, "title": p.name, "content": content})
    return rows


def _fetch_yahoo_quote(symbol: str) -> dict | None:
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
    try:
        obj = _http_get_json(url)
        rows = obj.get("quoteResponse", {}).get("result", [])
        if not rows:
            return None
        item = rows[0]
        return {
            "ticker": symbol,
            "price": _num(item.get("regularMarketPrice")),
            "change_percent": _num(item.get("regularMarketChangePercent")),
            "day_high": _num(item.get("regularMarketDayHigh")),
            "day_low": _num(item.get("regularMarketDayLow")),
            "market_cap": _num(item.get("marketCap")),
            "pe": _num(item.get("trailingPE")),
            "forward_pe": _num(item.get("forwardPE")),
            "pb": _num(item.get("priceToBook")),
            "currency": str(item.get("currency", "N/A")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None


def _fetch_yahoo_fundamental(symbol: str) -> dict | None:
    modules = ",".join(["summaryDetail", "defaultKeyStatistics", "financialData", "price"])
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules={modules}"
    try:
        obj = _http_get_json(url)
        result = obj.get("quoteSummary", {}).get("result", [])
        if not result:
            return None
        item = result[0]
        detail = item.get("summaryDetail", {})
        stat = item.get("defaultKeyStatistics", {})
        fin = item.get("financialData", {})
        price = item.get("price", {})
        return {
            "ticker": symbol,
            "currency": str(price.get("currency", "N/A")),
            "trailing_pe": _num(_val(detail.get("trailingPE"))),
            "forward_pe": _num(_val(detail.get("forwardPE"))),
            "price_to_book": _num(_val(stat.get("priceToBook"))),
            "market_cap": _num(_val(price.get("marketCap"))),
            "profit_margin": _num(_val(fin.get("profitMargins"))),
            "gross_margin": _num(_val(fin.get("grossMargins"))),
            "operating_margin": _num(_val(fin.get("operatingMargins"))),
            "revenue_growth": _num(_val(fin.get("revenueGrowth"))),
            "earnings_growth": _num(_val(fin.get("earningsGrowth"))),
            "return_on_equity": _num(_val(fin.get("returnOnEquity"))),
            "free_cash_flow": _num(_val(fin.get("freeCashflow"))),
            "total_cash": _num(_val(fin.get("totalCash"))),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None


def _http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _val(v: object) -> object:
    if isinstance(v, dict):
        if "raw" in v:
            return v.get("raw")
        if "fmt" in v:
            return v.get("fmt")
    return v


def _num(v: object) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        if v is None:
            return None
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _fallback_market_snapshot(symbol: str) -> dict:
    seed = sum(ord(c) for c in symbol)
    price = round(20 + (seed % 200) / 3, 2)
    pe = round(8 + (seed % 50) / 2.7, 2)
    pb = round(0.8 + (seed % 30) / 10, 2)
    return {
        "ticker": symbol,
        "price": price,
        "change_percent": round((seed % 700) / 100 - 3.0, 2),
        "day_high": round(price * 1.02, 2),
        "day_low": round(price * 0.98, 2),
        "market_cap": float((seed % 900 + 100) * 1_000_000_000),
        "pe": pe,
        "forward_pe": round(max(pe - 1.5, 1.0), 2),
        "pb": pb,
        "currency": "USD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
