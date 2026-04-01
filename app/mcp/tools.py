from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


_SEC_TICKER_CACHE: dict[str, str] | None = None
_SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "FinanceAgent/1.0 contact@example.com"),
    "Accept": "application/json",
}


def get_market_snapshot(ticker: str) -> dict:
    """拉取行情与估值快照。优先 Yahoo Chart（可用），失败回退 Stooq。"""

    symbol = ticker.upper().strip() or "AAPL"
    quote = _fetch_price_from_yahoo_chart(symbol) or _fetch_price_from_stooq(symbol)
    sec = _fetch_sec_fundamental(symbol, price=quote.get("price") if quote else None) if quote else None

    if quote is not None:
        return {
            "ticker": symbol,
            "price": _num(quote.get("price")),
            "change_percent": _num(quote.get("change_percent")),
            "day_high": _num(quote.get("day_high")),
            "day_low": _num(quote.get("day_low")),
            "market_cap": _num((sec or {}).get("market_cap")),
            "pe": _num((sec or {}).get("trailing_pe")),
            "forward_pe": _num((sec or {}).get("forward_pe")),
            "pb": _num((sec or {}).get("price_to_book")),
            "currency": str(quote.get("currency") or "USD"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_status": "ok",
            "source": quote.get("source", "unknown"),
        }

    return {
        "ticker": symbol,
        "price": None,
        "change_percent": None,
        "day_high": None,
        "day_low": None,
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "pb": None,
        "currency": "N/A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_status": "unavailable",
    }


def get_fundamental_snapshot(ticker: str) -> dict:
    """拉取财务核心指标快照。优先 SEC Companyfacts，失败回退 Yahoo。"""

    symbol = ticker.upper().strip() or "AAPL"
    quote = _fetch_price_from_yahoo_chart(symbol) or _fetch_price_from_stooq(symbol)
    price = quote.get("price") if quote else None

    sec_data = _fetch_sec_fundamental(symbol, price=price)
    if sec_data is not None and _fundamental_signal_count(sec_data) >= 2:
        sec_data["data_status"] = "ok"
        sec_data["source"] = "sec_companyfacts"
        return sec_data

    yahoo = _fetch_yahoo_fundamental(symbol)
    if yahoo is not None and _fundamental_signal_count(yahoo) >= 2:
        yahoo["data_status"] = "ok"
        yahoo["source"] = "yahoo_quotesummary"
        return yahoo

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
        "data_status": "unavailable",
    }


def load_news_file(base_dir: str, ticker: str) -> list[dict]:
    """从知识库读取新闻文本，兼容离线场景。"""

    folder = Path(base_dir)
    rows: list[dict] = []
    for p in sorted(folder.glob(f"*{ticker.upper()}*.md")):
        content = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({"source_id": p.stem, "title": p.name, "content": content})
    return rows


def _fetch_price_from_yahoo_chart(symbol: str) -> dict | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d"
    try:
        obj = _http_get_json(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        result = obj.get("chart", {}).get("result", [])
        if not result:
            return None
        r0 = result[0]
        quote = (r0.get("indicators", {}).get("quote", []) or [{}])[0]
        closes = [x for x in (quote.get("close") or []) if isinstance(x, (int, float))]
        highs = [x for x in (quote.get("high") or []) if isinstance(x, (int, float))]
        lows = [x for x in (quote.get("low") or []) if isinstance(x, (int, float))]
        if not closes:
            return None
        price = float(closes[-1])
        prev = float(closes[-2]) if len(closes) >= 2 else None
        chg = ((price - prev) / prev * 100.0) if prev and prev != 0 else None
        day_high = float(highs[-1]) if highs else None
        day_low = float(lows[-1]) if lows else None
        currency = str((r0.get("meta") or {}).get("currency") or "USD")
        return {
            "price": price,
            "change_percent": chg,
            "day_high": day_high,
            "day_low": day_low,
            "currency": currency,
            "source": "yahoo_chart",
        }
    except Exception:  # noqa: BLE001
        return None


def _fetch_price_from_stooq(symbol: str) -> dict | None:
    url = f"https://stooq.com/q/l/?s={symbol.lower()}.us&i=d"
    try:
        text = _http_get_text(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"})
        line = next((x.strip() for x in text.splitlines() if x.strip()), "")
        if not line:
            return None
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 8:
            return None
        open_p = _num(parts[3])
        high = _num(parts[4])
        low = _num(parts[5])
        close = _num(parts[6])
        if close is None:
            return None
        chg = ((close - open_p) / open_p * 100.0) if isinstance(open_p, float) and open_p != 0 else None
        return {
            "price": close,
            "change_percent": chg,
            "day_high": high,
            "day_low": low,
            "currency": "USD",
            "source": "stooq_daily",
        }
    except Exception:  # noqa: BLE001
        return None


def _fetch_sec_fundamental(symbol: str, price: float | None) -> dict | None:
    cik = _sec_cik(symbol)
    if not cik:
        return None

    try:
        obj = _http_get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=_SEC_HEADERS,
        )
    except Exception:  # noqa: BLE001
        return None

    facts = obj.get("facts", {})
    gaap = facts.get("us-gaap", {})
    dei = facts.get("dei", {})

    rev_latest, rev_prev = _latest_annual_pair(gaap, "Revenues", ["USD"])
    ni_latest, ni_prev = _latest_annual_pair(gaap, "NetIncomeLoss", ["USD"])
    gp_latest, _ = _latest_annual_pair(gaap, "GrossProfit", ["USD"])
    op_latest, _ = _latest_annual_pair(gaap, "OperatingIncomeLoss", ["USD"])
    cfo_latest, _ = _latest_annual_pair(gaap, "NetCashProvidedByUsedInOperatingActivities", ["USD"])
    capex_latest, _ = _latest_annual_pair(gaap, "PaymentsToAcquirePropertyPlantAndEquipment", ["USD"])

    equity = _latest_point_value(
        gaap,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        ["USD"],
    )
    total_cash = _latest_point_value(
        gaap,
        ["CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalentsAtCarryingValue"],
        ["USD"],
    )
    shares = _latest_point_value(dei, ["EntityCommonStockSharesOutstanding"], ["shares"])
    if shares is None:
        shares, _ = _latest_annual_pair(gaap, "WeightedAverageNumberOfDilutedSharesOutstanding", ["shares"])

    eps, _ = _latest_annual_pair(gaap, "EarningsPerShareDiluted", ["USD/shares"])

    revenue_growth = _growth(rev_latest, rev_prev)
    earnings_growth = _growth(ni_latest, ni_prev)
    profit_margin = _ratio(ni_latest, rev_latest)
    gross_margin = _ratio(gp_latest, rev_latest)
    operating_margin = _ratio(op_latest, rev_latest)
    return_on_equity = _ratio(ni_latest, equity)

    free_cash_flow = None
    if isinstance(cfo_latest, float):
        free_cash_flow = cfo_latest
        if isinstance(capex_latest, float):
            free_cash_flow = cfo_latest - abs(capex_latest)

    market_cap = None
    if isinstance(price, float) and isinstance(shares, float) and shares > 0:
        market_cap = price * shares

    trailing_pe = None
    if isinstance(price, float) and isinstance(eps, float) and eps > 0:
        trailing_pe = price / eps

    price_to_book = None
    if isinstance(market_cap, float) and isinstance(equity, float) and equity > 0:
        price_to_book = market_cap / equity

    return {
        "ticker": symbol,
        "currency": "USD",
        "trailing_pe": trailing_pe,
        "forward_pe": None,
        "price_to_book": price_to_book,
        "market_cap": market_cap,
        "profit_margin": profit_margin,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "return_on_equity": return_on_equity,
        "free_cash_flow": free_cash_flow,
        "total_cash": total_cash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _sec_cik(symbol: str) -> str | None:
    global _SEC_TICKER_CACHE
    if _SEC_TICKER_CACHE is None:
        obj = _http_get_json("https://www.sec.gov/files/company_tickers.json", headers=_SEC_HEADERS)
        cache: dict[str, str] = {}
        if isinstance(obj, dict):
            for item in obj.values():
                if not isinstance(item, dict):
                    continue
                t = str(item.get("ticker", "")).upper().strip()
                cik = str(item.get("cik_str", "")).strip()
                if t and cik:
                    cache[t] = cik.zfill(10)
        _SEC_TICKER_CACHE = cache
    return (_SEC_TICKER_CACHE or {}).get(symbol.upper().strip())


def _annual_rows(container: dict, concept: str, unit_candidates: list[str]) -> list[dict]:
    concept_obj = container.get(concept, {}) if isinstance(container, dict) else {}
    units = concept_obj.get("units", {}) if isinstance(concept_obj, dict) else {}
    rows: list[dict] = []
    for u in unit_candidates:
        vals = units.get(u)
        if isinstance(vals, list):
            rows.extend(vals)
    if not rows and isinstance(units, dict):
        for vals in units.values():
            if isinstance(vals, list):
                rows.extend(vals)
                if rows:
                    break

    kept: dict[int, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("form") != "10-K":
            continue
        fy = r.get("fy")
        val = _num(r.get("val"))
        if not isinstance(fy, int) or not isinstance(val, float):
            continue
        cur = kept.get(fy)
        if cur is None or str(r.get("filed", "")) > str(cur.get("filed", "")):
            kept[fy] = dict(r)

    out = sorted(kept.values(), key=lambda x: int(x.get("fy", 0)), reverse=True)
    return out


def _latest_annual_pair(container: dict, concept: str, unit_candidates: list[str]) -> tuple[float | None, float | None]:
    rows = _annual_rows(container, concept, unit_candidates)
    latest = _num(rows[0].get("val")) if len(rows) >= 1 else None
    prev = _num(rows[1].get("val")) if len(rows) >= 2 else None
    return latest, prev


def _latest_point_value(container: dict, concepts: list[str], unit_candidates: list[str]) -> float | None:
    best: tuple[str, float] | None = None
    for c in concepts:
        concept_obj = container.get(c, {}) if isinstance(container, dict) else {}
        units = concept_obj.get("units", {}) if isinstance(concept_obj, dict) else {}
        rows: list[dict] = []
        for u in unit_candidates:
            vals = units.get(u)
            if isinstance(vals, list):
                rows.extend(vals)
        if not rows and isinstance(units, dict):
            for vals in units.values():
                if isinstance(vals, list):
                    rows.extend(vals)
                    if rows:
                        break
        for r in rows:
            if not isinstance(r, dict):
                continue
            val = _num(r.get("val"))
            if not isinstance(val, float):
                continue
            key = f"{r.get('filed','')}|{r.get('end','')}"
            if best is None or key > best[0]:
                best = (key, val)
    return best[1] if best else None


def _growth(latest: float | None, prev: float | None) -> float | None:
    if not isinstance(latest, float) or not isinstance(prev, float) or prev == 0:
        return None
    return (latest - prev) / abs(prev)


def _ratio(num: float | None, den: float | None) -> float | None:
    if not isinstance(num, float) or not isinstance(den, float) or den == 0:
        return None
    return num / den


def _fundamental_signal_count(data: dict[str, Any]) -> int:
    keys = [
        "trailing_pe",
        "price_to_book",
        "market_cap",
        "profit_margin",
        "gross_margin",
        "operating_margin",
        "revenue_growth",
        "earnings_growth",
        "return_on_equity",
        "free_cash_flow",
        "total_cash",
    ]
    cnt = 0
    for k in keys:
        if isinstance(data.get(k), (int, float)):
            cnt += 1
    return cnt


def _fetch_yahoo_fundamental(symbol: str) -> dict | None:
    modules = ",".join(["summaryDetail", "defaultKeyStatistics", "financialData", "price"])
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules={modules}"
    try:
        obj = _http_get_json(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
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


def _http_get_text(url: str, headers: dict[str, str] | None = None) -> str:
    req = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="ignore")


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    text = _http_get_text(url, headers=headers or {"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    return json.loads(text)


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
