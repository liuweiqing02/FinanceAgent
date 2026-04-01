from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree


HttpGetter = Callable[[str, dict[str, str]], str]
_NEWS_LOOKBACK_DAYS = 7
_TICKER_ALIAS_MAP = {
    "GOOGL": "Alphabet Google",
    "GOOG": "Alphabet Google",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
}


@dataclass(slots=True)
class RealCollectStats:
    tickers: list[str]
    filings: int
    news: int
    technical: int
    output_file: Path


def collect_real_raw_data(
    raw_dir: Path,
    tickers: list[str],
    sec_user_agent: str,
    filing_limit: int = 4,
    news_limit: int = 4,
    http_get: HttpGetter | None = None,
) -> RealCollectStats:
    """采集真实财报/公告/新闻/技术文本并落盘为 jsonl。"""

    if not sec_user_agent.strip():
        raise ValueError("SEC_USER_AGENT 不能为空，例如 'FinanceAgent/1.0 your@email.com'")

    getter = http_get or _http_get
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / "real_events.jsonl"

    cik_map = _load_cik_mapping(getter, sec_user_agent)
    rows: list[dict] = []

    filing_count = 0
    news_count = 0
    technical_count = 0
    target_tickers = [t.upper() for t in tickers if t.strip()]
    for ticker in target_tickers:
        cik = cik_map.get(ticker)
        if cik:
            filing_rows = _collect_sec_filings(getter, ticker, cik, sec_user_agent, filing_limit)
            rows.extend(filing_rows)
            filing_count += len(filing_rows)

        rss_rows = _collect_news_rows(getter, ticker, news_limit)
        rows.extend(rss_rows)
        news_count += len(rss_rows)

        tech = _collect_technical_from_yahoo(getter, ticker)
        if tech is None and http_get is None:
            tech = _collect_technical_from_runtime_tool(ticker)
        if tech is not None:
            rows.append(tech)
            technical_count += 1

        if http_get is None:
            fundamental_row = _collect_fundamental_from_runtime_tool(ticker)
        else:
            fundamental_row = _collect_fundamental_from_yahoo(getter, ticker)
        if fundamental_row is not None:
            rows.append(fundamental_row)

        if http_get is None:
            valuation_row = _collect_valuation_from_runtime_tool(ticker)
        else:
            valuation_row = _collect_valuation_from_yahoo(getter, ticker)
        if valuation_row is not None:
            rows.append(valuation_row)

    new_rows = [_enrich_raw_row(x) for x in rows]
    merged_rows = _merge_incremental_rows(out_file, new_rows, set(target_tickers))

    with out_file.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return RealCollectStats(
        tickers=target_tickers,
        filings=filing_count,
        news=news_count,
        technical=technical_count,
        output_file=out_file,
    )
def _load_cik_mapping(getter: HttpGetter, user_agent: str) -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    text = getter(url, {"User-Agent": user_agent, "Accept": "application/json"})
    obj = json.loads(text)
    mapping: dict[str, str] = {}
    for _, item in obj.items():
        ticker = str(item.get("ticker", "")).upper()
        cik = str(item.get("cik_str", "")).strip()
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def _collect_sec_filings(getter: HttpGetter, ticker: str, cik: str, user_agent: str, filing_limit: int) -> list[dict]:
    cik10 = cik.zfill(10)
    sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    text = getter(sub_url, {"User-Agent": user_agent, "Accept": "application/json"})
    obj = json.loads(text)

    recent = obj.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    acc = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    rows: list[dict] = []
    for form, accession, doc, filing_date in zip(forms, acc, docs, dates):
        if form not in {"10-K", "10-Q", "8-K"}:
            continue
        accession_nodash = str(accession).replace("-", "")
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{doc}"
        try:
            html_text = getter(filing_url, {"User-Agent": user_agent, "Accept": "text/html"})
        except URLError:
            continue

        content = _summarize_financial_text(_extract_text_from_html(html_text))
        if len(content) < 200:
            continue

        # 8-K 在投研中依然偏基本面事件，避免 topic 过度倾斜到 news。
        topic = "fundamental"
        title = f"{ticker} {form} filing {filing_date}"

        rows.append(
            {
                "ticker": ticker,
                "topic": topic,
                "title": title,
                "content": content[:5000],
                "source": "sec",
                "url": filing_url,
                "published_at": str(filing_date),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if form in {"10-K", "10-Q"}:
            rows.append(
                {
                    "ticker": ticker,
                    "topic": "valuation",
                    "title": f"{ticker} valuation clues from {form} {filing_date}",
                    "content": _extract_valuation_paragraphs(content),
                    "source": "sec",
                    "url": filing_url,
                    "published_at": str(filing_date),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        if len(rows) >= filing_limit:
            break

    return rows[:filing_limit]


def _collect_news_rows(getter: HttpGetter, ticker: str, news_limit: int) -> list[dict]:
    """Collect news from multiple sources with recency filter and dedup."""

    if news_limit <= 0:
        return []

    per_source_limit = max(8, news_limit * 2)
    candidates: list[dict] = []

    for query in _news_queries_for_ticker(ticker):
        candidates.extend(_collect_google_news_rss(getter, ticker, query, per_source_limit))
    candidates.extend(_collect_yahoo_finance_rss(getter, ticker, per_source_limit))

    recent_rows = [x for x in candidates if _is_recent_news(x.get("published_at", ""))]
    rows = recent_rows if recent_rows else candidates

    dedup: dict[str, dict] = {}
    for row in rows:
        key = _news_dedup_key(row)
        old = dedup.get(key)
        if old is None or len(str(row.get("content", ""))) > len(str(old.get("content", ""))):
            dedup[key] = row

    merged = sorted(dedup.values(), key=lambda x: str(x.get("published_at", "")), reverse=True)
    return merged[:news_limit]


def _collect_google_news_rss(getter: HttpGetter, ticker: str, query: str, limit: int) -> list[dict]:
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        xml_text = getter(url, {"Accept": "application/rss+xml", "User-Agent": "Mozilla/5.0"})
    except Exception:  # noqa: BLE001
        return []

    try:
        root = ElementTree.fromstring(xml_text)
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict] = []
    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _normalize_published_at(item.findtext("pubDate") or "")

        desc_html = item.findtext("description") or ""
        desc_plain = _extract_text_from_html(desc_html)
        if not _is_ticker_relevant(ticker, title, desc_plain, link):
            continue
        desc = _summarize_financial_text(desc_plain, 4)
        body = _collect_article_snippet(getter, link, max_sentences=6)
        content = _merge_news_content(desc, body)
        if len(content) < 30:
            continue

        rows.append(
            {
                "ticker": ticker,
                "topic": "news",
                "title": title or f"{ticker} news event",
                "content": content[:2400],
                "source": "google_news_rss",
                "url": link,
                "published_at": pub_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


def _collect_yahoo_finance_rss(getter: HttpGetter, ticker: str, limit: int) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        xml_text = getter(url, {"Accept": "application/rss+xml", "User-Agent": "Mozilla/5.0"})
    except Exception:  # noqa: BLE001
        return []

    try:
        root = ElementTree.fromstring(xml_text)
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict] = []
    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _normalize_published_at(item.findtext("pubDate") or "")
        desc_html = item.findtext("description") or ""
        desc_plain = _extract_text_from_html(desc_html)
        if not _is_ticker_relevant(ticker, title, desc_plain, link):
            continue
        desc = _summarize_financial_text(desc_plain, 4)
        body = _collect_article_snippet(getter, link, max_sentences=6)
        content = _merge_news_content(desc, body)
        if len(content) < 30:
            continue
        rows.append(
            {
                "ticker": ticker,
                "topic": "news",
                "title": title or f"{ticker} Yahoo headline",
                "content": content[:2400],
                "source": "yahoo_finance_rss",
                "url": link,
                "published_at": pub_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


def _collect_article_snippet(getter: HttpGetter, url: str, max_sentences: int = 6) -> str:
    u = str(url or "").strip()
    if not u:
        return ""

    low_u = u.lower()
    # Google/Yahoo wrapper pages are script-heavy; skip body crawl there.
    if ("news.google.com/" in low_u) or ("finance.yahoo.com/" in low_u):
        return ""

    try:
        html_text = getter(
            u,
            {
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "Mozilla/5.0",
            },
        )
    except Exception:  # noqa: BLE001
        return ""

    text = _extract_text_from_html(html_text)
    if _is_web_noise(text):
        return ""

    return _summarize_financial_text(text, max_sentences=max_sentences)


def _merge_news_content(desc: str, body: str) -> str:
    parts = []
    d = " ".join((desc or "").split()).strip()
    b = " ".join((body or "").split()).strip()
    if d:
        parts.append(d)
    if b and b not in d:
        parts.append(b)
    return " ".join(parts).strip()


def _is_web_noise(text: str) -> bool:
    low = str(text or "").lower()
    markers = [
        "window.yahoo",
        "setprefs",
        "datalayer",
        "consent",
        "googletag",
        "enable",
        "feature",
        "pubads",
        "developer id",
    ]
    hit = sum(1 for m in markers if m in low)
    if hit >= 3:
        return True
    if low.count("window.") >= 2:
        return True
    return False


def _news_queries_for_ticker(ticker: str) -> list[str]:
    symbol = ticker.upper().strip()
    queries = [
        f"{symbol} stock",
        f"{symbol} earnings",
        f"{symbol} outlook",
    ]
    alias = _TICKER_ALIAS_MAP.get(symbol, "")
    if alias:
        queries.append(f"{alias} stock")

    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        out.append(q)
        seen.add(k)
    return out


def _is_ticker_relevant(ticker: str, *texts: str) -> bool:
    symbol = ticker.upper().strip()
    blob = " ".join(str(x or "") for x in texts)
    low = blob.lower()
    if _text_has_symbol(symbol, blob):
        return True

    alias = _TICKER_ALIAS_MAP.get(symbol, "")
    if alias:
        for tok in alias.lower().split():
            if len(tok) >= 4 and tok in low:
                return True

    # Common symbol style such as $GOOGL.
    if f"${symbol.lower()}" in low:
        return True
    return False


def _text_has_symbol(symbol: str, text: str) -> bool:
    if not symbol:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text.upper()) is not None


def _normalize_published_at(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc).isoformat()


def _is_recent_news(published_at: str) -> bool:
    try:
        dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return False
    return dt >= datetime.now(timezone.utc) - timedelta(days=_NEWS_LOOKBACK_DAYS)


def _news_dedup_key(row: dict) -> str:
    ticker = str(row.get("ticker", "")).upper().strip()
    title = str(row.get("title", "")).strip().lower()
    url = str(row.get("url", "")).strip().lower()
    pub = str(row.get("published_at", ""))[:10]
    key = "|".join([ticker, title, url, pub])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _collect_technical_from_yahoo(getter: HttpGetter, ticker: str) -> dict | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=6mo&interval=1d"
    try:
        text = getter(url, {"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    except URLError:
        return None

    try:
        obj = json.loads(text)
        result = obj["chart"]["result"][0]
        closes = [x for x in result["indicators"]["quote"][0]["close"] if x is not None]
        volumes = [x for x in result["indicators"]["quote"][0]["volume"] if x is not None]
    except Exception:  # noqa: BLE001
        return None

    if len(closes) < 80 or len(volumes) < 30:
        return None

    sma20 = sum(closes[-20:]) / 20
    sma60 = sum(closes[-60:]) / 60
    price = closes[-1]
    vol5 = sum(volumes[-5:]) / 5
    vol20 = sum(volumes[-20:]) / 20
    trend = "上行" if sma20 > sma60 else "震荡/下行"
    vol_desc = "放量" if vol5 > vol20 * 1.15 else "量能平稳"

    content = (
        f"近6个月价格最新收盘约为 {price:.2f}，SMA20={sma20:.2f}，SMA60={sma60:.2f}，趋势判断为{trend}。"
        f"近5日均量与20日均量对比显示{vol_desc}（5日均量 {vol5:.0f}，20日均量 {vol20:.0f}）。"
        "若价格持续运行在SMA20与SMA60上方，技术面偏强；若失守SMA60并放量下跌，需提升风控等级。"
    )

    return {
        "ticker": ticker,
        "topic": "technical",
        "title": f"{ticker} technical snapshot from Yahoo Finance",
        "content": content,
        "source": "yahoo_price",
        "url": url,
        "published_at": datetime.now(timezone.utc).date().isoformat(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }




def _collect_technical_from_runtime_tool(ticker: str) -> dict | None:
    try:
        from app.mcp.tools import get_market_snapshot

        m = get_market_snapshot(ticker)
        if m.get("data_status") != "ok" or m.get("price") is None:
            return None
        content = (
            f"最新价={_fmt_num(m.get('price'))}，日高={_fmt_num(m.get('day_high'))}，日低={_fmt_num(m.get('day_low'))}，"
            f"日内涨跌幅={_fmt_num(m.get('change_percent'))}% 。"
            "当前为实时快照模式，若需SMA20/SMA60请补充历史K线源。"
        )
        return {
            "ticker": ticker,
            "topic": "technical",
            "title": f"{ticker} technical snapshot from runtime market tool",
            "content": content,
            "source": "runtime_market_snapshot",
            "url": "runtime://finance_data.market_snapshot",
            "published_at": datetime.now(timezone.utc).date().isoformat(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None


def _collect_fundamental_from_runtime_tool(ticker: str) -> dict | None:
    try:
        from app.mcp.tools import get_fundamental_snapshot

        f = get_fundamental_snapshot(ticker)
        if f.get("data_status") != "ok":
            return None
        parts: list[str] = []
        for label, key, is_pct in [
            ("营收增速", "revenue_growth", True),
            ("利润增速", "earnings_growth", True),
            ("毛利率", "gross_margin", True),
            ("营业利润率", "operating_margin", True),
            ("净利率", "profit_margin", True),
            ("ROE", "return_on_equity", True),
            ("自由现金流", "free_cash_flow", False),
            ("总现金", "total_cash", False),
            ("Trailing PE", "trailing_pe", False),
            ("Forward PE", "forward_pe", False),
            ("P/B", "price_to_book", False),
        ]:
            v = f.get(key)
            if not isinstance(v, (int, float)):
                continue
            if is_pct:
                parts.append(f"{label}={float(v) * 100:.2f}%")
            elif label in {"自由现金流", "总现金"}:
                parts.append(f"{label}={_fmt_large(float(v))}")
            else:
                parts.append(f"{label}={float(v):.2f}")

        if len(parts) < 2:
            return None

        return {
            "ticker": ticker,
            "topic": "fundamental",
            "title": f"{ticker} fundamental snapshot from runtime tool",
            "content": "，".join(parts),
            "source": "runtime_fundamental_snapshot",
            "url": "runtime://finance_data.fundamental_snapshot",
            "published_at": datetime.now(timezone.utc).date().isoformat(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None


def _collect_valuation_from_runtime_tool(ticker: str) -> dict | None:
    try:
        from app.mcp.tools import get_fundamental_snapshot, get_market_snapshot

        m = get_market_snapshot(ticker)
        f = get_fundamental_snapshot(ticker)

        parts: list[str] = []
        for label, val in [
            ("PE", m.get("pe")),
            ("Forward PE", m.get("forward_pe")),
            ("P/B", m.get("pb")),
        ]:
            if isinstance(val, (int, float)):
                parts.append(f"{label}={float(val):.2f}")

        if isinstance(f.get("revenue_growth"), (int, float)):
            parts.append(f"营收增速={float(f.get('revenue_growth')) * 100:.2f}%")
        if isinstance(f.get("earnings_growth"), (int, float)):
            parts.append(f"利润增速={float(f.get('earnings_growth')) * 100:.2f}%")

        if isinstance(m.get("market_cap"), (int, float)):
            parts.append(f"市值={_fmt_large(float(m.get('market_cap')))}")

        if len(parts) < 2:
            return None

        return {
            "ticker": ticker,
            "topic": "valuation",
            "title": f"{ticker} valuation snapshot from runtime tools",
            "content": "，".join(parts),
            "source": "runtime_valuation_snapshot",
            "url": "runtime://finance_data.market_snapshot+fundamental_snapshot",
            "published_at": datetime.now(timezone.utc).date().isoformat(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None


def _fmt_large(v: float) -> str:
    if abs(v) >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    return f"{v:.2f}"


def _fmt_num(v: object) -> str:
    if not isinstance(v, (int, float)):
        return "N/A"
    return f"{float(v):.2f}"

def _collect_fundamental_from_yahoo(getter: HttpGetter, ticker: str) -> dict | None:
    modules = ",".join(["summaryDetail", "defaultKeyStatistics", "financialData", "price"])
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules}"
    try:
        text = getter(url, {"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
        obj = json.loads(text)
        result = obj.get("quoteSummary", {}).get("result", [])
        if not result:
            return None
        item = result[0]
        detail = item.get("summaryDetail", {})
        stat = item.get("defaultKeyStatistics", {})
        fin = item.get("financialData", {})

        def _raw(v: object) -> float | None:
            if isinstance(v, dict):
                v = v.get("raw")
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:  # noqa: BLE001
                return None

        metrics = {
            "营收增速": _raw(fin.get("revenueGrowth")),
            "利润增速": _raw(fin.get("earningsGrowth")),
            "毛利率": _raw(fin.get("grossMargins")),
            "营业利润率": _raw(fin.get("operatingMargins")),
            "净利率": _raw(fin.get("profitMargins")),
            "ROE": _raw(fin.get("returnOnEquity")),
            "自由现金流": _raw(fin.get("freeCashflow")),
            "总现金": _raw(fin.get("totalCash")),
            "Trailing PE": _raw(detail.get("trailingPE")),
            "Forward PE": _raw(detail.get("forwardPE")),
            "P/B": _raw(stat.get("priceToBook")),
        }

        valid_keys = [k for k, v in metrics.items() if v is not None]
        if len(valid_keys) < 2:
            return None

        def _fmt(k: str, v: float) -> str:
            if k in {"自由现金流", "总现金"}:
                if abs(v) >= 1_000_000_000:
                    return f"{v / 1_000_000_000:.2f}B"
                if abs(v) >= 1_000_000:
                    return f"{v / 1_000_000:.2f}M"
                return f"{v:.2f}"
            if "PE" in k or k == "P/B":
                return f"{v:.2f}"
            return f"{v * 100:.2f}%"

        parts = [f"{k}={_fmt(k, metrics[k])}" for k in metrics if metrics[k] is not None]
        content = "，".join(parts)
        return {
            "ticker": ticker,
            "topic": "fundamental",
            "title": f"{ticker} fundamental snapshot from Yahoo Finance",
            "content": content,
            "source": "yahoo_fundamental",
            "url": url,
            "published_at": datetime.now(timezone.utc).date().isoformat(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return None

def _merge_incremental_rows(out_file: Path, new_rows: list[dict], target_tickers: set[str]) -> list[dict]:
    existing: list[dict] = []
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:  # noqa: BLE001
                continue
            existing.append(_enrich_raw_row(obj))

    kept = [x for x in existing if str(x.get("ticker", "")).upper() not in target_tickers]
    combined = kept + new_rows

    dedup: dict[str, dict] = {}
    for row in combined:
        rid = str(row.get("doc_id", "")).strip()
        if not rid:
            key = "|".join([
                str(row.get("ticker", "")),
                str(row.get("topic", "")),
                str(row.get("title", "")),
                str(row.get("published_at", "")),
            ])
            rid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            row["doc_id"] = rid
        dedup[rid] = row

    return sorted(dedup.values(), key=lambda x: (str(x.get("ticker", "")), str(x.get("topic", "")), str(x.get("published_at", ""))))


def _collect_valuation_from_yahoo(getter: HttpGetter, ticker: str) -> dict | None:
    quote_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
    pe = None
    fpe = None
    pb = None
    market_cap = None
    try:
        q_obj = json.loads(getter(quote_url, {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}))
        rows = q_obj.get("quoteResponse", {}).get("result", [])
        if rows:
            item = rows[0]
            pe = item.get("trailingPE")
            fpe = item.get("forwardPE")
            pb = item.get("priceToBook")
            market_cap = item.get("marketCap")
    except Exception:  # noqa: BLE001
        pass

    growth_rev = None
    growth_eps = None
    modules = "financialData"
    sum_url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules}"
    try:
        s_obj = json.loads(getter(sum_url, {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}))
        result = s_obj.get("quoteSummary", {}).get("result", [])
        if result:
            fin = result[0].get("financialData", {})
            growth_rev = _raw_number(fin.get("revenueGrowth"))
            growth_eps = _raw_number(fin.get("earningsGrowth"))
    except Exception:  # noqa: BLE001
        pass

    parts: list[str] = []
    if isinstance(pe, (int, float)):
        parts.append(f"PE={float(pe):.2f}")
    if isinstance(fpe, (int, float)):
        parts.append(f"Forward PE={float(fpe):.2f}")
    if isinstance(pb, (int, float)):
        parts.append(f"P/B={float(pb):.2f}")
    if isinstance(growth_rev, (int, float)):
        parts.append(f"营收增速={float(growth_rev) * 100:.2f}%")
    if isinstance(growth_eps, (int, float)):
        parts.append(f"利润增速={float(growth_eps) * 100:.2f}%")
    if isinstance(market_cap, (int, float)):
        parts.append(f"市值={float(market_cap):.0f}")

    if len(parts) < 2:
        return None

    return {
        "ticker": ticker,
        "topic": "valuation",
        "title": f"{ticker} valuation snapshot from Yahoo Finance",
        "content": "，".join(parts),
        "source": "yahoo_valuation",
        "url": quote_url,
        "published_at": datetime.now(timezone.utc).date().isoformat(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def _raw_number(v: object) -> float | None:
    if isinstance(v, dict):
        v = v.get("raw")
    try:
        if v is None:
            return None
        return float(v)
    except Exception:  # noqa: BLE001
        return None
def _extract_text_from_html(html_text: str) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\\b[a-z]+:[A-Za-z0-9_]+\\b", " ", text)
    text = re.sub(r"\\b\\d{6,}\\b", " ", text)

    tokens = re.findall(r"[A-Za-z0-9.,%$'-]+", text)
    clean_tokens: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        alpha = sum(ch.isalpha() for ch in tok)
        digit = sum(ch.isdigit() for ch in tok)
        if digit > max(3, int(len(tok) * 0.5)):
            continue
        if alpha == 0 and digit > 0:
            continue
        clean_tokens.append(tok)

    cleaned = " ".join(clean_tokens)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned


def _summarize_financial_text(text: str, max_sentences: int = 30) -> str:
    sentences = re.split(r"(?<=[.!?])\\s+", text)
    keep: list[str] = []
    for s in sentences:
        s2 = s.strip()
        if len(s2) < 40:
            continue
        words = s2.split()
        if len(words) < 8:
            continue
        alpha = sum(ch.isalpha() for ch in s2)
        digit = sum(ch.isdigit() for ch in s2)
        ratio_alpha = alpha / max(len(s2), 1)
        ratio_digit = digit / max(len(s2), 1)
        if ratio_alpha < 0.6 or ratio_digit > 0.2:
            continue
        keep.append(s2)
        if len(keep) >= max_sentences:
            break
    if not keep:
        keep = [s.strip() for s in sentences if len(s.strip()) > 40][: max_sentences // 2]
    return " ".join(keep).strip()


def _extract_valuation_paragraphs(text: str) -> str:
    keywords = ["margin", "cash", "guidance", "revenue", "eps", "profit", "growth", "valuation"]
    sentences = re.split(r"(?<=[.!?])\\s+", text)
    picked = [s for s in sentences if any(k in s.lower() for k in keywords)]
    if not picked:
        picked = sentences[:25]
    return " ".join(picked[:60]).strip()[:6000]


def _http_get(url: str, headers: dict[str, str]) -> str:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="ignore")


def _enrich_raw_row(row: dict) -> dict:
    ticker = str(row.get("ticker", "")).upper().strip()
    topic = str(row.get("topic", "general")).lower().strip()
    source = str(row.get("source", "unknown")).strip()
    url = str(row.get("url", "")).strip()
    title = str(row.get("title", "")).strip()
    published_at = str(row.get("published_at", "")).strip()
    updated_at = str(row.get("updated_at") or row.get("collected_at") or published_at).strip()
    content = str(row.get("content", "")).strip()
    normalized = " ".join(content.replace("\n", " ").split())
    content_hash = str(row.get("hash") or row.get("content_hash") or "").strip() or hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    doc_id = str(row.get("doc_id", "")).strip()
    if not doc_id:
        base = "|".join([ticker, topic, source, url, title, published_at])
        suffix = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
        doc_id = f"{ticker}_{topic}_{suffix}" if ticker else suffix

    out = dict(row)
    out.update(
        {
            "ticker": ticker,
            "topic": topic,
            "updated_at": updated_at,
            "doc_id": doc_id,
            "hash": content_hash,
        }
    )
    return out











