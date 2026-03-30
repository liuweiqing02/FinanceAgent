from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree


HttpGetter = Callable[[str, dict[str, str]], str]


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
    for ticker in [t.upper() for t in tickers if t.strip()]:
        cik = cik_map.get(ticker)
        if cik:
            filing_rows = _collect_sec_filings(getter, ticker, cik, sec_user_agent, filing_limit)
            rows.extend(filing_rows)
            filing_count += len(filing_rows)

        rss_rows = _collect_news_rss(getter, ticker, news_limit)
        rows.extend(rss_rows)
        news_count += len(rss_rows)

        tech = _collect_technical_from_yahoo(getter, ticker)
        if tech is not None:
            rows.append(tech)
            technical_count += 1

    dedup = {}
    for row in rows:
        key = (row.get("ticker"), row.get("topic"), row.get("title"), row.get("published_at"))
        dedup[key] = row

    with out_file.open("w", encoding="utf-8") as f:
        for row in dedup.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return RealCollectStats(
        tickers=[t.upper() for t in tickers],
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


def _collect_news_rss(getter: HttpGetter, ticker: str, news_limit: int) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        xml_text = getter(url, {"Accept": "application/rss+xml"})
    except URLError:
        return []

    root = ElementTree.fromstring(xml_text)
    rows: list[dict] = []
    for item in root.findall("./channel/item")[:news_limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        desc = _summarize_financial_text(_extract_text_from_html(item.findtext("description") or ""), 5)
        if not title or not desc:
            continue
        rows.append(
            {
                "ticker": ticker,
                "topic": "news",
                "title": title,
                "content": desc[:1500],
                "source": "google_news_rss",
                "url": link,
                "published_at": pub_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


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
