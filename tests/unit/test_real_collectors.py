import json
from pathlib import Path

from app.rag.real_collectors import collect_real_raw_data


def test_collect_real_raw_data_with_mock_http(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    cik_json = json.dumps(
        {
            "0": {"ticker": "AAPL", "cik_str": 320193},
        }
    )
    sub_json = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K"],
                    "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
                    "primaryDocument": ["aapl10q.htm", "aapl8k.htm"],
                    "filingDate": ["2026-02-01", "2026-02-10"],
                }
            }
        }
    )
    long_body = " ".join(
        [
            "Revenue and margin improved with strong cash flow and guidance updates."
            for _ in range(30)
        ]
    )
    filing_html = f"<html><body><h1>{long_body}</h1><p>{long_body}</p></body></html>"
    rss_xml = """<?xml version='1.0'?><rss><channel>
    <item><title>AAPL rises after report</title><link>https://news.example/aapl1</link><pubDate>Mon, 02 Mar 2026 00:00:00 GMT</pubDate><description><![CDATA[<p>Apple stock moved higher after earnings surprise.</p>]]></description></item>
    </channel></rss>"""
    yahoo_json = json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "indicators": {
                            "quote": [
                                {
                                    "close": [100 + i * 0.2 for i in range(130)],
                                    "volume": [1000000 + i * 1000 for i in range(130)],
                                }
                            ]
                        }
                    }
                ]
            }
        }
    )

    def fake_get(url: str, headers: dict[str, str]) -> str:  # noqa: ARG001
        if url.endswith("company_tickers.json"):
            return cik_json
        if "submissions/CIK0000320193.json" in url:
            return sub_json
        if "aapl10q.htm" in url or "aapl8k.htm" in url:
            return filing_html
        if "news.google.com/rss/search" in url:
            return rss_xml
        if "query1.finance.yahoo.com/v8/finance/chart/AAPL" in url:
            return yahoo_json
        raise AssertionError(f"unexpected url: {url}")

    stats = collect_real_raw_data(
        raw_dir=raw_dir,
        tickers=["AAPL"],
        sec_user_agent="FinanceAgent/1.0 test@example.com",
        filing_limit=4,
        news_limit=3,
        http_get=fake_get,
    )

    assert stats.filings >= 2
    assert stats.news >= 1
    assert stats.technical >= 1
    assert stats.output_file.exists()

    rows = [json.loads(line) for line in stats.output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r["topic"] == "fundamental" for r in rows)
    assert any(r["topic"] == "valuation" for r in rows)
    assert any(r["topic"] == "news" for r in rows)
    assert any(r["topic"] == "technical" for r in rows)
    assert any("Revenue" in r["content"] or "stock" in r["content"] for r in rows)
