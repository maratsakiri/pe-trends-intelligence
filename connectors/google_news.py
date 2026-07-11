import requests
import os
import pandas as pd
import time
import logging
from datetime import datetime, date
import xml.etree.ElementTree as ET
from urllib.parse import quote

logging.basicConfig(
    filename="google_news.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── Search queries for UK financial services PE ────────────────────────────
GOOGLE_NEWS_QUERIES = [
    "private equity acquisition UK financial services",
    "private equity buyout UK fintech",
    "private equity UK insurance acquisition",
    "PE firm acquires UK asset manager",
    "private equity UK wealth management deal",
    "buyout fund UK financial services 2024 2025",
    "private equity portfolio company UK digital transformation",
    "UK fintech private equity investment deal",
]

# ── Core RSS function ──────────────────────────────────────────────────────
def fetch_google_news_rss(query):
    """
    Fetch Google News RSS feed for a specific query.
    No API key needed — completely free.
    """
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-GB&gl=GB&ceid=GB:en"

    try:
        time.sleep(0.5)
        headers = {"User-Agent": "Mozilla/5.0 (research bot - UCL MSc dissertation)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            content = response.content
            content = content.replace(
                b' xmlns:media="http://search.yahoo.com/mrss/"', b""
            )
            return content
        else:
            logging.error(f"Google News error {response.status_code} for: {query}")
            return None

    except Exception as e:
        logging.error(f"Google News request failed: {e}")
        return None

def parse_rss_feed(xml_content, query):
    """
    Parse RSS XML and extract article details.
    Returns list of article dictionaries.
    """
    if not xml_content:
        return []

    articles = []
    try:
        root = ET.fromstring(xml_content)
        items = root.findall(".//item")

        for item in items:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            description = item.findtext("description", "")
            source_elem = item.find("source")
            source = source_elem.text if source_elem is not None else ""

            articles.append({
                "title": title,
                "url": link,
                "published": pub_date[:16] if pub_date else "",
                "source": source,
                "description": description,
                "query_used": query
            })

    except ET.ParseError as e:
        logging.error(f"RSS parse error: {e}")
        print(f"  Parse error: {e}")

    return articles

def is_pe_relevant_news(article):
    """
    Filter articles for PE relevance.
    Checks title and description for PE and financial services terms.
    """
    title = article.get("title", "").lower()
    desc = article.get("description", "").lower()
    text = title + " " + desc

    pe_terms = [
        "private equity", "buyout", "acquisition", "pe firm",
        "portfolio company", "investment fund", "capital partners",
        "venture capital", "asset management", "hedge fund"
    ]

    fs_terms = [
        "financial", "fintech", "insurance", "banking", "payments",
        "asset manager", "wealth", "fund manager", "investment",
        "financial services", "financial technology"
    ]

    has_pe = any(term in text for term in pe_terms)
    has_fs = any(term in text for term in fs_terms)

    return has_pe and has_fs

def check_company_in_google_news(company_name):
    """
    Check if a specific company appears in Google News
    in a PE context. Used for signal triangulation.
    """
    query = f'"{company_name}" private equity OR acquisition OR buyout'
    xml_content = fetch_google_news_rss(query)
    if not xml_content:
        return []
    articles = parse_rss_feed(xml_content, query)
    relevant = [a for a in articles if is_pe_relevant_news(a)]
    return relevant

def run_google_news_pipeline(queries=None):
    """
    Main Google News pipeline function.
    Collects PE-relevant news articles and saves to CSV.
    """
    target_queries = queries or GOOGLE_NEWS_QUERIES
    all_articles = []

    print("=" * 60)
    print("Google News PE Signal Collector")
    print(f"Running {len(target_queries)} search queries")
    print("=" * 60)

    for query in target_queries:
        print(f"\nSearching: '{query}'")
        xml_content = fetch_google_news_rss(query)
        if not xml_content:
            print("  No content returned — skipping")
            continue
        articles = parse_rss_feed(xml_content, query)
        relevant = [a for a in articles if is_pe_relevant_news(a)]
        print(f"  Found {len(articles)} articles. {len(relevant)} PE-relevant after filtering.")
        all_articles.extend(relevant)

    if all_articles:
        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset="url")
        df = df.sort_values("published", ascending=False)
        filename = f"data/google_news_{date.today().isoformat()}.csv"
        df.to_csv(filename, index=False)
        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE")
        print(f"Total unique PE-relevant articles: {len(df)}")
        print(f"Saved to: {filename}")
        print(f"\nSample articles:")
        print(df[["title", "source", "published"]].head(5).to_string(index=False))
    else:
        print("\nNo PE-relevant articles collected")

    return all_articles

def validate_with_google_news(company_name):
    """
    Signal triangulation using Google News.
    More effective than Guardian for mid-market companies.
    """
    print(f"\nValidating: {company_name}")
    articles = check_company_in_google_news(company_name)

    if articles:
        print(f"  ✓ Found in Google News PE context — HIGH CONFIDENCE")
        for a in articles[:2]:
            print(f"    → {a.get('title', '')[:80]}")
            print(f"      Source: {a.get('source', 'unknown')}")
        return True
    else:
        print(f"  ✗ Not found in Google News PE context — LOW CONFIDENCE")
        return False

if __name__ == "__main__":
    run_google_news_pipeline()

    print("\n" + "=" * 60)
    print("SIGNAL TRIANGULATION TEST — Google News")
    print("=" * 60)

    companies_to_validate = [
        "Cantor Finance",
        "Biker Group Holdings",
        "SMH Woodland",
        "Innovation Investment Capital"
    ]

    results = {}
    for company in companies_to_validate:
        results[company] = validate_with_google_news(company)

    print(f"\n{'=' * 60}")
    print("TRIANGULATION SUMMARY")
    for company, validated in results.items():
        status = "✓ HIGH CONFIDENCE" if validated else "✗ LOW CONFIDENCE"
        print(f"  {status}: {company}")