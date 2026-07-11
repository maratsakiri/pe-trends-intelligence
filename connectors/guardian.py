import requests
import os
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, date

# ── Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
BASE_URL = "https://content.guardianapis.com"

logging.basicConfig(
    filename="guardian.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── More specific PE search queries ────────────────────────────────────────
PE_SEARCH_QUERIES = [
    "private equity buyout fintech UK acquisition",
    "private equity acquires insurance company UK",
    "private equity buyout asset manager UK",
    "PE firm acquires UK financial services",
    "private equity portfolio digital transformation financial services",
    "buyout EQT Blackstone KKR Bridgepoint UK financial",
    "private equity investment fintech payments UK deal",
    "hedge fund acquisition UK financial services company",
    "private equity backed financial services UK growth",
    "PE acquisition UK wealth management insurance fintech",
]

# ── Core API function ──────────────────────────────────────────────────────
def search_guardian(query, from_date="2021-01-01", page_size=50, section="business"):
    """Search Guardian API for articles matching a query."""
    url = f"{BASE_URL}/search"
    params = {
        "q": query,
        "from-date": from_date,
        "page-size": page_size,
        "order-by": "relevance",  # Changed to relevance not newest
        "show-fields": "bodyText,trailText",
        "api-key": GUARDIAN_API_KEY,
        "section": section
    }

    try:
        time.sleep(0.3)
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Guardian API error {response.status_code} for: {query}")
            return None
    except Exception as e:
        logging.error(f"Guardian request failed: {e}")
        return None

def is_pe_relevant(article):
    """
    Check if an article is genuinely about PE activity.
    Filters out general business news that matched by accident.
    """
    title = article.get("webTitle", "").lower()
    trail = article.get("fields", {}).get("trailText", "").lower()
    body = article.get("fields", {}).get("bodyText", "").lower()
    text = title + " " + trail + " " + body

    # Must contain PE-specific terms
    pe_terms = [
        "private equity", "buyout", "acquisition", "pe firm",
        "portfolio company", "venture capital", "investment fund",
        "asset management", "capital partners", "equity fund"
    ]

    # Must contain UK financial services terms
    fs_terms = [
        "financial services", "fintech", "insurance", "asset manager",
        "fund manager", "payments", "banking", "financial technology",
        "wealth management", "investment management"
    ]

    has_pe = any(term in text for term in pe_terms)
    has_fs = any(term in text for term in fs_terms)

    return has_pe and has_fs

def check_company_in_news(company_name, from_date="2021-01-01"):
    """
    Check if a specific company appears in Guardian financial news
    AND the article is genuinely PE-related.
    Used for signal triangulation.
    """
    # Search specifically for the company name
    results = search_guardian(
        query=f'"{company_name}"',  # Exact phrase match
        from_date=from_date,
        page_size=10,
        section="business"
    )

    if not results:
        return []

    articles = results.get("response", {}).get("results", [])

    # Filter for PE relevance
    relevant = [a for a in articles if is_pe_relevant(a)]
    return relevant

def run_guardian_pipeline(queries=None, from_date="2021-01-01"):
    """
    Main Guardian pipeline function.
    Collects and filters PE-relevant articles.
    """
    target_queries = queries or PE_SEARCH_QUERIES
    all_articles = []

    print("=" * 60)
    print("Guardian News PE Signal Collector")
    print(f"Collecting articles from {from_date} onwards")
    print(f"Running {len(target_queries)} search queries")
    print("=" * 60)

    for query in target_queries:
        print(f"\nSearching: '{query}'")
        results = search_guardian(query, from_date=from_date)

        if not results:
            print("  No results or error — skipping")
            continue

        articles = results.get("response", {}).get("results", [])
        total = results.get("response", {}).get("total", 0)

        # Filter for PE relevance
        relevant = [a for a in articles if is_pe_relevant(a)]
        print(f"  Found {total} total. {len(relevant)} PE-relevant after filtering.")

        for article in relevant:
            fields = article.get("fields", {})
            all_articles.append({
                "article_id": article.get("id"),
                "title": article.get("webTitle"),
                "date": article.get("webPublicationDate", "")[:10],
                "url": article.get("webUrl"),
                "section": article.get("sectionName"),
                "trail_text": fields.get("trailText", ""),
                "body_text": fields.get("bodyText", "")[:2000],
                "query_used": query,
                "collected_at": datetime.now().isoformat()
            })

    if all_articles:
        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset="article_id")
        df = df.sort_values("date", ascending=False)
        filename = f"data/guardian_articles_{date.today().isoformat()}.csv"
        df.to_csv(filename, index=False)
        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE")
        print(f"Total unique PE-relevant articles: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Saved to: {filename}")
        print(f"\nSample articles:")
        print(df[["title", "date"]].head(5).to_string(index=False))
    else:
        print("\nNo PE-relevant articles collected")

    return all_articles

def validate_companies_house_detection(company_name):
    """
    Signal triangulation — checks if a Companies House
    detection also appears in PE-relevant Guardian news.
    """
    print(f"\nValidating: {company_name}")
    articles = check_company_in_news(company_name)

    if articles:
        print(f"  ✓ Found in PE-relevant Guardian news — HIGH CONFIDENCE")
        for a in articles[:2]:
            print(f"    → {a.get('webTitle', '')[:80]}")
            print(f"      {a.get('webPublicationDate', '')[:10]}")
        return True
    else:
        print(f"  ✗ Not found in Guardian PE news — LOW CONFIDENCE")
        print(f"    Manual verification recommended")
        return False

if __name__ == "__main__":
    run_guardian_pipeline()

    print("\n" + "=" * 60)
    print("SIGNAL TRIANGULATION TEST")
    print("=" * 60)

    companies_to_validate = [
        "Cantor Finance",
        "Biker Group Holdings",
        "SMH Woodland",
        "Innovation Investment Capital"
    ]

    results = {}
    for company in companies_to_validate:
        results[company] = validate_companies_house_detection(company)

    print(f"\n{'=' * 60}")
    print("TRIANGULATION SUMMARY")
    for company, validated in results.items():
        status = "✓ HIGH CONFIDENCE" if validated else "✗ LOW CONFIDENCE"
        print(f"  {status}: {company}")