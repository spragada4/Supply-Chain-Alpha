# scraper/edgar_scraper.py

import requests
import pandas as pd
import time
import re
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import SEC_HEADERS, SEC_BASE_URL, COMPANIES, START_YEAR, END_YEAR

# ─────────────────────────────────────────────
# Rate limiter — stay well under SEC's 10 req/sec
# ─────────────────────────────────────────────

def rate_limited_get(url, pause=0.2):
    time.sleep(pause)
    headers = {
        "User-Agent": "{yourname} {provide-mail-id}",
        "Accept-Encoding": "gzip, deflate",
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r


# ─────────────────────────────────────────────
# Step 1: Get all 10-Q filings for a company
# ─────────────────────────────────────────────

def get_10q_filings(cik: str, company_name: str) -> list:
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        r = rate_limited_get(url)
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] Could not fetch filings for {company_name}: {e}")
        return []

    filings    = data.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])
    dates      = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])

    results = []
    for form, date, acc in zip(forms, dates, accessions):
        if form == "10-Q":
            results.append({
                "company":   company_name,
                "cik":       cik,
                "date":      date,
                "accession": acc,  # keep original format with dashes
            })

    print(f"  Found {len(results)} 10-Q filings for {company_name}")
    return results


# ─────────────────────────────────────────────
# Step 2: Parse filing index page to get
# the primary document URL (always Type 1)
# ─────────────────────────────────────────────

def get_primary_doc_url(cik: str, accession: str):
    cik_int   = int(cik)
    acc_clean = accession.replace("-", "")

    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_clean}/{accession}-index.htm"
    )

    try:
        r    = rate_limited_get(index_url)
        soup = BeautifulSoup(r.text, "html.parser")

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                doc_type = cells[0].get_text(strip=True)
                link     = cells[2].find("a")

                if doc_type == "1" and link:
                    href = link["href"]
                    # Strip iXBRL viewer prefix if present
                    href = re.sub(r"^/ix\?doc=", "", href)
                    # Make absolute URL
                    if href.startswith("/"):
                        href = f"https://www.sec.gov{href}"
                    return href

    except Exception as e:
        print(f"    [ERROR] Index fetch failed: {e}")

    return None


# ─────────────────────────────────────────────
# Step 3: Extract MD&A section from HTML
# ─────────────────────────────────────────────

def extract_mda(html_text: str):
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", html_text)
    clean = re.sub(r"\s+", " ", clean).strip()

    # MD&A section start patterns
    mda_patterns = [
        r"(?i)item\s*2[\.\s]*management.{0,50}discussion",
        r"(?i)item\s*2[\.\s]*md&a",
        r"(?i)management.{0,30}discussion.{0,30}analysis\s+of",
    ]

    # Next section patterns (where MD&A ends)
    end_patterns = [
        r"(?i)item\s*3[\.\s]*quantitative",
        r"(?i)item\s*3[\.\s]*market risk",
        r"(?i)item\s*4[\.\s]*controls",
        r"(?i)item\s*4[\.\s]*mine safety",
    ]

    mda_start = None
    for pattern in mda_patterns:
        match = re.search(pattern, clean)
        if match:
            mda_start = match.start()
            break

    if mda_start is None:
        return None

    mda_end = len(clean)
    for pattern in end_patterns:
        match = re.search(pattern, clean[mda_start + 200:])
        if match:
            mda_end = mda_start + 200 + match.start()
            break

    mda_text = clean[mda_start:mda_end].strip()

    if len(mda_text) < 500:
        return None

    return mda_text[:15000]


# ─────────────────────────────────────────────
# Step 4: Master scraper
# ─────────────────────────────────────────────

def scrape_all_companies(
    companies:   dict,
    start_year:  int = 2019,
    end_year:    int = 2024,
    output_path: str = "data/mda_raw.csv"
) -> pd.DataFrame:

    all_records = []

    for company_name, cik in companies.items():
        print(f"\n{'='*50}")
        print(f"Processing: {company_name} (CIK: {cik})")
        print(f"{'='*50}")

        filings = get_10q_filings(cik, company_name)

        # Filter to date range
        filings = [
            f for f in filings
            if start_year <= int(f["date"][:4]) <= end_year
        ]
        print(f"  Filtered to {len(filings)} filings ({start_year}–{end_year})")

        for filing in tqdm(filings, desc=f"  {company_name}"):
            try:
                doc_url = get_primary_doc_url(cik, filing["accession"])
                if not doc_url:
                    continue

                r        = rate_limited_get(doc_url, pause=0.25)
                mda_text = extract_mda(r.text)

                if mda_text:
                    all_records.append({
                        "company":    company_name,
                        "cik":        cik,
                        "date":       filing["date"],
                        "accession":  filing["accession"],
                        "doc_url":    doc_url,
                        "mda_text":   mda_text,
                        "mda_length": len(mda_text),
                    })
                else:
                    print(f"    [SKIP] {filing['date']} — MD&A not found in doc")

            except Exception as e:
                print(f"    [SKIP] {filing['date']}: {e}")
                continue

    df = pd.DataFrame(all_records)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["company", "date"]).reset_index(drop=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} MD&A sections to {output_path}")
        print(df.groupby("company")["date"].count().to_string())
    else:
        print("\n⚠️  No data collected")

    return df


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = scrape_all_companies(
        companies   = COMPANIES,
        start_year  = START_YEAR,
        end_year    = END_YEAR,
        output_path = "data/mda_raw.csv"
    )
