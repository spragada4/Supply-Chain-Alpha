# config.py

# SEC EDGAR requires you identify yourself
SEC_HEADERS = {
    "User-Agent": "SriPragada sripragada4@gmail.com",  # ← change this
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

SEC_BASE_URL = "https://data.sec.gov"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/EFTS/hits.json"

# Companies to track — mix of manufacturers & retailers
# Format: "Company Name": "CIK number"
# CIK = unique company ID on EDGAR
COMPANIES = {
    "Apple":        "0000320193",
    "Nike":         "0000320187",
    "Ford":         "0000037996",
    "General Motors": "0001467858",
    "Target":       "0000027419",
    "Walmart":      "0000104169",
    "FedEx":        "0001048911",
    "UPS":          "0001090727",
    "Caterpillar":  "0000018230",
    "3M":           "0000066740",
}

# Date range
START_YEAR = 2019  # captures pre-COVID baseline
END_YEAR   = 2024

# Supply chain keywords to track in filings
SUPPLY_CHAIN_KEYWORDS = [
    "supply chain", "supplier", "inventory", "logistics",
    "freight", "shipping", "port", "congestion", "shortage",
    "backlog", "delay", "disruption", "sourcing", "procurement"
]