"""
GuardURL - Phase 3: Feature Engineering & URL Extraction Engine
Extracts lexical, structural, statistical, and security features from raw URLs.
Works identically for both offline training and real-time user-entered URLs.
Applies domain and path normalization to prevent dataset shortcut bias.
"""

import re
import math
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np

# Keywords frequently associated with phishing lures, credential theft, and spoofing
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "verification", "update", "banking", "secure", "security",
    "account", "signin", "sign-in", "paypal", "apple", "wallet", "confirm",
    "confirmation", "support", "free", "bonus", "billing", "ebay", "amazon",
    "credential", "auth", "authorize", "passcode", "password", "recovery",
    "alert", "checkout", "netflix", "microsoft", "google", "outlook", "bank",
    "invoice", "token", "validate", "suspended", "security-alert"
]

COMMON_TLDS = [
    ".com", ".org", ".net", ".edu", ".gov", ".mil", ".io", ".co",
    ".info", ".biz", ".top", ".xyz", ".ru", ".cn", ".de", ".uk"
]

# Baseline empirical TLD legitimate probabilities calculated from PhiUSIIL reference
DEFAULT_TLD_PROB_MAP = {
    "com": 0.53, "org": 0.91, "net": 0.58, "edu": 0.99, "gov": 0.99,
    "uk": 0.88, "de": 0.84, "jp": 0.89, "au": 0.86, "ca": 0.85,
    "fr": 0.82, "it": 0.81, "app": 0.45, "io": 0.65, "co": 0.52,
    "ru": 0.35, "xyz": 0.12, "top": 0.08, "gq": 0.04, "cf": 0.04,
    "tk": 0.03, "ml": 0.04, "ga": 0.04, "fit": 0.15, "club": 0.22,
    "online": 0.30, "site": 0.25, "buzz": 0.10, "work": 0.18
}

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "is_https",
    "is_ip",
    "no_of_dots",
    "no_of_hyphens",
    "no_of_at",
    "no_of_question_marks",
    "no_of_equals",
    "no_of_ampersands",
    "path_depth",
    "no_of_subdomains",
    "tld_length",
    "tld_legitimate_prob",
    "no_of_letters",
    "letter_ratio",
    "no_of_digits",
    "digit_ratio",
    "no_of_special_chars",
    "special_char_ratio",
    "has_obfuscation",
    "no_of_obfuscated_char",
    "obfuscation_ratio",
    "no_of_parameters",
    "has_fragment",
    "url_entropy",
    "suspicious_keyword_count",
    "has_suspicious_keyword",
    "has_punycode",
    "has_port",
    "has_double_slash_redirect",
    "tld_in_path",
    "char_continuation_rate",
    "url_char_prob"
]


def calculate_entropy(text: str) -> float:
    """Calculates Shannon entropy of a string (higher = more random/obfuscated)."""
    if not text:
        return 0.0
    text_len = len(text)
    prob = [float(text.count(c)) / text_len for c in dict.fromkeys(list(text))]
    return -sum([p * math.log2(p) for p in prob if p > 0])


def calculate_char_continuation_rate(domain: str) -> float:
    """
    Calculates the ratio of the longest uninterrupted alphabetic sequence
    in the domain to the domain character length.
    Normal domains produce high continuation rates (~0.6 - 1.0).
    Hyphenated/fragmented phishing lures produce low continuation rates (<0.3).
    """
    if not domain:
        return 0.0
    letters_seq = re.findall(r"[a-zA-Z]+", domain)
    if not letters_seq:
        return 0.0
    longest = max(len(s) for s in letters_seq)
    clean_len = len(domain.replace(".", ""))
    return round(float(longest) / max(clean_len, 1), 4)


def calculate_char_prob(text: str) -> float:
    """
    Approximates character unigram probability distribution score.
    Benign URLs have standard alphanumeric distributions; phishing URLs skew toward specials/hex.
    """
    if not text:
        return 0.0
    text_len = len(text)
    alpha_cnt = sum(c.isalpha() for c in text)
    num_cnt = sum(c.isdigit() for c in text)
    special_cnt = text_len - (alpha_cnt + num_cnt)
    score = (alpha_cnt * 0.7 + num_cnt * 0.2 + special_cnt * 0.1) / text_len
    return round(score, 4)


def canonicalize_url(url_str: str) -> str:
    """
    Normalizes apex domains to canonical web host aliases to prevent dataset
    sampling distortion between www and non-www root domains.
    """
    url_clean = str(url_str).strip()
    if not url_clean:
        return url_clean
    if "://" not in url_clean:
        url_clean = ("https:" if url_clean.startswith("//") else "http://") + url_clean
    try:
        parsed = urlparse(url_clean)
        host = parsed.netloc.lower().split(":")[0]
        # If domain is an apex domain (e.g. 'example.com', exactly one dot, not an IP)
        if host and host.count(".") == 1 and not host.startswith("www.") and not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
            new_netloc = "www." + parsed.netloc
            url_clean = parsed._replace(netloc=new_netloc).geturl()
    except Exception:
        pass
    return url_clean


def extract_url_features(url: str, tld_prob_map: dict = None) -> dict:
    """
    Extracts all lexical, structural, and statistical features from a single raw URL string.
    Applies domain normalization (www alias handling) and structural parsing.
    Returns a dictionary mapping feature names to numerical values.
    """
    if tld_prob_map is None:
        tld_prob_map = DEFAULT_TLD_PROB_MAP

    raw_input = str(url).strip()
    if not raw_input:
        return {feat: 0.0 for feat in FEATURE_NAMES}

    url_str = canonicalize_url(raw_input)

    # Normalize protocol prefix for safe parsing
    full_url = url_str if "://" in url_str else ("https://" if url_str.startswith("//") else "http://" + url_str)

    try:
        parsed = urlparse(full_url)
    except Exception:
        parsed = urlparse("http://" + url_str)

    scheme = parsed.scheme.lower() if parsed.scheme else ("https" if url_str.lower().startswith("https") else "http")
    is_https = 1 if scheme == "https" else 0

    raw_netloc = parsed.netloc.lower()
    if not raw_netloc and parsed.path and "/" not in parsed.path:
        raw_netloc = parsed.path.lower()
        path = ""
    else:
        path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    # Clean port if present
    clean_host = raw_netloc.split(":")[0]

    # Domain normalization: treat www. as standard prefix so www.domain.com and domain.com
    # produce consistent domain metrics without artificial subdomain distortion
    clean_domain = clean_host[4:] if clean_host.startswith("www.") else clean_host
    domain_len = len(clean_domain)

    # IP detection (IPv4 pattern)
    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", clean_domain) else 0

    # Subdomains and TLD parsing on normalized domain
    domain_parts = [p for p in clean_domain.split(".") if p]
    if is_ip or len(domain_parts) <= 1:
        no_of_subdomains = 0
        tld = ""
    else:
        # e.g., 'sub.example.com' -> subdomains = 1
        no_of_subdomains = max(0, len(domain_parts) - 2)
        tld = domain_parts[-1]

    tld_len = len(tld)
    tld_legitimate_prob = float(tld_prob_map.get(tld, 0.50))

    # Path depth
    path_clean = path.strip("/")
    path_dirs = [d for d in path_clean.split("/") if d]
    path_depth = len(path_dirs)

    # Character distributions
    url_len = len(url_str.rstrip("/"))
    letters = sum(c.isalpha() for c in url_str)
    digits = sum(c.isdigit() for c in url_str)
    # Special characters: exclude standard protocol delimiters '://' so normal URLs aren't penalized
    specials = sum(not c.isalnum() and c not in [":", "/"] for c in url_str)
    safe_len = max(url_len, 1)

    # Obfuscation (hex encoded characters like %20, %2F)
    hex_obfuscations = len(re.findall(r"%[0-9a-fA-F]{2}", url_str))
    has_obfuscation = 1 if hex_obfuscations > 0 or "@" in url_str else 0
    obfuscation_ratio = float(hex_obfuscations * 3) / safe_len

    # Parameter extraction
    try:
        params_dict = parse_qs(query)
        no_of_parameters = len(params_dict)
    except Exception:
        no_of_parameters = query.count("=") if query else 0

    # Suspicious keywords
    url_lower = url_str.lower()
    matched_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    suspicious_keyword_count = len(matched_keywords)
    has_suspicious_keyword = 1 if suspicious_keyword_count > 0 else 0

    # Advanced phishing flags
    has_punycode = 1 if "xn--" in clean_domain else 0
    has_port = 1 if ":" in raw_netloc and not is_ip else 0
    has_double_slash_redirect = 1 if "//" in path else 0

    # Check if TLD appears fraudulently inside the path or query
    tld_in_path = 1 if any(tld_item in path.lower() or tld_item in query.lower() for tld_item in COMMON_TLDS) else 0

    # Entropy and continuation rate
    entropy = calculate_entropy(url_str)
    continuation_rate = calculate_char_continuation_rate(clean_domain)
    char_prob = calculate_char_prob(url_str)

    features = {
        "url_length": float(url_len),
        "domain_length": float(domain_len),
        "is_https": float(is_https),
        "is_ip": float(is_ip),
        "no_of_dots": float(clean_domain.count(".")),
        "no_of_hyphens": float(url_str.count("-")),
        "no_of_at": float(url_str.count("@")),
        "no_of_question_marks": float(url_str.count("?")),
        "no_of_equals": float(url_str.count("=")),
        "no_of_ampersands": float(url_str.count("&")),
        "path_depth": float(path_depth),
        "no_of_subdomains": float(no_of_subdomains),
        "tld_length": float(tld_len),
        "tld_legitimate_prob": float(tld_legitimate_prob),
        "no_of_letters": float(letters),
        "letter_ratio": float(letters / safe_len),
        "no_of_digits": float(digits),
        "digit_ratio": float(digits / safe_len),
        "no_of_special_chars": float(specials),
        "special_char_ratio": float(specials / safe_len),
        "has_obfuscation": float(has_obfuscation),
        "no_of_obfuscated_char": float(hex_obfuscations),
        "obfuscation_ratio": float(obfuscation_ratio),
        "no_of_parameters": float(no_of_parameters),
        "has_fragment": 1.0 if fragment else 0.0,
        "url_entropy": float(entropy),
        "suspicious_keyword_count": float(suspicious_keyword_count),
        "has_suspicious_keyword": float(has_suspicious_keyword),
        "has_punycode": float(has_punycode),
        "has_port": float(has_port),
        "has_double_slash_redirect": float(has_double_slash_redirect),
        "tld_in_path": float(tld_in_path),
        "char_continuation_rate": float(continuation_rate),
        "url_char_prob": float(char_prob)
    }

    return features


def extract_features_batch(urls: list, tld_prob_map: dict = None) -> pd.DataFrame:
    """
    Extracts features for a collection of URLs.
    Returns a pandas DataFrame indexed identically to input URLs.
    """
    records = [extract_url_features(u, tld_prob_map) for u in urls]
    return pd.DataFrame(records, columns=FEATURE_NAMES)


def build_tld_probability_map(df: pd.DataFrame, url_col: str = "URL", label_col: str = "label") -> dict:
    """
    Builds an empirical TLD legitimate probability lookup table from training data.
    `1` = Legitimate, `0` = Phishing.
    """
    tld_stats = {}
    for _, row in df.iterrows():
        try:
            u = str(row[url_col])
            parsed = urlparse(u if "://" in u else "http://" + u)
            host = parsed.netloc.split(":")[0].lower()
            clean_host = host[4:] if host.startswith("www.") else host
            parts = clean_host.split(".")
            if len(parts) > 1 and not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", clean_host):
                tld = parts[-1].lower()
                lbl = int(row[label_col])
                if tld not in tld_stats:
                    tld_stats[tld] = {"legit": 0, "total": 0}
                tld_stats[tld]["total"] += 1
                if lbl == 1:
                    tld_stats[tld]["legit"] += 1
        except Exception:
            continue

    prob_map = {}
    for tld, stats in tld_stats.items():
        if stats["total"] >= 5:
            prob_map[tld] = round(stats["legit"] / stats["total"], 4)

    return prob_map
