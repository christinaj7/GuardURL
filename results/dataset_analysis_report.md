# GuardURL — Phase 1: Dataset Analysis Report

## 1. Overview
The dataset provided is the **PhiUSIIL Phishing URL Dataset** (`data/PhiUSIIL_Phishing_URL_Dataset.csv`), a widely recognized cybersecurity benchmark dataset consisting of real-world benign and phishing URLs.

## 2. Key Dataset Dimensions
- **Total Records (Rows):** 235,795
- **Total Attributes (Columns):** 56
- **Dataset File Size:** ~56.8 MB
- **Target Column:** `label`
- **Primary Identifier/URL Column:** `URL`
- **Metadata Column:** `FILENAME`

## 3. Data Integrity & Quality
- **Missing / Null Values:** **0** (No missing values in any of the 56 columns)
- **Full Row Duplicates:** **0**
- **Duplicate URLs:** **425** (0.18% of the dataset)
  - Duplicate URLs will be deduplicated during preprocessing to prevent data leakage.

## 4. Target Label Distribution
The PhiUSIIL dataset uses the following ground truth convention:
- **`1` = Legitimate (Benign)**
- **`0` = Phishing (Malicious)**

| Class | Label Value | Count | Proportion |
| :--- | :---: | :---: | :---: |
| **Legitimate** | `1` | 134,850 | **57.19%** |
| **Phishing** | `0` | 100,945 | **42.81%** |
| **Total** | — | 235,795 | **100.0%** |

### Class Imbalance Assessment:
- **Ratio (Majority / Minority):** 1.34 : 1
- **Status:** **Well-balanced**. No extreme sampling techniques (e.g. heavy oversampling) are mandatory, though stratified splitting is recommended to maintain exact class proportions across train and test sets.
- **Safety Directive:** Label orientation is strictly preserved (`1` = Legitimate, `0` = Phishing).

## 5. Feature Classification & Granularity
The dataset includes two distinct groups of attributes:
1. **URL Structural & Lexical Attributes (Columns 3–26):**
   - URL length, domain length, TLD, subdomain count, letter count, digit count, special characters (`=`, `?`, `&`, other), obfuscation, HTTPS usage, etc.
   - Computable directly from any URL string in real-time.
2. **Webpage / Content Features (Columns 27–55):**
   - `LineOfCode`, `LargestLineLength`, `Title`, `DomainTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `HasSubmitButton`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, etc.
   - These features represent crawler-scraped webpage DOM properties.

## 6. Preprocessing & Extraction Strategy
To ensure that GuardURL can analyze any user-entered URL safely and instantaneously without requiring unsafe live scraping of malicious domains:
- GuardURL builds a comprehensive URL feature extraction engine capable of extracting lexical, structural, statistical, and security features from any URL string.
- The feature extractor is verified on the training dataset and used identically at inference time.
