# Affordable Care Act Marketplace Plan Premium and Cost-Sharing Variation Analysis

This repository contains a reproducible analysis of 2026 Affordable Care Act Marketplace plan availability, gross premiums, deductible exposure, maximum out-of-pocket fields, and issuer choice.

## Report

- [Open the rendered web report](https://shiga-1993.github.io/affordable-care-act-marketplace-plan-analysis/)

Opening `index.html` directly on GitHub shows the source. Use the GitHub Pages link above to view the rendered report.

## Analytical Question

How do Affordable Care Act Marketplace plan choices differ by state and metal level when premium, deductible, maximum out-of-pocket, issuer count, and plan availability are viewed together?

The report focuses on plan availability and plan design. It is not an enrollment analysis, subsidy analysis, plan recommendation, medical advice, legal advice, compliance advice, or insurance-purchasing advice.

## Main Takeaways

- Metal level is useful, but it is not a simple price ladder.
- Bronze, Expanded Bronze, Silver, and Gold premium distributions overlap.
- Lower-premium tiers generally shift more exposure into deductibles.
- Plan menus differ sharply by state: some states have many issuers and plans, while others have thin choice.
- Lowest-cost Silver premiums vary widely across states even before subsidies are considered.
- A consumer-facing plan menu has at least three dimensions: monthly premium, potential cost-sharing exposure, and how many issuers/plans are available in the rating area.

## Data Source

- Source: Centers for Medicare & Medicaid Services Health Insurance Exchange Public Use Files
- Source page: https://www.cms.gov/marketplace/resources/data/public-use-files
- Plan year: 2026
- Source checked: 2026-05-25
- Source note: the Centers for Medicare & Medicaid Services page states that 2026 Exchange Public Use File data were last imported by April 28, 2026.
- Files used:
  - Plan Attributes Public Use File
  - Rate Public Use File
  - Benefits and Cost Sharing Public Use File

## Method

The pipeline downloads the public source zip files, filters to individual-market, non-dental, standard on-exchange medical plan variants, joins age-40 gross premium rows by plan and rating area, parses selected cost-sharing fields, builds summary tables, creates figures, and renders a standalone web report.

The heavy raw source zip files are not committed to this repository.

## Repository Contents

- `index.html`: rendered GitHub Pages report
- `run_marketplace_plan_analysis.py`: reproducible analysis script
- `outputs/figures/`: generated report figures
- `outputs/tables/`: aggregate summary tables and run metadata
- `data/README.md`: data handling notes
- `tests/`: focused parsing tests

## Reproduce

From this repository:

```bash
python3 -m pip install --user -r requirements.txt
python3 run_marketplace_plan_analysis.py
python3 -m pytest -q
```

The script writes report outputs under `outputs/` and removes the raw downloaded zip files unless `--keep-raw` is passed.

## Limitations

- This is plan availability data, not enrollment data.
- Premiums are gross premiums, not subsidy-adjusted net premiums.
- Rows are not weighted by county population, enrollment, household income, morbidity, or issuer market share.
- Some State-based Exchanges are outside this federal Exchange Public Use File scope when they do not rely on the federal platform.
- Deductible, maximum out-of-pocket, and copay values are simplified plan-design fields.
