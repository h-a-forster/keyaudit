# Upstream reports and outcomes

Reports filed from keyaudit's findings. State read from GitHub on 2026-11-13.

| Project | Issue | Filed | State | Outcome |
|---|---|---|---|---|
| BIG-Bench Hard | [suzgunmirac/BIG-Bench-Hard#20](https://github.com/suzgunmirac/BIG-Bench-Hard/issues/20): geometric_shapes and date_understanding keys | 2026-11-09 | open | no response yet |

The issue comes from the E6 study (REPORT.md 3.8): 50 geometric_shapes items keyed
"trapezoid" or "ellipse" where "rectangle" or "circle" is also offered and more specific, and
20 wrong date_understanding keys, 13 of them with no correct option. The date groups were
already reported upstream in BIG-bench #926 and #944 (2023, open); the issue adds the BBH
item numbers.

Not filed: ARC-Challenge (2 wrong keys and 6 other defects in a sample of 200; low stakes per
item) and TruthfulQA (contested or dated items only).
