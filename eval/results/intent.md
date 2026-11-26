# Intent router evaluation

54 examples · accuracy **1.000** · macro-F1 **1.000** · preference accuracy 1.000

Latency p50 1.48s · p95 5.75s · LLM calls 54 · Groq fallbacks 0 · errors 0

## Per intent

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| GENERAL_CHAT | 1.000 | 1.000 | 1.000 | 9 |
| RAILWAY_RULES | 1.000 | 1.000 | 1.000 | 14 |
| STATION_SEARCH | 1.000 | 1.000 | 1.000 | 6 |
| TRAIN_DETAILS | 1.000 | 1.000 | 1.000 | 12 |
| TRAIN_SEARCH | 1.000 | 1.000 | 1.000 | 13 |

## Confusion matrix

| gold \ predicted | GENERAL_CHAT | RAILWAY_RULES | STATION_SEARCH | TRAIN_DETAILS | TRAIN_SEARCH |
|---|---|---|---|---|---|
| GENERAL_CHAT | 9 | 0 | 0 | 0 | 0 |
| RAILWAY_RULES | 0 | 14 | 0 | 0 | 0 |
| STATION_SEARCH | 0 | 0 | 6 | 0 | 0 |
| TRAIN_DETAILS | 0 | 0 | 0 | 12 | 0 |
| TRAIN_SEARCH | 0 | 0 | 0 | 0 | 13 |

## Misclassified

- none
