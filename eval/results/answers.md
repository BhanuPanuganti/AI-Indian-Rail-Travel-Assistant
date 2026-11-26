# RAG answer evaluation (LLM judge)

45 questions · judge: same Gemini model as the generator (scores are indicative).

| Metric | Value |
|---|---|
| Faithfulness | 1.000 |
| Answer relevance | 0.983 |
| Key-fact recall | 0.878 |
| Citation precision | 0.404 |
| Correct refusals (unanswerable) | 1.000 |
| False refusals (answerable) | 0.024 |
| Answer latency p50 (s) | 2.481 |
| Answer latency p95 (s) | 30.604 |
| LLM calls (answer + judge) | 90 |
| Groq fallbacks | 5 |
| Errors | 1 |

## Answers to review (faithfulness < 0.8 or missing key facts)

- `eticket_student_concession` faithfulness 1.00, key facts 1/2
- `booking_hours` faithfulness 1.00, key facts 0/2
- `tatkal_sleeper_max` faithfulness 1.00, key facts 1/2
- `tatkal_refund` faithfulness 1.00, key facts 1/2
- `train_fully_cancelled` faithfulness 1.00, key facts 0/2
- `luggage_free_sleeper` faithfulness 1.00, key facts 1/2
- `concession_students_hometown` faithfulness 1.00, key facts 1/2
- `duplicate_ticket_after_chart` faithfulness 1.00, key facts 1/2
