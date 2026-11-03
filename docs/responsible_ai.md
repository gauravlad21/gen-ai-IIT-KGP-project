# Responsible AI

- **Prompt injection:** Treat text from PDFs, tables, OCR, and images as untrusted evidence. Never follow instructions embedded in retrieved content. Test with controlled fixtures and verify the response remains within the question and domain.
- **PII:** Financial documents may contain account numbers, names, addresses, and other sensitive data. Use synthetic/redacted documents in demonstrations, avoid logging raw document text or images, restrict access to local files, and delete temporary uploads when no longer needed. Add a reviewed redaction step before external model calls.
- **Refusals:** Refuse when evidence is absent, citations cannot be validated, visual content is unclear, or the request requires advice outside this project’s scope. Explain the limitation without fabricating a substitute answer.
- **Financial boundaries:** The copilot provides document-grounded information and arithmetic, not personalized recommendations. It must not autonomously execute trades, transfers, payments, filings, or other consequential financial actions.
- **Human oversight:** A human must verify outputs before using them in a financial decision. Manually review evaluation labels, visual extraction, and any reported experiment results.

This is a teaching prototype, not a regulated financial product or a privacy/security certification.
