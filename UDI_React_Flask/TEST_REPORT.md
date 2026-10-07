# UDI Verification Report

Date: 2026-10-07

## Backend verification

- `python -m py_compile *.py` — PASS
- `python -m pytest -q` — PASS
- Test result: **12 passed**
- Added generalized acceptance tests covering:
  - same dataset / different questions → different operations
  - requested summary aggregation
  - group comparison
  - uploaded-schema regression
  - uploaded-schema classification
  - invented target rejection
  - time-series forecasting requirements
  - anomaly detection

## Architectural verification

- Runtime uploaded-data regression/classification no longer loads domain reference models.
- Domain model artifacts remain only as reference/validation/demo assets.
- Analysis plans include explicit aggregation and feature columns.
- Plans are validated against actual uploaded columns before execution.
- Scenario analysis reuses the actual trained model's feature set.
- Structured backend error codes are returned for important failures.
- No arbitrary Gemini-generated Python execution is present.
- No API key is packaged.

## Frontend verification

`npm run build` was **not certified in this Linux sandbox**.

Reason: the uploaded `node_modules` contains Windows-specific native Rolldown/Vite optional dependencies, while the sandbox is Linux. Reinstalling from the npm registry was attempted, but this environment has no registry/network access. Therefore no false build-pass claim is made.

Run on Windows:

```powershell
cd frontend
npm install
npm run build
```

## Final package hygiene

The distributable package excludes:
- `.env`
- virtual environments
- `node_modules`
- `__pycache__`
- `.pyc`
- temporary uploads
- debug/cache artifacts
