# UDI Architecture

```text
React UI
   │
   ├── Domain context selection
   ├── CSV/XLS/XLSX upload
   └── Conversational chat + follow-ups
   │
   ▼
Flask REST API
   │
   ▼
Dataset Profiler
   │  schema, types, missingness, dates, identifiers, statistics
   ▼
Gemini Structured Planner
   │  operation + exact dataset columns + parameters
   ▼
Validation Gate
   │  rejects invented/missing columns and unsupported tasks
   ▼
Controlled Python Operation Library
   ├── Descriptive analytics
   ├── Group comparison
   ├── Trend/current-state analysis
   ├── Correlation
   ├── IQR/Z-score/Isolation Forest anomaly detection
   ├── Dynamic regression model selection
   ├── Dynamic classification model selection
   ├── Time-aware forecasting
   └── Controlled what-if scenario analysis
   │
   ▼
Verified Result Object
   │
   ▼
Gemini Explanation Layer
   │  receives verified results only
   ▼
React result cards + charts + tables
```

## Runtime model principle

The seven domains are context labels, not model locks. Runtime ML is trained/evaluated from the uploaded dataset and the requested task. Domain-labelled model artifacts are retained only for reproducibility, validation, and demonstration; they are not automatically loaded for uploaded data.

Gemini never receives the full dataset, never calculates the result, never generates executable Python, and never becomes the source of numerical truth.
