# Universal Dataset Intelligence (UDI)

UDI is a conversational, generalized analytics and machine-learning application for heterogeneous CSV/XLS/XLSX datasets.

The application combines:
- **React** for the UI
- **Flask/Python** for data processing and execution
- **Gemini** for natural-language planning and explanation
- **Pandas/scikit-learn/etc.** for verified computation

## Core principle

UDI is **data-driven and question-driven, not keyword-driven**.

The selected domain provides context only. The uploaded dataset determines which columns, statistics, models, and operations are valid.

Runtime flow:

```text
Upload dataset
  ↓
Profile schema and data
  ↓
Gemini creates structured analysis plan
  ↓
Validate plan against actual dataset
  ↓
Execute controlled Python operation
  ↓
Return verified result
  ↓
Gemini explains verified result
```

Gemini does not calculate numerical answers and cannot execute arbitrary Python.

## Domains

Company · Hospital · College · Municipality · NGO · Community · Event Organizer

These are contextual environments, not fixed model assignments.

## Supported operations

- `get_summary`
- `get_current_state`
- `compare_groups`
- `analyze_trend`
- `detect_anomalies`
- `calculate_correlation`
- `predict_value`
- `predict_class`
- `forecast`
- `run_scenario`

If the dataset cannot support a requested operation, UDI returns a structured limitation instead of substituting an unrelated column or fabricating a result.

## Dynamic ML

For uploaded datasets, UDI evaluates candidate models on the uploaded data when ML is actually required.

Regression candidates:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor when available

Classification candidates:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier when available

Regression evaluation includes MAE, RMSE and R². Classification evaluation includes Accuracy, Precision, Recall, F1 and ROC-AUC where appropriate.

Identifier-like columns, missing values, categorical variables, dates and preprocessing are handled through the runtime pipeline. No domain model is blindly applied to an uploaded dataset.

## Forecasting

Forecasting requires a real time column, numerical target, sufficient historical observations and distinct time points. Validation is time-aware; observations are not randomly shuffled for the time-series backtest.

## Scenario analysis

A what-if question identifies a numerical feature change, trains/uses a verified regression relationship, compares baseline and scenario predictions, and clearly labels the result as a model-based estimate rather than causal evidence.

## Follow-up context

Each session retains recent user/assistant messages, the previous plan and previous result. Follow-up questions can refer to the prior analysis without repeating the dataset or target. Ambiguous follow-ups are rejected or clarified rather than guessed.

## Gemini configuration

Create `backend/.env` from `backend/.env.example`:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.8-flash
UDI_MOCK_MODE=false
FLASK_PORT=5000
```

The default model is configurable through `GEMINI_MODEL`; the packaged default is `gemini-3.8-flash`.

Never commit a real API key. No API key is included in this ZIP.

Google's current Gemini API model documentation lists Gemini 3.8 Flash as a stable model and recommends current Gemini 3.x models for new projects. Always verify model availability for the API account before deployment.

## Windows setup

The repository includes:

```text
run_windows.bat
```

It:
1. Creates `backend\.venv` if needed.
2. Installs backend dependencies.
3. Creates `backend\.env` from `.env.example` when missing.
4. Installs frontend dependencies.
5. Builds the React frontend.
6. Starts Flask.

You can also run the pieces manually.

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
# Edit .env and add GEMINI_API_KEY
python app.py
```

### Frontend development

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend defaults to:

```text
http://localhost:5173
```

and calls the Flask backend at:

```text
http://localhost:5000
```

Override the API URL with `frontend/.env`:

```env
VITE_API_URL=http://localhost:5000
```

### Production React build

```powershell
cd frontend
npm install
npm run build
```

The Flask application serves `frontend/dist` when that build exists.

## Testing

Backend unit/integration tests:

```powershell
cd backend
python -m pytest -q
```

Python syntax validation:

```powershell
python -m py_compile *.py
```

Frontend production build:

```powershell
cd frontend
npm install
npm run build
```

The generalized acceptance tests include:
- summary
- aggregation
- comparison
- trend
- correlation
- anomaly detection
- regression
- classification
- forecasting
- scenario routing
- invalid-column rejection
- same-dataset/different-question routing

## Sample datasets

Realistic sample datasets for all seven domains are retained under:

```text
backend/sample_data/
```

They are demonstration/test data only. Uploaded datasets do not need to use the same column names or structures.

## Reference model artifacts

`backend/models/<domain>/` contains domain-labelled artifacts retained for:
- reproducibility
- validation
- demonstration

They are **not** automatically used for uploaded runtime datasets.

## Security

- API keys are excluded from the package.
- Uploaded files are stored temporarily and removed after parsing.
- Gemini receives dataset schema/profile information and limited context rather than the entire dataset.
- Gemini output is parsed into a strict Pydantic plan.
- Plans are validated against actual columns before execution.
- Gemini-generated Python is never executed.
- Numerical results come from Python.
- Explanation generation receives verified results rather than raw untrusted calculations.

## Limitations

UDI does not claim to answer every possible question on every dataset. An analysis is supported only when the dataset contains sufficient, valid information for the requested operation.

Forecasts are model-based projections and can be unreliable when historical patterns are unstable. Correlation does not establish causation. Scenario analysis estimates model response and does not establish causal impact. Anomaly flags require domain review.

## Final project structure

```text
UDI_React_Flask/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── gemini_client.py
│   ├── planner.py
│   ├── schemas.py
│   ├── validator.py
│   ├── data_profiler.py
│   ├── analytics_engine.py
│   ├── ml_engine.py
│   ├── forecasting_engine.py
│   ├── anomaly_engine.py
│   ├── visualization.py
│   ├── result_formatter.py
│   ├── session_manager.py
│   ├── reference_models.py
│   ├── train_models.py
│   ├── sample_data/
│   ├── models/
│   └── tests/
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   └── .env.example
├── docs/
├── run_windows.bat
└── README.md
```

No `.env`, virtual environment, `node_modules`, Python cache, uploaded dataset, or temporary build cache is distributed.
