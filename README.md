# SAARTHI

SAARTHI is an agricultural decision-support MVP with crop ANN and NEUTRA-BOOST fertilizer recommendations, source-aware consultancy, farmer accounts/history, request-based officer consultations, and a user-owned text-document workflow. Disease inference and verified scheme recommendations are explicitly unavailable because their project-owned model/data/source assets are absent.

## Architecture

- **Frontend:** React and Vite single-page application.
- **Backend:** one modular Flask application, served through `backend.wsgi`.
- **Database:** SQLAlchemy. Local development uses a persistent SQLite file so the application runs without a database service. Set `DATABASE_URL` to PostgreSQL for deployment or use the optional Compose service.
- **Development communication:** Vite proxies `/health` and `/api/*` to Flask, avoiding permissive cross-origin configuration.
- **Configuration:** backend settings come from environment variables or the root `.env` file. Frontend production API origin is the non-secret `VITE_API_BASE_URL`.

## Requirements

- Python 3.10 or newer
- Node.js 22.12 or newer
- pnpm 11.19.0 (declared in `frontend/package.json`)
- PostgreSQL and Docker are optional for development; SQLite is the default.

## Run locally

From the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m flask --app backend.wsgi run --debug
```

In a second terminal:

```powershell
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend dev
```

Open the Vite URL shown in the terminal. Crop and fertilizer forms continue to use their existing prediction APIs. The database schema is initialized from SQLAlchemy models at Flask startup. Farmer registration/sign-in uses signed, HTTP-only, same-site session cookies and CSRF tokens; passwords are stored as scrypt hashes. Set a persistent `SECRET_KEY` in the environment before production, and set `SESSION_COOKIE_SECURE=true` behind HTTPS.

Public registration creates farmer accounts only. Provision real officer accounts locally with `python -m backend.manage create-officer --email officer@example.org --name "Officer Name"`; the command prompts for the password. Do not use invented officers or availability. Farmer prediction history and consultancy messages are saved to the signed-in account. Guest predictions and chat still work but are not persisted.

`GET /api/schemes/status` reports that no verified scheme source is installed. Signed-in farmers can upload a UTF-8 `.txt` document for account-scoped text extraction; the document remains user-provided and unverified, and no eligibility decision is made. PDF/image uploads and OCR are unavailable. Appointment submissions create pending requests only. Officers must be provisioned before a request can be accepted; acceptance records an officer response, while no availability calendar or roster is fabricated.

## Crop model

The crop ANN was trained from the supplied `Crop_recommendation (1).csv`; the CSV itself is not copied into this repository. To retrain, install the CPU PyTorch wheel followed by the backend requirements, then run:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend\requirements.txt
python -m ml_training.crop.train --dataset "C:\path\to\Crop_recommendation (1).csv"
```

Training writes the model weights, scaler, class mapping, metadata, held-out evaluation, and history under `backend/app/ml/artifacts/`. The supplied CSV itself is not copied into this repository. Set `CROP_MODEL_ARTIFACT_DIR` to the artifact directory when it differs from the default. Flask loads the model once during application startup. The API accepts JSON fields `nitrogen`, `phosphorus`, `potassium`, `temperature`, `humidity`, `ph`, and `rainfall` and returns the predicted crop, confidence, and probability-ranked Top 3.

## NEUTRA-BOOST fertilizer model

The fertilizer model uses all 19 predictors from the supplied `fertilizer_recommendation.csv`: `Soil_Type`, `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Nitrogen_Level`, `Phosphorus_Level`, `Potassium_Level`, `Temperature`, `Humidity`, `Rainfall`, `Crop_Type`, `Crop_Growth_Stage`, `Season`, `Irrigation_Type`, `Previous_Crop`, `Region`, `Fertilizer_Used_Last_Season`, and `Yield_Last_Season`. The eight fields listed in an earlier app-spec section are insufficient for this notebook model, so the form does not present them as a valid reduced input set. The dataset itself is not copied into this repository. Reproduce the fixed-parameter held-out training run with:

```powershell
python -m ml_training.fertilizer.train --dataset "C:\path\to\fertilizer_recommendation.csv"
```

The training script preserves per-column label encoding for categorical inputs followed by standard scaling across the encoded 19-feature matrix. Both encoders and scaler are fitted only on the stratified training split. It writes the custom NEUTRA-BOOST model (including its confusion-pair specialists), scaler, category mappings, class mapping, metadata, and held-out evaluation under `backend/app/ml/fertilizer_artifacts/`. Flask loads these artifacts once during startup; set `FERTILIZER_MODEL_ARTIFACT_DIR` to override the directory. The API schema endpoint supplies valid categorical choices. Its reported confidence is the active decision path's raw score and is not calibrated; recommendations are fertilizer classes, not dosage advice. The supplied notebook used the full dataset during cross-validation and sensitivity selection, so its historical scores and any later split from that same CSV are not a pristine external validation.

## Agentic consultancy

The source-aware consultancy is available at `POST /api/agent/chat` and in the chat interface below the prediction modules. It routes crop and fertilizer questions and, when current module inputs are attached, recomputes the corresponding predictions using the loaded SAARTHI services. It does not accept client-asserted model outputs as facts. Authenticated conversation messages are persisted in the database; lightweight routing context is held in memory for up to one hour. There is no connected agricultural knowledge corpus, external LLM, disease model, scheme source, weather provider, or market feed; requests requiring those sources receive an explicit unavailable response. No external credentials are required.

## Optional PostgreSQL development database

Set `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` in `.env`. Change `DATABASE_URL` to a matching URL, for example:

```text
postgresql+psycopg://saarthi:YOUR_LOCAL_PASSWORD@127.0.0.1:5432/saarthi?connect_timeout=3
```

Then run `docker compose up -d db`. Docker is not required when using SQLite. Do not commit `.env` or put real credentials in source control.

## Production frontend build

For deployment, the simplest setup is to serve the frontend and route `/health` and `/api/*` to Flask through the same public origin. If the frontend and API use separate origins, restricted CORS configuration will be needed before setting `VITE_API_BASE_URL` in `frontend/.env`. Then run:

```powershell
pnpm --dir frontend build
```

Serve the generated `frontend/dist` from a frontend host. Configure Flask and PostgreSQL credentials through the hosting provider's environment settings.

## Verified checks

- `GET /health` executes `SELECT 1` through SQLAlchemy and returns HTTP 200 only when the configured database responds.
- API and database failure return a structured HTTP 503 response without exposing connection credentials.
- Crop and fertilizer model artifacts are loaded at Flask startup; prediction endpoints return 503 if required artifacts are missing.
- Fertilizer input schema and validation use the full 19-feature dataset contract.
