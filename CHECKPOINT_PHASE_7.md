# SAARTHI checkpoint — Phase 7

Checkpoint date: 2026-10-05

## Phase status

- Phase 1 — project/material audit: complete.
- Phase 2 — new repository foundation: complete.
- Phase 3 — crop recommendation ANN: complete; trained artifacts and held-out evaluation are included.
- Phase 4 — NEUTRA-BOOST fertilizer recommendation: complete; trained pipeline and evaluation are included.
- Phase 5 — disease asset audit and truthful unavailable state: complete; no disease ML was fabricated.
- Phase 6 — source-aware agricultural consultancy MVP: complete.
- Phase 7 — authentication, persistence/history, scheme/document status, and appointments: complete.

## Currently working modules

- React/Vite frontend and one Flask backend.
- Crop ANN prediction with seven inputs, ranked probabilities, and authenticated history.
- NEUTRA-BOOST fertilizer prediction with its full 19-input contract and authenticated history.
- Source-aware consultancy with persisted authenticated conversations.
- Farmer registration/login and session-based authentication; officer accounts provisioned locally.
- Appointment requests and officer accept/decline workflow.
- UTF-8 plain-text document intake, explicitly marked user-provided and unverified.
- Disease UI explicitly reports that disease analysis is unavailable.

## Important model artifacts

- Crop ANN: `backend/app/ml/artifacts/crop_model.pt`, `scaler.joblib`, `label_mapping.json`, `metadata.json`, `evaluation.json`, `training_history.json`.
- NEUTRA-BOOST: `backend/app/ml/fertilizer_artifacts/neutra_boost.joblib`, `scaler.joblib`, `category_mappings.json`, `label_mapping.json`, `metadata.json`, `evaluation.json`.
- Training entry points: `ml_training/crop/train.py` and `ml_training/fertilizer/train.py`.
- Raw supplied CSV datasets are not stored in this repository.

## Known limitations

- No project-owned disease dataset, class mapping, ViT weights/code, authenticity gate, or severity/treatment evidence was present. Disease inference is unavailable; no detector is represented as functional.
- No verified scheme corpus/rules exist. Scheme recommendations and eligibility determinations are unavailable.
- Document workflow accepts UTF-8 `.txt` only. PDF/image OCR is unavailable.
- Consultancy has no external LLM, verified agricultural knowledge corpus, weather or market feeds.
- Appointment requests do not establish real officer availability; officer accounts must be provisioned by an administrator.
- SQLAlchemy creates initial tables, but no schema migration framework, account recovery, email verification, or rate limiting is implemented.
- SQLite development persistence is local-only; production database deployment has not been verified.

## Disease module and next step

Disease module status: unavailable pending valid project-owned disease assets and evaluation evidence. The interface must continue to say so.

Recommended next step: **Phase 8 Disease ViT**. First obtain/identify valid project-owned disease images, labels, and any ViT/authenticity/severity assets; implement only capabilities supported by those assets and verified evaluation. Do not claim disease inference until then.

## Development/runtime information

- Frontend: React + Vite; development server uses the Vite URL (currently `http://127.0.0.1:5173/`) and proxies API traffic to Flask.
- Backend: one Flask app via `backend.wsgi`; local development command: `python -m flask --app backend.wsgi run --debug`.
- Database: SQLAlchemy with default `sqlite:///saarthi.db`, resolved under `instance/`; database/runtime files are ignored by Git. PostgreSQL through Docker Compose is optional and was not verified for this checkpoint.
- Requirements recorded in `README.md`: Python 3.10+, Node.js 22.12+, pnpm 11.19.0.
- Production requires a persistent `SECRET_KEY`; it must remain in environment configuration and never be committed.
- Frontend build command: `pnpm --dir frontend build`.
