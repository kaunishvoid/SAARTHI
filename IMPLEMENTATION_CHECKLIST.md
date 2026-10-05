# SAARTHI phased implementation plan

This is the status ledger for the phase-by-phase build. A phase is not started until approved and is closed with a report and a pause for review.

## Phase 1 — Project audit

- [x] Inspect the supplied synopsis, implementation specification, crop and fertilizer CSVs, NEUTRA-BOOST notebook, and attached kickoff notes.
- [x] Record unavailable project inputs and dataset/model limitations.

## Phase 2 — New repository foundation

- [x] Establish a new local Git repository without a remote.
- [x] Create a React/Vite frontend and a single Flask backend.
- [x] Configure SQLAlchemy with persistent local SQLite and environment-configured PostgreSQL support.
- [x] Connect the frontend to Flask through the Vite development proxy.
- [x] Verify production frontend build, frontend HTTP response, Flask boot, and database health.
- [x] Verify database failure returns HTTP 503 without exposing connection credentials.
- [ ] Verify PostgreSQL against a running PostgreSQL service (Docker is not installed in the current environment).

## Phase 3 — Crop ANN

- [x] Train a new ANN from the supplied crop dataset with stratified splits, train-only scaling, early stopping, and held-out evaluation.
- [x] Save model, preprocessing, class mapping, and metadata artifacts.
- [x] Add validated Flask prediction API and Top-3 user interface.
- [x] Verify API prediction, validation responses, startup artifact loading, and frontend-to-backend smoke flow.

## Phase 4 — NEUTRA-BOOST fertilizer recommendation

- [x] Resolve the 19-input notebook model versus eight-field implementation-spec contract by exposing the complete dataset feature set.
- [x] Build a stratified held-out evaluation and artifact export workflow with train-only preprocessing around the supplied NEUTRA-BOOST implementation; document the notebook's prior full-dataset sensitivity/CV use.
- [x] Add validated prediction and schema APIs plus a user interface for all 19 model inputs.
- [x] Verify startup artifact loading, API validation, sorted class scores, and browser-to-model-to-UI prediction flow.

## Phase 5 — Disease analysis

- [ ] Obtain or identify project-owned disease data/model weights and any real treatment mapping before implementing inference.
- [ ] Keep derived severity distinct from learned disease classification.

## Phase 6 — Agentic consultancy and knowledge retrieval

- [x] Implement project-scoped crop/fertilizer routing and model-context orchestration inside Flask.
- [x] Keep unavailable knowledge explicit; no agricultural corpus, external services, or credentials are configured.
- [x] Add a source-aware chat interface and verify chat routing, validation, context use, and browser-to-Flask flow.

## Phase 7 — Application completion

- [x] Add farmer authentication with password hashing, role checks, signed sessions, and CSRF protection.
- [x] Persist authenticated crop/fertilizer predictions and expose per-account history.
- [x] Persist authenticated consultancy conversations and messages.
- [x] Add account-owned plain-text document intake with explicit unverified-source labeling.
- [x] Report verified scheme recommendations and OCR unavailable when no source corpus/OCR engine exists.
- [x] Add pending appointment requests and officer accept/decline workflow without fabricated officers or availability.
- [x] Keep disease inference visibly unavailable because no supported disease assets exist.
- [x] Verify API auth/ownership/persistence, schemes, appointments, frontend build, and browser flows.
- [ ] Obtain real scheme sources and OCR support before enabling factual eligibility advice or scanned document handling.
- [ ] Resolve voice and market data/service requirements before enabling those capabilities.

## Phase 8 — Integration, deployment, and new GitHub repository

- [ ] Complete deployment configuration and clean-checkout verification.
- [ ] Deploy the app and database.
- [ ] After review approval, create a new GitHub repository for SAARTHI and push this repository there.
