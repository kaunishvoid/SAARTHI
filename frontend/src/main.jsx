import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  analyzeSchemeDocument, createAppointment, getAgentHistory, getAppointments, getCropHistory,
  getFertilizerHistory, getFertilizerSchema, getHealth, getSchemeDocuments, getSchemeStatus,
  getSession, login, logout, predictCrop, predictFertilizer, registerFarmer,
  sendConsultancyMessage, updateAppointment, uploadSchemeDocument,
} from './api/client.js'
import './style.css'

const fields = [
  { name: 'nitrogen', label: 'Nitrogen (N)', step: 'any' },
  { name: 'phosphorus', label: 'Phosphorus (P)', step: 'any' },
  { name: 'potassium', label: 'Potassium (K)', step: 'any' },
  { name: 'temperature', label: 'Temperature', step: 'any' },
  { name: 'humidity', label: 'Humidity (%)', step: 'any' },
  { name: 'ph', label: 'Soil pH', step: 'any' },
  { name: 'rainfall', label: 'Rainfall', step: 'any' },
]

const initialValues = Object.fromEntries(fields.map(({ name }) => [name, '']))
const percentage = (value) => value < 0.001 ? '<0.1%' : `${(value * 100).toFixed(1)}%`

function App() {
  const [health, setHealth] = useState('checking')
  const [values, setValues] = useState(initialValues)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [fertilizerContext, setFertilizerContext] = useState(null)
  const [cropInputs, setCropInputs] = useState(null)
  const [user, setUser] = useState(null)
  const [historyVersion, setHistoryVersion] = useState(0)

  function handleUserChange(nextUser) {
    setUser(nextUser)
    setResult(null)
    setCropInputs(null)
    setValues(initialValues)
    setFertilizerContext(null)
  }

  useEffect(() => {
    getHealth().then(() => setHealth('connected')).catch(() => setHealth('disconnected'))
    getSession().then((state) => setUser(state.authenticated ? state.user : null)).catch(() => setUser(null))
  }, [])

  async function submit(event) {
    event.preventDefault()
    setError('')
    setResult(null)
    setCropInputs(null)
    setSubmitting(true)
    try {
      const payload = Object.fromEntries(fields.map(({ name }) => [name, Number(values[name])]))
      const prediction = await predictCrop(payload)
      setResult(prediction)
      setCropInputs(payload)
      if (user) setHistoryVersion((value) => value + 1)
    } catch (submissionError) {
      setError(submissionError.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="SAARTHI home"><span className="brand-mark">S</span><span>SAARTHI</span></a>
        <span className="prototype">Agricultural decision support</span>
      </header>

      <AuthSection user={user} onUserChange={handleUserChange} />

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">CROP RECOMMENDATION · ANN MODEL</p>
        <h1 id="page-title">Find a crop that<br />fits your field.</h1>
        <p className="intro">Enter the soil and climate values to receive a ranked prediction from the supplied crop dataset.</p>
        <div className="connection-row"><div className="status-card" data-state={health} role="status" aria-live="polite"><span className="status-dot" />{health === 'connected' ? 'SAARTHI API and database are connected.' : health === 'checking' ? 'Checking SAARTHI API…' : 'API connection unavailable.'}</div></div>
      </section>

      <section className="recommendation-layout">
        <form className="input-card" onSubmit={submit}>
          <div className="section-heading"><div><p className="eyebrow">FIELD CONDITIONS</p><h2>Your measurements</h2></div><span className="field-count">7 values</span></div>
          <div className="field-grid">
            {fields.map(({ name, label, step }) => (
              <label className="field" key={name} htmlFor={name}>
                <span>{label}</span>
                <input id={name} name={name} type="number" step={step} required value={values[name]} onChange={(event) => { setValues({ ...values, [name]: event.target.value }); setResult(null); setCropInputs(null) }} />
              </label>
            ))}
          </div>
          <p className="range-hint">Use numeric values. pH must be between 0 and 14; rainfall cannot be negative.</p>
          <button className="submit-button" type="submit" disabled={submitting || health !== 'connected'}>{submitting ? 'Getting recommendation…' : 'Recommend crops'}<span aria-hidden="true">→</span></button>
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>

        <aside className="result-card" aria-live="polite">
          <p className="eyebrow">YOUR RESULT</p>
          {result ? <>
            <p className="result-label">Recommended crop</p>
            <h2 className="best-crop">{result.predicted_crop}</h2>
            <p className="confidence">Model probability · {percentage(result.confidence)}</p>
            <div className="ranked-list"><h3>Top 3 predictions</h3>{result.top_3.map((crop, index) => <div className="rank-row" key={crop.crop}><span className="rank-number">0{index + 1}</span><span className="rank-crop">{crop.crop}</span><span className="rank-prob">{percentage(crop.probability)}</span></div>)}</div>
            <p className="caution">{result.caution}</p>
          </> : <div className="empty-result"><span className="seed-mark" aria-hidden="true">✳</span><h2>Recommendations will appear here</h2><p>Submit all seven values to see the predicted crop and its top alternatives.</p></div>}
        </aside>
      </section>

      <FertilizerSection onResult={setFertilizerContext} user={user} onSaved={() => setHistoryVersion((value) => value + 1)} />
      <ConsultancySection cropInputs={cropInputs} fertilizerInputs={fertilizerContext?.inputs} user={user} onSaved={() => setHistoryVersion((value) => value + 1)} />
      <AccountWorkflows user={user} refreshKey={historyVersion} />
      <DiseaseUnavailableCard />
      <footer>SAARTHI · Model predictions are not guaranteed agricultural outcomes or fertilizer dosage advice.</footer>
    </main>
  )
}

const fertilizerLabels = {
  Soil_Type: 'Soil type', Soil_pH: 'Soil pH', Soil_Moisture: 'Soil moisture',
  Organic_Carbon: 'Organic carbon', Electrical_Conductivity: 'Electrical conductivity',
  Nitrogen_Level: 'Nitrogen level', Phosphorus_Level: 'Phosphorus level',
  Potassium_Level: 'Potassium level', Temperature: 'Temperature', Humidity: 'Humidity',
  Rainfall: 'Rainfall', Crop_Type: 'Crop type', Crop_Growth_Stage: 'Crop growth stage',
  Season: 'Season', Irrigation_Type: 'Irrigation type', Previous_Crop: 'Previous crop',
  Region: 'Region', Fertilizer_Used_Last_Season: 'Fertilizer used last season',
  Yield_Last_Season: 'Yield last season',
}

function FertilizerSection({ onResult, user, onSaved }) {
  const [schema, setSchema] = useState(null)
  const [schemaError, setSchemaError] = useState('')
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setLoading(true)
    getFertilizerSchema()
      .then((definition) => {
        setSchema(definition)
        setValues(Object.fromEntries(definition.input_fields.map(({ name }) => [name, ''])))
      })
      .catch((loadError) => setSchemaError(loadError.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    setResult(null)
    onResult(null)
    if (schema) setValues(Object.fromEntries(schema.input_fields.map(({ name }) => [name, ''])))
  }, [user?.id])

  async function submit(event) {
    event.preventDefault()
    setError('')
    setResult(null)
    onResult(null)
    setSubmitting(true)
    try {
      const payload = Object.fromEntries(schema.input_fields.map((field) => [
        field.name,
        field.type === 'number' ? Number(values[field.name]) : values[field.name],
      ]))
      const prediction = await predictFertilizer(payload)
      setResult(prediction)
      onResult({ inputs: payload, prediction })
      if (user) onSaved()
    } catch (submissionError) {
      setError(submissionError.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="fertilizer-section" aria-labelledby="fertilizer-title">
      <div className="fertilizer-heading">
        <p className="eyebrow">FERTILIZER RECOMMENDATION · NEUTRA-BOOST</p>
        <h2 id="fertilizer-title">Use the model’s full input set.</h2>
        <p className="intro">The supplied NEUTRA-BOOST implementation requires all 19 dataset fields. No field is inferred from the crop model or filled in for you.</p>
      </div>
      {schemaError ? <p className="error-message" role="alert">Could not load NEUTRA-BOOST input options: {schemaError}</p> : loading || !schema ? <p className="status-card" role="status">Loading the model input schema…</p> : (
        <div className="fertilizer-layout">
          <form className="input-card" onSubmit={submit}>
            <div className="section-heading"><div><p className="eyebrow">19 MODEL INPUTS</p><h2>Field and crop context</h2></div><span className="field-count">{schema.input_fields.length} required</span></div>
            <div className="fertilizer-fields">
              {schema.input_fields.map((field) => (
                <label className="field" key={field.name} htmlFor={`fert-${field.name}`}>
                  <span>{fertilizerLabels[field.name] || field.name}</span>
                  {field.type === 'categorical' ? (
                    <select id={`fert-${field.name}`} name={field.name} required value={values[field.name] || ''} onChange={(event) => { setValues({ ...values, [field.name]: event.target.value }); onResult(null); setResult(null) }}>
                      <option value="" disabled>Select {fertilizerLabels[field.name]?.toLowerCase() || 'a value'}</option>
                      {field.options.map((option) => <option value={option} key={option}>{option}</option>)}
                    </select>
                  ) : (
                    <input id={`fert-${field.name}`} name={field.name} type="number" step="any" required value={values[field.name] || ''} onChange={(event) => { setValues({ ...values, [field.name]: event.target.value }); onResult(null); setResult(null) }} />
                  )}
                </label>
              ))}
            </div>
            <p className="range-hint">Enter measured values in the same units and scale as the supplied dataset. The result is a fertilizer class prediction, not a dosage.</p>
            <button className="submit-button" type="submit" disabled={submitting}>{submitting ? 'Getting recommendation…' : 'Recommend fertilizer'}<span aria-hidden="true">→</span></button>
            {error && <p className="error-message" role="alert">{error}</p>}
          </form>

          <aside className="result-card fertilizer-result" aria-live="polite">
            <p className="eyebrow">NEUTRA-BOOST RESULT</p>
            {result ? <>
              <p className="result-label">Recommended fertilizer</p>
              <h2 className="best-crop">{result.fertilizer}</h2>
              <p className="confidence">Decision confidence · {percentage(result.confidence)} · {result.confidence_source}</p>
              <div className="specialist-status" data-used={result.used_specialist}>
                {result.used_specialist ? `AMADR specialist used${result.specialist_pair ? ` for ${result.specialist_pair.join(' / ')}` : ''}.` : result.confusion_gate_triggered ? 'AMADR confusion gate triggered; no specialist override was applied.' : 'Direct NEUTRA-BOOST prediction; no specialist override was applied.'}
              </div>
              <div className="ranked-list"><h3>Top class probabilities</h3>{result.top_3.map((item, index) => <div className="rank-row" key={item.fertilizer}><span className="rank-number">0{index + 1}</span><span className="rank-crop">{item.fertilizer}</span><span className="rank-prob">{percentage(item.probability)}</span></div>)}</div>
              {result.used_specialist && <p className="range-hint">The specialist-adjusted recommendation can differ from the highest base softmax probability. Listed class probabilities are the base model scores.</p>}
              <p className="caution">{result.caution} Scores are not calibrated probabilities.</p>
            </> : <div className="empty-result"><span className="seed-mark" aria-hidden="true">✳</span><h2>Fertilizer result will appear here</h2><p>Complete all 19 fields to run the supplied NEUTRA-BOOST model.</p></div>}
          </aside>
        </div>
      )}
    </section>
  )
}

function ConsultancySection({ cropInputs, fertilizerInputs, user, onSaved }) {
  const [conversationId, setConversationId] = useState(() => globalThis.crypto?.randomUUID?.() || `chat_${Date.now()}`)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setMessages([])
    setConversationId(globalThis.crypto?.randomUUID?.() || `chat_${Date.now()}`)
  }, [user?.id])

  async function submit(event) {
    event.preventDefault()
    const message = draft.trim()
    if (!message || sending) return
    setDraft('')
    setError('')
    setMessages((current) => [...current, { role: 'user', text: message }])
    setSending(true)
    try {
      const result = await sendConsultancyMessage({
        message,
        conversation_id: conversationId,
        context: { ...(cropInputs ? { crop_inputs: cropInputs } : {}), ...(fertilizerInputs ? { fertilizer_inputs: fertilizerInputs } : {}) },
      })
      setMessages((current) => [...current, { role: 'assistant', text: result.response, intent: result.intent, sources: result.sources }])
      if (user) onSaved()
    } catch (sendError) {
      setError(sendError.message)
    } finally {
      setSending(false)
    }
  }

  return (
    <section className="consultancy-section" aria-labelledby="consultancy-title">
      <div className="fertilizer-heading">
        <p className="eyebrow">SAARTHI CONSULTANCY · SOURCE-AWARE</p>
        <h2 id="consultancy-title">Ask about your SAARTHI results.</h2>
        <p className="intro">This assistant can summarize crop and NEUTRA-BOOST results from this page. It does not have a verified agricultural knowledge library or live data sources.</p>
      </div>
      <div className="chat-card">
        <div className="chat-transcript" aria-live="polite" aria-label="Conversation">
          {messages.length === 0 && <p className="chat-empty">Try asking “What crop did the model recommend?” after running a recommendation above.</p>}
          {messages.map((item, index) => (
            <article className={`chat-message ${item.role}`} key={`${index}-${item.role}`}>
              <p>{item.text}</p>
              {item.role === 'assistant' && <span className="chat-meta">{item.intent.replaceAll('_', ' ')}{item.sources?.length ? ` · source: ${item.sources.map((source) => source.id).join(', ')}` : ' · no verified source used'}</span>}
            </article>
          ))}
          {sending && <p className="chat-pending" role="status">SAARTHI is checking available module context…</p>}
        </div>
        <form className="chat-form" onSubmit={submit}>
          <label className="sr-only" htmlFor="consultancy-message">Your question</label>
          <textarea id="consultancy-message" rows="2" maxLength="2000" required value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask a question about your crop or fertilizer result…" />
          <button className="submit-button" type="submit" disabled={sending || !draft.trim()}>Send question <span aria-hidden="true">→</span></button>
        </form>
        {error && <p className="error-message" role="alert">{error}</p>}
        <p className="range-hint">{user ? 'This conversation is saved to your account.' : 'Sign in to save conversations.'} No external LLM, knowledge base, disease detector, schemes, or live agricultural data are connected.</p>
      </div>
    </section>
  )
}

function AuthSection({ user, onUserChange }) {
  const [mode, setMode] = useState('login')
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const result = mode === 'register'
        ? await registerFarmer({ full_name: fullName, email, password })
        : await login({ email, password })
      onUserChange(result.user)
      setPassword('')
    } catch (authError) {
      setError(authError.message)
    } finally {
      setBusy(false)
    }
  }

  async function signOut() {
    setError('')
    try {
      await logout()
      onUserChange(null)
    } catch (authError) { setError(authError.message) }
  }

  return (
    <section className="auth-section" aria-label="Account">
      {user ? (
        <div className="account-bar"><span>Signed in as <strong>{user.full_name}</strong> · {user.role}{user.demo_account && <span className="demo-badge">DEMO ACCOUNT</span>}</span><button type="button" className="secondary-button" onClick={signOut}>Sign out</button></div>
      ) : (
        <form className="auth-form" onSubmit={submit}>
          <div className="auth-copy"><p className="eyebrow">FARMER ACCOUNT</p><strong>{mode === 'login' ? 'Sign in to save results and request a consultation.' : 'Create a farmer account.'}</strong><p>Officer accounts are provisioned by the local administrator.</p></div>
          <div className="auth-inputs">
            {mode === 'register' && <label className="field"><span>Full name</span><input autoComplete="name" required minLength="2" maxLength="80" value={fullName} onChange={(event) => setFullName(event.target.value)} /></label>}
            <label className="field"><span>Email</span><input type="email" autoComplete="email" required maxLength="254" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
            <label className="field"><span>Password</span><input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength="10" maxLength="256" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
          </div>
          <div className="auth-actions"><button className="submit-button" type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</button><button type="button" className="text-button" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError('') }}>{mode === 'login' ? 'Create an account' : 'I already have an account'}</button></div>
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>
      )}
    </section>
  )
}

function AccountWorkflows({ user, refreshKey }) {
  const [schemeStatus, setSchemeStatus] = useState(null)
  const [cropHistory, setCropHistory] = useState([])
  const [fertilizerHistory, setFertilizerHistory] = useState([])
  const [conversations, setConversations] = useState([])
  const [documents, setDocuments] = useState([])
  const [appointments, setAppointments] = useState([])
  const [requestedFor, setRequestedFor] = useState('')
  const [topic, setTopic] = useState('')
  const [notes, setNotes] = useState('')
  const [file, setFile] = useState(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function reloadAccountData() {
    if (!user) { setCropHistory([]); setFertilizerHistory([]); setConversations([]); setDocuments([]); setAppointments([]); return }
    setError('')
    try {
      const [crops, fertilizers, chats, appts] = await Promise.all([getCropHistory(), getFertilizerHistory(), getAgentHistory(), getAppointments()])
      setCropHistory(crops.items)
      setFertilizerHistory(fertilizers.items)
      setConversations(chats.items)
      setAppointments(appts.items)
      if (user.role === 'farmer') {
        const docs = await getSchemeDocuments()
        setDocuments(docs.items)
      } else setDocuments([])
    } catch (loadError) { setError(loadError.message) }
  }

  useEffect(() => { getSchemeStatus().then(setSchemeStatus).catch((loadError) => setError(loadError.message)) }, [])
  useEffect(() => { reloadAccountData() }, [user?.id, refreshKey])

  async function submitAppointment(event) {
    event.preventDefault()
    setBusy(true); setError(''); setStatusMessage('')
    try {
      const localTime = new FormData(event.currentTarget).get('requested_for')
      const requestedDate = new Date(localTime)
      if (!localTime || Number.isNaN(requestedDate.getTime())) throw new Error('Choose a valid requested date and time.')
      await createAppointment({ requested_for: requestedDate.toISOString(), topic, notes })
      setStatusMessage('Your appointment request was saved as pending. No officer availability is confirmed yet.')
      setRequestedFor(''); setTopic(''); setNotes('')
      await reloadAccountData()
    } catch (submitError) { setError(submitError.message) }
    finally { setBusy(false) }
  }

  async function handleDocumentUpload(event) {
    event.preventDefault()
    if (!file) return
    setBusy(true); setError(''); setStatusMessage('')
    try {
      const uploaded = await uploadSchemeDocument(file)
      const analysis = await analyzeSchemeDocument(uploaded.document.id)
      setDocuments((current) => [uploaded.document, ...current])
      setStatusMessage(`${uploaded.message} ${analysis.message}`)
      setFile(null)
      event.target.reset()
    } catch (uploadError) { setError(uploadError.message) }
    finally { setBusy(false) }
  }

  async function appointmentAction(id, action) {
    setError(''); setStatusMessage('')
    try { await updateAppointment(id, action); setStatusMessage(`Appointment ${action}ed.`); await reloadAccountData() }
    catch (actionError) { setError(actionError.message) }
  }

  const latestLocalDateTime = () => {
    const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000)
    return now.toISOString().slice(0, 16)
  }

  return (
    <section className="account-workflows" aria-labelledby="account-workflows-title">
      <div className="fertilizer-heading"><p className="eyebrow">ACCOUNT & SUPPORT WORKFLOWS</p><h2 id="account-workflows-title">History, documents, and consultation.</h2></div>
      {error && <p className="error-message" role="alert">{error}</p>}
      {statusMessage && <p className="success-message" role="status">{statusMessage}</p>}
      <div className="workflow-grid">
        <article className="workflow-card scheme-card">
          <p className="eyebrow">SCHEMES & DOCUMENTS</p><h3>Verified scheme information is unavailable.</h3>
          <p>{schemeStatus?.message || 'Checking scheme source status…'}</p>
          <p className="range-hint">{schemeStatus?.ocr_available ? 'OCR is available.' : 'PDF and image OCR are not installed. The intake currently supports UTF-8 .txt files only.'}</p>
          {user?.role === 'farmer' ? <>
            <form className="document-form" onSubmit={handleDocumentUpload}>
              <label className="field"><span>Upload your own text document</span><input type="file" accept=".txt,text/plain" onChange={(event) => setFile(event.target.files?.[0] || null)} /></label>
              <button type="submit" className="submit-button" disabled={busy || !file}>Extract text and check availability</button>
            </form>
            {documents.map((document) => <details className="document-item" key={document.id}><summary>{document.filename} · user-provided, unverified</summary><pre>{document.extracted_text}</pre></details>)}
          </> : <p className="range-hint">Sign in as a farmer to store a text document under your account.</p>}
        </article>

        <article className="workflow-card appointment-card">
          <p className="eyebrow">OFFICER CONSULTATION</p><h3>{user?.role === 'officer' ? 'Appointment requests' : 'Request a consultation.'}</h3>
          {user?.role === 'farmer' ? <>
            <p>Choose a requested time. This records a request; it does not indicate that an officer is available or assigned.</p>
            <form className="appointment-form" onSubmit={submitAppointment}>
              <label className="field"><span>Requested date and time</span><input name="requested_for" type="datetime-local" required min={latestLocalDateTime()} value={requestedFor} onInput={(event) => setRequestedFor(event.currentTarget.value)} onChange={(event) => setRequestedFor(event.target.value)} /></label>
              <label className="field"><span>Topic</span><input required minLength="3" maxLength="120" value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="What would you like to discuss?" /></label>
              <label className="field"><span>Notes (optional)</span><textarea rows="3" maxLength="1000" value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
              <button className="submit-button" type="submit" disabled={busy}>Send request</button>
            </form>
          </> : user?.role === 'officer' ? <p>Requests appear here only after an officer account is provisioned and farmers submit requests. No demo officer or availability was seeded.</p> : <p>Sign in as a farmer to request a consultation. Officer accounts are created by a local administrator.</p>}
          {appointments.map((appointment) => <div className="appointment-row" key={appointment.id}><strong>{appointment.topic} {appointment.demo_record && <span className="demo-badge">DEMO REQUEST</span>}</strong><span>{new Date(appointment.requested_for).toLocaleString()} · {appointment.status}</span>{appointment.officer_name && <span>Officer: {appointment.officer_name}</span>}{user?.role === 'farmer' && appointment.status === 'requested' && <button className="secondary-button" onClick={() => appointmentAction(appointment.id, 'cancel')}>Cancel</button>}{user?.role === 'officer' && appointment.status === 'requested' && <div className="appointment-actions"><button className="secondary-button" onClick={() => appointmentAction(appointment.id, 'accept')}>Accept request</button><button className="secondary-button" onClick={() => appointmentAction(appointment.id, 'decline')}>Decline</button></div>}</div>)}
        </article>

        <article className="workflow-card history-card">
          <p className="eyebrow">YOUR HISTORY</p><h3>{user ? `Saved for ${user.full_name}` : 'Sign in to see saved history.'}</h3>
          {user && <button className="secondary-button" onClick={reloadAccountData}>Refresh history</button>}
          {user && <>
            <details open><summary>Crop recommendations ({cropHistory.length})</summary>{cropHistory.map((item) => <div className="history-row" key={item.id}><strong>{item.result.predicted_crop}</strong><span>{new Date(item.created_at).toLocaleString()} · prediction #{item.id}</span></div>)}</details>
            <details><summary>Fertilizer recommendations ({fertilizerHistory.length})</summary>{fertilizerHistory.map((item) => <div className="history-row" key={item.id}><strong>{item.result.fertilizer}</strong><span>{new Date(item.created_at).toLocaleString()} · prediction #{item.id}</span></div>)}</details>
            <details><summary>Consultancy conversations ({conversations.length})</summary>{conversations.map((chat) => <div className="history-row" key={chat.conversation_id}><strong>{chat.messages.length} messages</strong><span>{chat.messages.slice(-2).map((message) => `${message.role}: ${message.content}`).join(' · ')}</span></div>)}</details>
          </>}
        </article>
      </div>
    </section>
  )
}

function DiseaseUnavailableCard() {
  return <section className="disease-unavailable" aria-label="Disease analysis unavailable"><p className="eyebrow">PLANT DISEASE ANALYSIS</p><h2>Unavailable in this build</h2><p>No project-owned disease dataset or trained disease model was supplied. SAARTHI will not analyze disease images until validated project assets are available.</p></section>
}

const rootContainer = document.getElementById('root')
// Reuse the root when Vite replaces this module during development.
const root = rootContainer._saarthiReactRoot || createRoot(rootContainer)
rootContainer._saarthiReactRoot = root
root.render(<App />)
