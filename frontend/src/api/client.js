const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
let csrfToken = ''

async function ensureCsrfToken() {
  if (csrfToken) return csrfToken
  const response = await fetch(`${apiBaseUrl}/api/auth/csrf`, { headers: { Accept: 'application/json' }, cache: 'no-store' })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw apiError(body, response.status)
  csrfToken = body.csrf_token || ''
  return csrfToken
}

function saveCsrfToken(body) {
  if (body?.csrf_token) csrfToken = body.csrf_token
}

function apiError(body, status) {
  const details = body.details && typeof body.details === 'object'
    ? Object.entries(body.details).map(([field, message]) => `${field}: ${message}`).join('; ')
    : ''
  return new Error(details || body.message || `Backend returned HTTP ${status}`)
}

async function postJson(url, payload, method = 'POST') {
  const token = await ensureCsrfToken()
  const response = await fetch(`${apiBaseUrl}${url}`, {
    method,
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': token },
    body: JSON.stringify(payload),
  })
  const body = await response.json().catch(() => ({}))
  saveCsrfToken(body)
  if (!response.ok) throw apiError(body, response.status)
  return body
}

async function getJson(url) {
  const response = await fetch(`${apiBaseUrl}${url}`, { headers: { Accept: 'application/json' }, cache: 'no-store' })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw apiError(body, response.status)
  return body
}

export async function getSession() {
  await ensureCsrfToken()
  return getJson('/api/users/me')
}

export async function registerFarmer(values) {
  return postJson('/api/auth/register', values)
}

export async function login(values) {
  return postJson('/api/auth/login', values)
}

export async function logout() {
  return postJson('/api/auth/logout', {})
}

export async function getHealth() {
  const response = await fetch(`${apiBaseUrl}/health`, {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  })

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body.message || `Backend returned HTTP ${response.status}`)
  }

  return body
}

export async function predictCrop(values) {
  return postJson('/api/crop/predict', values)
}

export async function getFertilizerSchema() {
  const response = await fetch(`${apiBaseUrl}/api/fertilizer/schema`, {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw apiError(body, response.status)
  return body
}

export async function predictFertilizer(values) {
  return postJson('/api/fertilizer/predict', values)
}

export async function sendConsultancyMessage(payload) {
  return postJson('/api/agent/chat', payload)
}

export async function getCropHistory() { return getJson('/api/crop/predictions') }
export async function getFertilizerHistory() { return getJson('/api/fertilizer/predictions') }
export async function getAgentHistory() { return getJson('/api/agent/history') }
export async function getSchemeStatus() { return getJson('/api/schemes/status') }
export async function getSchemeDocuments() { return getJson('/api/schemes/documents') }
export async function analyzeSchemeDocument(document_id) { return postJson('/api/schemes/analyze', { document_id }) }
export async function getAppointments() { return getJson('/api/appointments') }
export async function updateAppointment(id, action) { return postJson(`/api/appointments/${id}`, { action }, 'PATCH') }
export async function createAppointment(values) { return postJson('/api/appointments', values) }

export async function uploadSchemeDocument(file) {
  const token = await ensureCsrfToken()
  const form = new FormData()
  form.append('document', file)
  const response = await fetch(`${apiBaseUrl}/api/schemes/documents`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'X-CSRF-Token': token },
    body: form,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw apiError(body, response.status)
  return body
}
