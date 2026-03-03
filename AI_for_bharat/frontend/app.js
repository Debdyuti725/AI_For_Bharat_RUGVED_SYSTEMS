/* ═══════════════════════════════════════════════════════════════
   Bharat Access Hub — Frontend JavaScript
   Handles: Auth, Routing, Profile (Tier 1 + Tier 2), Dashboard,
            Chat (TTS/STT), Schemes, Document Upload, Jobs
   ═══════════════════════════════════════════════════════════════ */

const API = '';

// ─── State ──────────────────────────────────────────────────
let state = {
    token: localStorage.getItem('bah_token') || null,
    user: JSON.parse(localStorage.getItem('bah_user') || 'null'),
    language: localStorage.getItem('bah_lang') || 'en',
    profile: null,
    eligibility: [],
    currentPage: 'dashboard',
};

// ─── API Client ─────────────────────────────────────────────
async function api(method, path, body = null) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
    const opts = { method, headers };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(API + path, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    return data;
}

async function apiFormData(path, formData) {
    const headers = {};
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
    const res = await fetch(API + path, { method: 'POST', headers, body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    return data;
}

// ─── Auth ───────────────────────────────────────────────────
function saveAuth(result) {
    state.token = result.token;
    state.user = { id: result.user_id, email: result.email, name: result.name };
    localStorage.setItem('bah_token', result.token);
    localStorage.setItem('bah_user', JSON.stringify(state.user));
}

function logout() {
    state.token = null; state.user = null; state.profile = null; state.eligibility = [];
    localStorage.removeItem('bah_token'); localStorage.removeItem('bah_user');
    showAuth();
}

function showAuth() {
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    document.getElementById('page-auth').classList.remove('hidden');
    document.getElementById('navbar').style.display = 'none';
}

function showApp() {
    document.getElementById('page-auth').classList.add('hidden');
    document.getElementById('navbar').style.display = 'flex';
    if (state.user) {
        document.getElementById('user-badge').textContent = state.user.name || state.user.email;
        document.getElementById('user-name-display').textContent = state.user.name || 'User';
        document.getElementById('logout-btn').style.display = 'inline-flex';
    }
    showPage('dashboard');
    // Check tier 2 reminder after page loads
    setTimeout(() => checkTier2Reminder(), 1500);
}

// Auth tabs
document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const isLogin = tab.dataset.tab === 'login';
        document.getElementById('login-form').style.display = isLogin ? 'block' : 'none';
        document.getElementById('signup-form').style.display = isLogin ? 'none' : 'block';
        document.getElementById('auth-error').textContent = '';
    });
});

document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const errEl = document.getElementById('auth-error');
    errEl.textContent = '';
    try {
        const result = await api('POST', '/api/auth/login', {
            email: document.getElementById('login-email').value,
            password: document.getElementById('login-password').value,
        });
        saveAuth(result); showApp();
    } catch (err) { errEl.textContent = err.message; }
});

document.getElementById('signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const errEl = document.getElementById('auth-error');
    errEl.textContent = '';
    try {
        const result = await api('POST', '/api/auth/signup', {
            email: document.getElementById('signup-email').value,
            password: document.getElementById('signup-password').value,
            name: document.getElementById('signup-name').value,
        });
        saveAuth(result); showApp();
    } catch (err) { errEl.textContent = err.message; }
});

document.getElementById('logout-btn').addEventListener('click', logout);

// ─── Page Routing ───────────────────────────────────────────
function showPage(page) {
    state.currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    document.getElementById('page-' + page).classList.remove('hidden');
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.toggle('active', l.dataset.page === page);
    });
    if (page === 'dashboard') loadDashboard();
    if (page === 'profile')   loadProfile();
    if (page === 'schemes')   loadSchemes();
    if (page === 'chat')      focusChat();
    if (page === 'jobs')      initJobsPage();
}
window.showPage = showPage;

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => { e.preventDefault(); showPage(link.dataset.page); });
});

const langSelect = document.getElementById('lang-select');
langSelect.value = state.language;
langSelect.addEventListener('change', (e) => {
    state.language = e.target.value;
    localStorage.setItem('bah_lang', state.language);
});

function getLang() {
    // Use in-page chat language selector if it exists and is visible, else navbar selector
    const chatLang = document.getElementById('chat-lang-select');
    if (chatLang) return chatLang.value || 'en';
    return document.getElementById('lang-select').value || 'en';
}


// ═══════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════
async function loadDashboard() {
    try {
        const profileData = await api('GET', '/api/profile');
        if (!profileData.profile || !profileData.tier1_complete) {
            document.getElementById('no-profile-banner').style.display = 'flex';
            document.getElementById('scheme-grid').innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:3rem;">Complete your profile to see personalized scheme matches.</p>';
            updateStats([]);
            return;
        }
        document.getElementById('no-profile-banner').style.display = 'none';
        state.profile = profileData.profile;
        const scores = await api('GET', '/api/eligibility');
        state.eligibility = scores;
        renderSchemeCards(scores);
        updateStats(scores);
    } catch (err) {
        if (err.message.includes('401') || err.message.includes('token')) logout();
    }
}

function updateStats(scores) {
    const total = scores.length;
    const high = scores.filter(s => s.eligibility_score >= 75).length;
    const benefit = scores.filter(s => s.eligibility_score >= 50).reduce((sum, s) => sum + s.benefit_amount, 0);
    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-high').textContent = high;
    document.getElementById('stat-benefit').textContent = '₹' + (benefit >= 100000 ? (benefit / 100000).toFixed(1) + 'L' : benefit.toLocaleString('en-IN'));
}

function renderSchemeCards(scores, filter = 'all') {
    const grid = document.getElementById('scheme-grid');
    const filtered = filter === 'all' ? scores : scores.filter(s => s.category === filter);
    if (filtered.length === 0) {
        grid.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:2rem;">No schemes in this category.</p>';
        return;
    }
    grid.innerHTML = filtered.map(s => {
        const score = s.eligibility_score;
        const badgeClass = score >= 75 ? 'badge-high' : score >= 50 ? 'badge-medium' : 'badge-low';
        const badgeText = score >= 75 ? 'HIGH MATCH' : score >= 50 ? 'PARTIAL' : 'LOW';
        const barColor = score >= 75 ? 'var(--success)' : score >= 50 ? 'var(--warning)' : 'var(--danger)';
        const benefit = s.benefit_amount >= 100000 ? '₹' + (s.benefit_amount / 100000).toFixed(1) + ' Lakh' : '₹' + s.benefit_amount.toLocaleString('en-IN');
        return `<div class="scheme-card" onclick="openSchemeDetail('${s.scheme_id}')">
            <div class="scheme-card-header"><h3>${s.scheme_name}</h3><span class="scheme-badge ${badgeClass}">${badgeText}</span></div>
            <div class="scheme-card-body">${s.reason}</div>
            <div class="scheme-card-footer"><span class="scheme-score" style="color:${barColor}">${score.toFixed(1)}%</span><span class="scheme-benefit">${benefit}</span></div>
            <div class="score-bar"><div class="score-bar-fill" style="width:${score}%;background:${barColor}"></div></div>
        </div>`;
    }).join('');
}

document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderSchemeCards(state.eligibility, btn.dataset.cat);
    });
});


// ═══════════════════════════════════════════════════════════════
// PROFILE
// ═══════════════════════════════════════════════════════════════
const TIER2_CATEGORIES = [
    { id: 'agriculture', icon: '🌾', label: 'Agriculture' },
    { id: 'education',   icon: '🎓', label: 'Education' },
    { id: 'housing',     icon: '🏠', label: 'Housing' },
    { id: 'employment',  icon: '💼', label: 'Employment' },
    { id: 'health',      icon: '🏥', label: 'Health' },
];

let currentQuestions = [], currentAnswers = {}, currentQuestionIdx = 0;
let profileMode = 'tier1';
let tier2Category = '';

async function loadProfile() {
    const dashboard          = document.getElementById('profile-dashboard');
    const questionnaireSection = document.getElementById('questionnaire-section');
    const noProfileState     = document.getElementById('no-profile-state');

    dashboard.style.display = 'none';
    questionnaireSection.style.display = 'none';
    noProfileState.style.display = 'none';

    try {
        const profileData = await api('GET', '/api/profile');
        if (profileData.profile && profileData.tier1_complete) {
            state.profile = profileData.profile;
            const completedCats = profileData.tier2_categories || [];
            renderProfileDashboard(profileData.profile, completedCats);
            dashboard.style.display = 'block';
            return;
        }
    } catch (err) {}

    noProfileState.style.display = 'block';
    document.getElementById('start-questionnaire-btn').onclick = startTier1;
}

function renderProfileDashboard(profile, completedCats) {
    const name = profile.name || state.user?.name || 'User';
    const email = state.user?.email || '';
    document.getElementById('profile-avatar').textContent = name.charAt(0).toUpperCase();
    document.getElementById('profile-name-display').textContent = name;
    document.getElementById('profile-email-display').textContent = email;

    document.getElementById('tier2-badge').textContent = `Tier 2: ${completedCats.length}/5`;
    if (completedCats.length > 0) document.getElementById('tier2-badge').classList.add('badge-complete');

    const capitalize = s => s ? s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ') : '—';
    const formatIncome = v => v ? '₹' + Number(v).toLocaleString('en-IN') : '—';
    const fields = [
        { label: 'Full Name',    value: profile.name || '—' },
        { label: 'Age',          value: profile.age || '—' },
        { label: 'Gender',       value: capitalize(profile.gender) },
        { label: 'State',        value: capitalize(profile.state) },
        { label: 'Area',         value: capitalize(profile.area_type) },
        { label: 'Category',     value: (profile.category || '—').toUpperCase() },
        { label: 'Education',    value: capitalize(profile.education_level) },
        { label: 'Employment',   value: capitalize(profile.employment_status) },
        { label: 'Annual Income',value: formatIncome(profile.annual_income) },
        { label: 'Family Size',  value: profile.family_size || '—' },
        { label: 'Owns Land',    value: profile.owns_land ? 'Yes ✓' : 'No' },
        { label: 'BPL Card',     value: profile.bpl_card ? 'Yes ✓' : 'No' },
    ];

    document.getElementById('profile-info-grid').innerHTML = fields.map(f =>
        `<div class="info-item"><div class="info-label">${f.label}</div><div class="info-value">${f.value}</div></div>`
    ).join('');

    document.getElementById('tier2-category-grid').innerHTML = TIER2_CATEGORIES.map(c => {
        const done = completedCats.includes(c.id);
        return `<div class="tier2-cat-card ${done ? 'completed' : ''}" onclick="${done ? '' : `startTier2('${c.id}')`}">
            <span class="tier2-cat-icon">${c.icon}</span>
            <div class="tier2-cat-name">${c.label}</div>
            <div class="tier2-cat-status">${done ? '10/10 Complete' : 'Not started'}</div>
        </div>`;
    }).join('');

    loadProfileSchemeMatches();
}

async function loadProfileSchemeMatches() {
    try {
        const scores = await api('GET', '/api/eligibility');
        state.eligibility = scores;
        const top5 = scores.slice(0, 5);
        if (top5.length > 0) {
            document.getElementById('profile-matches').style.display = 'block';
            document.getElementById('profile-top-schemes').innerHTML = top5.map(s => {
                const scoreClass = s.eligibility_score >= 75 ? 'high' : s.eligibility_score >= 50 ? 'medium' : 'low';
                return `<div class="match-item" onclick="openSchemeDetail('${s.scheme_id}')">
                    <span class="match-name">${s.scheme_name}</span>
                    <span class="match-score ${scoreClass}">${s.eligibility_score.toFixed(0)}%</span>
                </div>`;
            }).join('');
        }
    } catch (err) {}
}

function showQuestionnaire() {
    document.getElementById('profile-dashboard').style.display = 'none';
    document.getElementById('no-profile-state').style.display = 'none';
    document.getElementById('questionnaire-section').style.display = 'block';
}

async function startTier1() {
    profileMode = 'tier1';
    showQuestionnaire();
    try {
        currentQuestions = await api('GET', '/api/questions/tier1');
        currentAnswers = {}; currentQuestionIdx = 0;
        document.getElementById('profile-status').textContent = 'Tier 1: Basic Information';
        renderQuestion();
    } catch (err) {
        document.getElementById('questionnaire-container').innerHTML = '<p class="error-msg">Could not load questions.</p>';
    }
}

window.startTier2 = async function (category) {
    profileMode = 'tier2';
    tier2Category = category;
    showQuestionnaire();
    try {
        currentQuestions = await api('GET', `/api/questions/tier2/${category}`);
        currentAnswers = {}; currentQuestionIdx = 0;
        document.getElementById('profile-status').textContent = `Tier 2: ${category.charAt(0).toUpperCase() + category.slice(1)} Questions`;
        renderQuestion();
    } catch (err) {
        document.getElementById('questionnaire-container').innerHTML = `<p class="error-msg">Could not load ${category} questions: ${err.message}</p>`;
    }
};

window.resetProfile = function () { startTier1(); };

function renderQuestion() {
    const container = document.getElementById('questionnaire-container');
    const progressEl = document.getElementById('profile-progress');

    if (currentQuestionIdx >= currentQuestions.length) {
        if (profileMode === 'tier1') submitTier1();
        else submitTier2();
        return;
    }

    const q = currentQuestions[currentQuestionIdx];
    const pct = Math.round((currentQuestionIdx / currentQuestions.length) * 100);
    progressEl.style.width = pct + '%'; progressEl.innerHTML = `<span>${pct}%</span>`;

    const tierLabel = profileMode === 'tier1' ? 'Tier 1' : `Tier 2 (${tier2Category})`;
    document.getElementById('profile-status').textContent = `${tierLabel} — Question ${currentQuestionIdx + 1} of ${currentQuestions.length}`;

    let inputHTML = '';
    if (q.qtype === 'choice' && q.options) {
        inputHTML = `<div class="option-grid">${q.options.map(o =>
            `<button class="option-btn" onclick="selectOption(this,'${q.key}','${o.value}')">${o.label}</button>`
        ).join('')}</div>`;
    } else if (q.qtype === 'number') {
        inputHTML = `<div class="form-group"><input type="number" id="q-input" min="${q.validation?.min || 0}" max="${q.validation?.max || 99999999}" placeholder="Enter a number" class="search-input"></div>
        <button class="btn btn-primary" onclick="submitNumberAnswer('${q.key}')">Next →</button>`;
    } else if (q.qtype === 'bool') {
        inputHTML = `<div class="option-grid">
            <button class="option-btn" onclick="selectOption(this,'${q.key}',true)">Yes</button>
            <button class="option-btn" onclick="selectOption(this,'${q.key}',false)">No</button></div>`;
    } else {
        inputHTML = `<div class="form-group"><input type="text" id="q-input" placeholder="Your answer..." class="search-input"></div>
        <button class="btn btn-primary" onclick="submitTextAnswer('${q.key}')">Next →</button>`;
    }

    container.innerHTML = `<div class="question-card"><h3>${q.text}</h3>
        ${q.text_hi ? `<div class="q-hindi">${q.text_hi}</div>` : ''}
        ${q.help_text ? `<div class="q-hint">${q.help_text}</div>` : ''}${inputHTML}</div>`;
}

window.selectOption = function (btn, key, value) { currentAnswers[key] = value; currentQuestionIdx++; renderQuestion(); };
window.submitNumberAnswer = function (key) { const v = parseInt(document.getElementById('q-input').value); if (isNaN(v)) return; currentAnswers[key] = v; currentQuestionIdx++; renderQuestion(); };
window.submitTextAnswer = function (key) { const v = document.getElementById('q-input').value.trim(); if (!v) return; currentAnswers[key] = v; currentQuestionIdx++; renderQuestion(); };

async function submitTier1() {
    const container = document.getElementById('questionnaire-container');
    document.getElementById('profile-progress').style.width = '100%';
    container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">Saving your profile...</p>';
    try {
        const result = await api('POST', '/api/profile', currentAnswers);
        state.profile = result.profile;
        loadProfile();
    } catch (err) { container.innerHTML = `<p class="error-msg">Error: ${err.message}</p>`; }
}

async function submitTier2() {
    const container = document.getElementById('questionnaire-container');
    document.getElementById('profile-progress').style.width = '100%';
    container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:2rem;">Saving Tier 2 answers...</p>';
    try {
        const result = await api('POST', '/api/profile/extend', {
            category: tier2Category,
            answers: currentAnswers,
        });
        state.profile = result.profile;
        loadProfile();
    } catch (err) { container.innerHTML = `<p class="error-msg">Error: ${err.message}</p>`; }
}


// ═══════════════════════════════════════════════════════════════
// SCHEME EXPLORER
// ═══════════════════════════════════════════════════════════════
async function loadSchemes() {
    try { const schemes = await api('GET', '/api/schemes'); renderSchemeList(schemes); }
    catch (err) { document.getElementById('scheme-list').innerHTML = '<p class="error-msg">Could not load schemes.</p>'; }
}

function renderSchemeList(schemes) {
    document.getElementById('scheme-list').innerHTML = schemes.map(s => `
        <div class="scheme-list-item" onclick="openSchemeDetail('${s.scheme_id}')">
            <div><h3>${s.name}</h3><p>${(s.description || '').substring(0, 120)}${s.description && s.description.length > 120 ? '...' : ''}</p></div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.4rem;">
                <span class="scheme-cat-badge">${s.category}</span>
                <span style="color:var(--success);font-weight:600;font-size:0.85rem;">₹${s.benefit_amount ? s.benefit_amount.toLocaleString('en-IN') : '0'}</span>
            </div>
        </div>`).join('');
}

document.getElementById('search-btn').addEventListener('click', doSearch);
document.getElementById('scheme-search').addEventListener('keypress', (e) => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
    const query = document.getElementById('scheme-search').value.trim();
    if (!query) { loadSchemes(); return; }
    try {
        const results = await api('GET', `/api/schemes/search/${encodeURIComponent(query)}`);
        const list = document.getElementById('scheme-list');
        if (results.length === 0) { list.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No results. Try different keywords.</p>'; return; }
        const seen = new Set();
        const unique = results.filter(r => { if (seen.has(r.scheme_id)) return false; seen.add(r.scheme_id); return true; });
        list.innerHTML = unique.map(r => `
            <div class="scheme-list-item" onclick="openSchemeDetail('${r.scheme_id}')">
                <div><h3>${r.scheme_id}</h3><p>${r.content}</p></div>
                <span class="scheme-cat-badge">${r.section}</span>
            </div>`).join('');
    } catch (err) { document.getElementById('scheme-list').innerHTML = `<p class="error-msg">Search error: ${err.message}</p>`; }
}

window.openSchemeDetail = async function (schemeId) {
    try {
        const scheme = await api('GET', `/api/schemes/${schemeId}`);
        const modal = document.getElementById('scheme-modal');
        const docs = scheme.required_documents || [];
        document.getElementById('scheme-detail-content').innerHTML = `
            <h2>${scheme.name}</h2>
            <span class="scheme-cat-badge" style="margin-bottom:1rem;display:inline-block;">${scheme.category}</span>
            <div class="detail-section"><h3>Description</h3><p>${scheme.description || 'N/A'}</p></div>
            <div class="detail-section"><h3>Benefits</h3><p style="color:var(--success);font-size:1.1rem;font-weight:700;">₹${(scheme.benefit_amount || 0).toLocaleString('en-IN')}</p>
                <p style="color:var(--text-secondary);font-size:0.9rem;">${scheme.benefit_type || ''}</p></div>
            ${scheme.how_to_apply ? `<div class="detail-section"><h3>How to Apply</h3><p>${scheme.how_to_apply}</p></div>` : ''}
            ${docs.length > 0 ? `<div class="detail-section"><h3>Required Documents</h3><ul>${docs.map(d => `<li>${d}</li>`).join('')}</ul></div>` : ''}
            ${scheme.application_url ? `<div class="detail-section"><h3>Official Portal</h3><a href="${scheme.application_url}" target="_blank" class="btn btn-primary" style="margin-top:0.5rem;">Visit Website →</a></div>` : ''}`;
        modal.classList.remove('hidden');
    } catch (err) { alert('Could not load scheme details.'); }
};

document.getElementById('modal-close').addEventListener('click', () => document.getElementById('scheme-modal').classList.add('hidden'));
document.getElementById('scheme-modal').addEventListener('click', (e) => { if (e.target === e.currentTarget) e.target.classList.add('hidden'); });


// ═══════════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════════
function focusChat() { setTimeout(() => document.getElementById('chat-input').focus(), 100); }

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-input').addEventListener('keypress', (e) => { if (e.key === 'Enter') sendChat(); });

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addChatMessage('user', msg);
    addTypingIndicator();
    try {
        const lang = getLang();
        const result = await api('POST', '/api/chat', { message: msg, language: lang });
        removeTypingIndicator();
        addChatMessage('bot', result.response);
        const ttsToggle = document.getElementById('tts-toggle');
        if (ttsToggle && ttsToggle.checked) speakText(result.response, lang);
    } catch (err) {
        removeTypingIndicator();
        addChatMessage('bot', 'Sorry, error: ' + err.message);
    }
}

function addChatMessage(role, text) {
    const container = document.getElementById('chat-messages');
    const initials = state.user?.name ? state.user.name.charAt(0).toUpperCase() : 'U';
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    const formatted = text.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    if (role === 'bot') {
        div.innerHTML = `<div class="chat-avatar">🤖</div>
            <div class="chat-bubble">${formatted}</div>
            <button class="btn btn-outline btn-sm" onclick="speakThis(this)" style="align-self:flex-start;margin-top:0.25rem;" title="Read aloud">🔊</button>`;
    } else {
        div.innerHTML = `<div class="chat-avatar">${initials}</div><div class="chat-bubble">${formatted}</div>`;
    }
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-message bot'; div.id = 'typing-indicator';
    div.innerHTML = `<div class="chat-avatar">🤖</div><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    container.appendChild(div); container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() { const el = document.getElementById('typing-indicator'); if (el) el.remove(); }

// ─── TTS ─────────────────────────────────────────────────────
let currentAudio = null;
let ttsAbortController = null;
let ttsLoading = false;

async function speakText(text, langCode) {
    if (ttsAbortController) { ttsAbortController.abort(); ttsAbortController = null; }
    if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; currentAudio = null; }

    const cleaned = text.replace(/<[^>]*>/g, '').replace(/\*\*/g, '').replace(/•/g, '')
        .replace(/\n+/g, ' ').trim().substring(0, 300);
    if (!cleaned) return;

    const lang = langCode || getLang();
    ttsLoading = true;
    ttsAbortController = new AbortController();

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
        const res = await fetch('/api/tts', {
            method: 'POST', headers,
            body: JSON.stringify({ text: cleaned, language: lang }),
            signal: ttsAbortController.signal,
        });
        if (!res.ok) { ttsLoading = false; return; }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudio = audio;
        ttsLoading = false;
        audio.onended = () => { URL.revokeObjectURL(url); if (currentAudio === audio) currentAudio = null; };
        audio.onerror = () => { URL.revokeObjectURL(url); if (currentAudio === audio) currentAudio = null; };
        audio.play().catch(() => { URL.revokeObjectURL(url); if (currentAudio === audio) currentAudio = null; });
    } catch (err) {
        ttsLoading = false;
        if (err.name !== 'AbortError') console.error('TTS error:', err);
    }
}
window.speakText = speakText;

window.speakThis = function (btn) {
    const bubble = btn.previousElementSibling;
    if (!bubble) return;
    if (ttsLoading) {
        if (ttsAbortController) ttsAbortController.abort();
        if (currentAudio) { currentAudio.pause(); currentAudio = null; }
        ttsLoading = false; btn.textContent = '🔊'; return;
    }
    btn.textContent = '⏳';
    speakText(bubble.innerText, getLang()).finally(() => { btn.textContent = '🔊'; });
};

document.getElementById('tts-toggle').addEventListener('change', (e) => {
    if (!e.target.checked && currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; currentAudio = null; }
});

// ─── STT ─────────────────────────────────────────────────────
const STT_LANG_MAP = {
    en: 'en-IN', hi: 'hi-IN', kn: 'kn-IN', te: 'te-IN', ta: 'ta-IN',
    bn: 'bn-IN', mr: 'mr-IN', gu: 'gu-IN', ml: 'ml-IN', pa: 'pa-IN',
};
let recognition = null;
let isRecording = false;

function initSTT() {
    const voiceBtn = document.getElementById('voice-input-btn');
    const stopBtn  = document.getElementById('voice-stop-btn');
    if (!voiceBtn) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceBtn.title = 'Speech recognition requires Chrome or Edge';
        voiceBtn.style.opacity = '0.4';
        voiceBtn.style.cursor = 'not-allowed';
        voiceBtn.addEventListener('click', () =>
            alert('Speech recognition requires Chrome or Edge browser.')
        );
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    function setRecordingState(active) {
        isRecording = active;
        voiceBtn.style.display  = active ? 'none'         : 'inline-flex';
        if (stopBtn) stopBtn.style.display = active ? 'inline-flex' : 'none';
    }

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i++) transcript += event.results[i][0].transcript;
        document.getElementById('chat-input').value = transcript;
    };
    recognition.onend = () => setRecordingState(false);
    recognition.onerror = () => setRecordingState(false);

    voiceBtn.addEventListener('click', () => {
        recognition.lang = STT_LANG_MAP[getLang()] || 'en-IN';
        setRecordingState(true);
        recognition.start();
    });

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            recognition.stop();
            setRecordingState(false);
        });
    }

    // Sync chat-lang-select with navbar lang-select as default
    const chatLangSel = document.getElementById('chat-lang-select');
    const navLangSel  = document.getElementById('lang-select');
    if (chatLangSel && navLangSel) {
        chatLangSel.value = navLangSel.value;
        chatLangSel.addEventListener('change', (e) => {
            state.language = e.target.value;
            localStorage.setItem('bah_lang', e.target.value);
            navLangSel.value = e.target.value;
        });
    }
}

initSTT();


// ═══════════════════════════════════════════════════════════════
// DOCUMENT UPLOAD
// ═══════════════════════════════════════════════════════════════
const uploadArea = document.getElementById('upload-area');
const fileInput  = document.getElementById('file-input');

if (uploadArea && fileInput) {
    document.getElementById('upload-btn').addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', (e) => { if (e.target.id !== 'upload-btn') fileInput.click(); });
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault(); uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]); });
}

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) { alert('Only PDF files are supported.'); return; }
    if (file.size > 10 * 1024 * 1024) { alert('File too large. Max 10MB.'); return; }

    uploadArea.style.display = 'none';
    document.getElementById('doc-result').style.display = 'none';
    document.getElementById('doc-loading').style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', getLang());
        const result = await apiFormData('/api/document/upload', formData);
        document.getElementById('doc-loading').style.display = 'none';

        if (result.status === 'error') {
            alert(result.message || 'Could not process.');
            uploadArea.style.display = 'block'; return;
        }

        const analysis = result.analysis || {};
        document.getElementById('doc-filename').textContent = result.filename;
        document.getElementById('doc-summary').textContent = analysis.summary || analysis.raw_explanation || 'No summary.';

        const showList = (sectionId, listId, items) => {
            const section = document.getElementById(sectionId);
            if (items && items.length > 0) {
                section.style.display = 'block';
                document.getElementById(listId).innerHTML = items.map(f => `<li>${f}</li>`).join('');
            } else { section.style.display = 'none'; }
        };
        showList('doc-fields-section', 'doc-fields', analysis.key_fields);
        showList('doc-actions-section', 'doc-actions', analysis.actions_required);
        showList('doc-docs-section', 'doc-docs-needed', analysis.documents_needed);

        document.getElementById('doc-raw-text').textContent = result.extracted_text || '';
        document.getElementById('doc-result').style.display = 'block';

    } catch (err) {
        document.getElementById('doc-loading').style.display = 'none';
        uploadArea.style.display = 'block';
        alert('Upload failed: ' + err.message);
    }
}

document.getElementById('doc-tts-btn')?.addEventListener('click', () => {
    speakText(document.getElementById('doc-summary').textContent, getLang());
});

document.getElementById('upload-another')?.addEventListener('click', () => {
    document.getElementById('doc-result').style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = '';
});


// ═══════════════════════════════════════════════════════════════
// JOBS PAGE
// ═══════════════════════════════════════════════════════════════
let jobsPage = 0;
const JOBS_LIMIT = 15;
let jobsTotal = 0;

// Keywords in job titles that indicate non-job listings
const NON_JOB_TITLE_WORDS = [
    'mobile', 'phone', 'iphone', 'samsung', 'redmi', 'oppo', 'vivo', 'realme', 'oneplus',
    'laptop', 'computer', 'tablet', 'tv', 'television', 'fridge', 'refrigerator',
    'washing machine', 'ac ', 'air conditioner', 'microwave', 'mixer', 'grinder',
    'utensil', 'vessel', 'cooker', 'pressure cooker', 'pan ', 'kadai', 'tawa',
    'sofa', 'bed ', 'chair', 'table', 'cupboard', 'almirah', 'wardrobe',
    'car ', 'bike ', 'scooter', 'activa', 'plot', 'flat ', 'apartment', 'house for',
    'land ', 'property', 'pg ', 'room for rent', 'pg for',
    'puppy', 'dog ', 'cat ', 'kitten', 'pet ',
    'for sale', 'sell ', 'selling', 'second hand', 'used ',
];

// OLX job category codes in URLs (c4=jobs, c22xx=job subcategories)
function isJobListing(job) {
    const title = (job.title || '').toLowerCase();
    const url   = (job.url   || '').toLowerCase();

    // Filter by title — reject if it contains non-job keywords
    if (NON_JOB_TITLE_WORDS.some(w => title.includes(w))) return false;

    // Filter by URL category code
    const codeMatch = url.match(/-c(\d+)-/);
    if (codeMatch) {
        const code = parseInt(codeMatch[1]);
        // OLX jobs: c4 (parent) and c2201–c2299 (subcategories)
        // Reject known non-job ranges
        if (code < 4) return false;                    // c1=vehicles, c2=property, c3=electronics
        if (code > 4 && code < 2200) return false;     // misc non-job categories
        if (code > 2299) return false;                  // outside job subcategory range
    }

    return true;
}

async function loadJobs(page = 0) {
    jobsPage = page;
    const offset = page * JOBS_LIMIT;

    const keyword  = document.getElementById('job-search')?.value.trim() || '';
    const city     = document.getElementById('job-city-filter')?.value || '';
    const category = document.getElementById('job-cat-filter')?.value || '';

    const params = new URLSearchParams({ limit: JOBS_LIMIT, offset });
    if (keyword)  params.set('keyword', keyword);
    if (city)     params.set('city', city);
    if (category) params.set('category', category);

    const list       = document.getElementById('job-list');
    const stats      = document.getElementById('jobs-stats');
    const pagination = document.getElementById('jobs-pagination');
    const pageInfo   = document.getElementById('jobs-page-info');
    const prevBtn    = document.getElementById('jobs-prev-btn');
    const nextBtn    = document.getElementById('jobs-next-btn');

    list.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted);">Loading...</div>';

    try {
        const res = await fetch(`/api/jobs?${params}`, {
            headers: state.token ? { 'Authorization': `Bearer ${state.token}` } : {}
        });
        const data = await res.json();
        jobsTotal = data.total || 0;

        // Filter out non-job listings on the frontend
        const jobs = (data.jobs || []).filter(isJobListing);

        list.innerHTML = '';

        if (jobs.length === 0) {
            list.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--text-muted);">No jobs found. Try different filters.</div>';
            if (stats)      stats.textContent = '0 jobs found';
            if (pagination) pagination.style.display = 'none';
            return;
        }

        jobs.forEach(job => {
            const card = document.createElement('a');
            card.href   = job.url || '#';
            card.target = '_blank';
            card.rel    = 'noopener noreferrer';
            card.className = 'job-card';
            card.innerHTML = `
                <div style="flex:1;min-width:0;">
                    <div class="job-title">${job.title}</div>
                    <div class="job-meta">
                        <span>📍 ${job.location || job.city || '—'}</span>
                        <span>🗓 ${job.date_posted || '—'}</span>
                        <span style="background:var(--bg-elevated);border:1px solid var(--border-subtle);padding:0.15rem 0.5rem;border-radius:100px;font-size:0.65rem;">OLX</span>
                    </div>
                </div>
                <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.4rem;flex-shrink:0;">
                    ${job.salary && job.salary !== 'Not specified' ? `<div style="font-size:0.85rem;font-weight:700;color:var(--success);">${job.salary}</div>` : ''}
                    <div style="padding:0.18rem 0.55rem;border-radius:100px;font-size:0.65rem;background:var(--green-50);color:var(--green-700);border:1px solid rgba(22,163,74,0.18);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap;">${job.category || 'General'}</div>
                </div>`;
            list.appendChild(card);
        });

        const totalPages = Math.max(1, Math.ceil(jobsTotal / JOBS_LIMIT));
        if (stats)    stats.textContent = `Showing ${offset + 1}–${Math.min(offset + jobs.length, jobsTotal)} of ${jobsTotal} jobs`;
        if (pageInfo) pageInfo.textContent = `Page ${page + 1} of ${totalPages}`;

        if (pagination) pagination.style.display = totalPages > 1 ? 'block' : 'none';
        if (prevBtn) prevBtn.disabled = page === 0;
        if (nextBtn) nextBtn.disabled = page >= totalPages - 1;

        // Scroll list into view
        document.getElementById('page-jobs')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (e) {
        console.error('Jobs load error:', e);
        list.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--text-muted);">Failed to load jobs. Make sure the server is running.</div>';
    }
}

async function initJobsPage() {
    const citySelect = document.getElementById('job-city-filter');
    const catSelect  = document.getElementById('job-cat-filter');

    citySelect.innerHTML = '<option value="">All Cities</option>';
    catSelect.innerHTML  = '<option value="">All Categories</option>';

    try {
        const cities = await api('GET', '/api/jobs/cities');
        cities.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.city;
            opt.textContent = `${c.city.charAt(0).toUpperCase() + c.city.slice(1)} (${c.count})`;
            citySelect.appendChild(opt);
        });
    } catch {}

    try {
        const cats = await api('GET', '/api/jobs/categories');
        cats.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.category;
            opt.textContent = `${c.category} (${c.count})`;
            catSelect.appendChild(opt);
        });
    } catch {}

    await loadJobs(0);
}

// Bind jobs page button listeners once at startup (not inside initJobsPage)
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('job-search-btn')?.addEventListener('click', () => loadJobs(0));
    document.getElementById('job-search')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') loadJobs(0);
    });
    document.getElementById('jobs-prev-btn')?.addEventListener('click', () => {
        if (jobsPage > 0) loadJobs(jobsPage - 1);
    });
    document.getElementById('jobs-next-btn')?.addEventListener('click', () => {
        loadJobs(jobsPage + 1);
    });
});

// ═══════════════════════════════════════════════════════════════
// TIER 2 REMINDER POPUP
// ═══════════════════════════════════════════════════════════════
async function checkTier2Reminder() {
    if (!state.token) return;

    // Don't show if dismissed within the last hour
    const lastDismissed = localStorage.getItem('tier2_reminder_dismissed');
    if (lastDismissed && Date.now() - parseInt(lastDismissed) < 3600000) return;

    try {
        const profileData = await api('GET', '/api/profile');
        if (!profileData.tier1_complete || !profileData.profile) return;

        const completedCats = profileData.tier2_categories || [];
        const allCats = ['agriculture', 'education', 'health', 'housing', 'employment'];
        const missing = allCats.filter(c => !completedCats.includes(c));
        if (missing.length === 0) return;

        const catLabels = {
            agriculture: '🌾 Agriculture & Farming',
            education:   '📚 Education & Scholarships',
            health:      '🏥 Health & Wellness',
            housing:     '🏠 Housing & Infrastructure',
            employment:  '💼 Employment & Training',
        };

        document.getElementById('tier2-reminder-cats').innerHTML = missing.map(c => `
            <div style="display:flex;align-items:center;gap:0.6rem;background:var(--bg-elevated);border:1px solid var(--border-subtle);border-radius:var(--radius-sm);padding:0.5rem 0.75rem;font-size:0.85rem;">
                <span style="color:var(--warning);">○</span>
                <span>${catLabels[c] || c}</span>
            </div>`).join('');

        document.getElementById('tier2-reminder-modal').classList.remove('hidden');

        document.getElementById('tier2-go-btn').onclick = () => {
            document.getElementById('tier2-reminder-modal').classList.add('hidden');
            showPage('profile');
        };
        document.getElementById('tier2-later-btn').onclick = dismissTier2Reminder;

    } catch {}
}

function dismissTier2Reminder() {
    localStorage.setItem('tier2_reminder_dismissed', Date.now().toString());
    document.getElementById('tier2-reminder-modal').classList.add('hidden');
}
window.dismissTier2Reminder = dismissTier2Reminder;


// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
(function init() {
    if (state.token && state.user) showApp();
    else showAuth();
})();
