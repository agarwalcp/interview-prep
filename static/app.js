(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const ROLE_META = {
  'ML Engineer': ['🤖', 'Builds and ships models: deep learning, NLP, vision and LLMs.'],
  'Data Scientist': ['📊', 'Statistics, model evaluation and feature engineering.'],
  'Frontend Developer': ['🎨', 'HTML, CSS and JavaScript for interactive, fast web interfaces.'],
  'Backend Developer': ['🛠️', 'Databases, architecture, security and concurrency.'],
  'Software Developer': ['💻', 'Data structures, algorithms and complexity.'],
};
const VIEWS = ['welcome', 'role', 'config', 'devices', 'review', 'interview', 'results'];
const STEPS = ['role', 'config', 'devices', 'review'];
const S = { role: null, opts: null, itype: 'Mixed', diff: 'Intermediate', nTech: 6, nHr: 2, name: '', mode: 'typed',
            sid: null, questions: [], idx: 0, records: [], startedAt: 0, timer: null, submitting: false, config: null,
            transcriptFinal: '', transcriptInterim: '', recog: null, listening: false, recMs: 0, recStart: 0, firstAt: null, shownAt: 0,
            viewingHistory: false, lastReport: null, voice: false };
const Dev = { cam: { state: 'idle', stream: null }, mic: { state: 'idle', stream: null, ctx: null, raf: null, recorder: null, url: null } };

/* ------------------------------------------------------------ voice interviewer (browser text-to-speech) */
const TTS = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
function pickVoice() { const v = window.speechSynthesis.getVoices() || []; return v.find(x => /en-IN/i.test(x.lang)) || v.find(x => /^en/i.test(x.lang)) || null; }
function setSpeakStatus(t) { const el = $('#speakStatus'); if (el) el.textContent = t || ''; }
function speak(text, onend) {
  if (!TTS) { if (onend) onend(); return; }
  window.speechSynthesis.cancel();
  const u = new window.SpeechSynthesisUtterance(text), v = pickVoice();
  if (v) { u.voice = v; u.lang = v.lang; } else u.lang = 'en-US';
  u.rate = 0.95;
  u.onstart = () => setSpeakStatus('Interviewer is speaking…');
  u.onend = () => { setSpeakStatus(''); if (onend) onend(); };
  u.onerror = () => { setSpeakStatus(''); };
  window.speechSynthesis.speak(u);
}
function stopSpeaking() { if (TTS) window.speechSynthesis.cancel(); setSpeakStatus(''); }
function speakCurrentQuestion() {
  if (!S.voice || !S.questions[S.idx]) return;
  const mine = S.idx;
  speak(`Question ${S.idx + 1}. ${S.questions[S.idx].question}`, () => {
    // spoken mode: start listening only after the interviewer has finished (so the mic does not hear the voice)
    if (S.mode === 'spoken' && S.idx === mine && !S.submitting && !S.listening && !$('#view-interview').hidden) { if (S.firstAt === null) S.firstAt = performance.now(); startRecognition(); }
  });
}

/* ------------------------------------------------------------ helpers */
async function api(path, body) {
  const opt = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `Server error (${r.status})`);
  return data;
}
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 5000); }
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const nice = s => String(s || '').replace(/_/g, ' ');
const niceH = s => esc(nice(s));
const num = x => (x === null || x === undefined || Number.isNaN(x) ? null : Number(x));
const sc = x => (num(x) === null ? '—' : String(Math.round(x)));
const pct = x => `${Math.round(x * 100)}%`;
const barClass = v => (v < 40 ? 'r' : v < 60 ? 'a' : 'g');
const beliefColor = b => (b < 0.4 ? '#ef4444' : b < 0.6 ? '#f59e0b' : '#22c55e');
const mmss = ms => { const s = Math.floor(ms / 1000); return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };

function show(view) {
  VIEWS.forEach(v => ($(`#view-${v}`).hidden = v !== view));
  const st = $('#stepper'), i = STEPS.indexOf(view);
  st.hidden = i < 0;
  st.querySelectorAll('span').forEach((el, k) => { el.className = k < i ? 'done' : k === i ? 'cur' : ''; });
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------ skill graph (SVG) */
function drawGraph(svg, g) {
  const NS = 'http://www.w3.org/2000/svg', mk = (t, a = {}) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const parents = {}; g.nodes.forEach(n => (parents[n.id] = []));
  g.edges.forEach(([p, c]) => parents[c].push(p));
  const depth = {}; const d = id => (depth[id] ??= parents[id].length ? 1 + Math.max(...parents[id].map(d)) : 0);
  g.nodes.forEach(n => d(n.id));
  const cols = Math.max(...Object.values(depth)) + 1, byDepth = {};
  g.nodes.forEach(n => (byDepth[depth[n.id]] ??= []).push(n));
  const rows = Math.max(...Object.values(byDepth).map(a => a.length)), W = 640, R = 27, H = Math.max(200, rows * (R * 2 + 44));
  const pos = {};
  for (const [dp, arr] of Object.entries(byDepth)) arr.forEach((n, i) => (pos[n.id] = { x: ((+dp + 0.5) * W) / cols, y: ((i + 1) * H) / (arr.length + 1) }));
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
  const defs = mk('defs'), mark = mk('marker', { id: 'arrR', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' });
  mark.appendChild(mk('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#4b5a95' })); defs.appendChild(mark); svg.appendChild(defs);
  g.edges.forEach(([p, c]) => {
    const a = pos[p], b = pos[c], dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    svg.appendChild(mk('line', { x1: a.x + (dx / L) * R, y1: a.y + (dy / L) * R, x2: b.x - (dx / L) * (R + 5), y2: b.y - (dy / L) * (R + 5), stroke: '#4b5a95', 'stroke-width': 1.6, 'marker-end': 'url(#arrR)' }));
  });
  g.nodes.forEach(n => {
    const { x, y } = pos[n.id], col = beliefColor(n.belief), grp = mk('g');
    const tip = mk('title'); tip.textContent = `${nice(n.id)}: ${pct(n.belief)} (${n.asked ? 'asked directly' : 'inferred from related skills'})`; grp.appendChild(tip);
    grp.appendChild(mk('circle', { cx: x, cy: y, r: R, fill: col + '33', stroke: col, 'stroke-width': n.asked ? 3 : 2, 'stroke-dasharray': n.asked ? '' : '4 3' }));
    const t = mk('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-weight': 700 }); t.textContent = pct(n.belief); grp.appendChild(t);
    const l = mk('text', { x, y: y + R + 14, 'text-anchor': 'middle' }); l.textContent = nice(n.id); l.style.fontSize = '11.5px'; grp.appendChild(l);
    svg.appendChild(grp);
  });
}

/* ------------------------------------------------------------ 1. welcome + history */
async function loadWelcome() {
  show('welcome');
  const box = $('#historyList'), stats = $('#statsRow');
  try {
    const { items } = await api('/api/history');
    if (!items.length) { box.textContent = 'No previous interviews yet. Your finished interviews will appear here.'; stats.hidden = true; return; }
    const scores = items.map(i => num(i.overall_score)).filter(x => x !== null);
    stats.hidden = false;
    stats.innerHTML = `<div class="stat"><b>${items.length}</b><span>interviews completed</span></div>` +
      (scores.length ? `<div class="stat"><b>${sc(scores.reduce((a, b) => a + b, 0) / scores.length)}</b><span>average overall score /100</span></div><div class="stat"><b>${sc(Math.max(...scores))}</b><span>best overall score /100</span></div>` : '');
    box.innerHTML = items.slice(0, 10).map(i => `<div class="hist"><span>${esc(i.saved_at)}</span><span><b>${esc(i.role)}</b> · ${esc(i.interview_type)} · ${esc(i.difficulty)}${i.candidate_name ? ' · ' + esc(i.candidate_name) : ''}</span><span><b>${sc(i.overall_score)}</b>/100 <small class="muted">(${i.answered}/${i.total})</small></span><button class="btn small-btn" type="button" data-hid="${esc(i.id)}">View</button></div>`).join('');
    box.querySelectorAll('[data-hid]').forEach(b => (b.onclick = () => openHistory(b.dataset.hid)));
  } catch (e) { box.textContent = 'Could not load history: ' + e.message; stats.hidden = true; }
}
async function openHistory(id) {
  try { const e = await api(`/api/history/${id}`); S.viewingHistory = true; S.sid = null; S.config = e.report.config; renderResults(e.report); show('results'); }
  catch (e) { toast(e.message); }
}

/* ------------------------------------------------------------ 2. role */
async function initRoles() {
  const [info, opts] = await Promise.all([api('/api/roles'), api('/api/options')]);
  S.opts = opts; S.roleInfo = Object.fromEntries(info.roles.map(r => [r.name, r]));
  const box = $('#roles'); box.innerHTML = '';
  info.roles.forEach(r => {
    const [ic, desc] = ROLE_META[r.name] || ['🎓', ''];
    const b = document.createElement('button'); b.className = 'role'; b.type = 'button'; b.setAttribute('aria-pressed', 'false'); b.dataset.role = r.name;
    b.innerHTML = `<div class="ic" aria-hidden="true">${ic}</div><b>${esc(r.name)}</b><span>${esc(desc)}</span><div class="skills">${r.skill_areas.slice(0, 6).map(s => `<i>${esc(s)}</i>`).join('')}</div>`;
    b.onclick = () => selectRole(r.name);
    box.appendChild(b);
  });
  buildSegments();
}
function selectRole(name) {
  S.role = name;
  document.querySelectorAll('#roles .role').forEach(x => { const on = x.dataset.role === name; x.classList.toggle('sel', on); x.setAttribute('aria-pressed', on); });
  $('#roleNext').disabled = false;
}

/* ------------------------------------------------------------ 3. configure */
function buildSegments() {
  const mk = (id, name, values, cur, onchange) => {
    $(id).innerHTML = values.map(v => `<label class="segopt"><input type="radio" name="${name}" value="${esc(v)}" ${v === cur ? 'checked' : ''}><span>${esc(v)}</span></label>`).join('');
    $(id).querySelectorAll('input').forEach(i => (i.onchange = () => onchange(i.value)));
  };
  mk('#typeSeg', 'itype', S.opts.interview_types, S.itype, v => { S.itype = v; syncConfig(); });
  mk('#diffSeg', 'diff', S.opts.difficulties, S.diff, v => { S.diff = v; syncConfig(); });
}
function configErrors() {
  const e = [];
  if (S.itype !== 'HR/Behavioral' && S.nTech < 1) e.push('Choose at least one technical question.');
  if (S.itype !== 'Technical' && S.nHr < 1) e.push('Choose at least one HR question.');
  if (S.name.length > 40) e.push('Name must be 40 characters or fewer.');
  return e;
}
function syncConfig() {
  S.nTech = +$('#nTech').value; S.nHr = +$('#nHr').value; S.name = $('#candName').value.trim();
  $('#nTechVal').textContent = S.nTech; $('#nHrVal').textContent = S.nHr;
  $('#lblTech').hidden = S.itype === 'HR/Behavioral'; $('#lblHr').hidden = S.itype === 'Technical';
  const nt = S.itype === 'HR/Behavioral' ? 0 : S.nTech, nh = S.itype === 'Technical' ? 0 : S.nHr;
  $('#estMin').textContent = `about ${nt * S.opts.minutes_per_tech + nh * S.opts.minutes_per_hr} minutes (${nt} technical + ${nh} HR; a planning estimate, not a time limit)`;
  const errs = configErrors(), box = $('#cfgErrors');
  box.hidden = !errs.length; box.textContent = errs.join(' '); $('#cfgNext').disabled = errs.length > 0;
}
function counts() { return { n_tech: S.itype === 'HR/Behavioral' ? 0 : S.nTech, n_hr: S.itype === 'Technical' ? 0 : S.nHr }; }

/* ------------------------------------------------------------ 4. devices */
function friendlyMediaError(e) {
  const hints = { NotAllowedError: 'Permission was blocked. Click the camera/lock icon in the address bar, choose Allow, then press the button again.',
    NotFoundError: 'No camera or microphone was found on this computer.', NotReadableError: 'The device is being used by another app (Zoom, Teams, camera app). Close it and retry.',
    OverconstrainedError: 'The device does not support the requested settings.', SecurityError: 'The browser blocked access. Use http://localhost in Edge or Chrome.' };
  return hints[e && e.name] || `Could not start the device (${(e && (e.name || e.message)) || 'unknown error'}).`;
}
function setBadge(id, text, kind) { const b = $(id); b.textContent = text; b.className = 'badge ' + (kind || ''); }
function sttSupported() { return !!(window.SpeechRecognition || window.webkitSpeechRecognition); }
function stopStream(s) { if (s) s.getTracks().forEach(t => t.stop()); }

async function startCamera() {
  stopStream(Dev.cam.stream); Dev.cam = { state: 'idle', stream: null };
  $('#camStatus').textContent = 'Asking the browser for permission… choose Allow.'; $('#camStatus').className = 'status';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { Dev.cam.state = 'error'; setBadge('#camBadge', 'Unsupported', 'bad'); $('#camStatus').textContent = 'This browser cannot access the camera. Use Edge or Chrome on http://localhost.'; $('#camStatus').className = 'status bad'; return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    Dev.cam.stream = stream;
    const v = $('#devVideo'); v.srcObject = stream; v.play().catch(() => {});
    setBadge('#camBadge', 'Connecting…', 'warn');
    const confirmed = await new Promise(res => { const t = setTimeout(() => res(false), 5000); v.onloadeddata = () => { clearTimeout(t); res(true); }; if (v.readyState >= 2) { clearTimeout(t); res(true); } });
    if (confirmed) { Dev.cam.state = 'ok'; $('#devVideo').parentElement.classList.add('on'); setBadge('#camBadge', 'Working', 'ok'); $('#camStatus').textContent = 'Camera is delivering video. Nothing is recorded or uploaded.'; $('#camStatus').className = 'status ok'; }
    else { Dev.cam.state = 'error'; stopStream(stream); setBadge('#camBadge', 'No video', 'bad'); $('#camStatus').textContent = 'The camera opened but sent no video frames. Check it is not covered or in use, then retry.'; $('#camStatus').className = 'status bad'; }
  } catch (e) { Dev.cam.state = 'error'; setBadge('#camBadge', 'Error', 'bad'); $('#camStatus').textContent = friendlyMediaError(e); $('#camStatus').className = 'status bad'; }
  $('#btnCam').textContent = 'Retry Camera'; updateDevNext();
}

function stopMicMeter() { if (Dev.mic.raf) cancelAnimationFrame(Dev.mic.raf); Dev.mic.raf = null; if (Dev.mic.ctx) { Dev.mic.ctx.close().catch(() => {}); Dev.mic.ctx = null; } $('#micLevel').style.width = '0'; }
async function startMic() {
  stopMicMeter(); stopStream(Dev.mic.stream); Dev.mic.state = 'idle'; Dev.mic.stream = null; $('#btnRecTest').disabled = true;
  $('#micStatus').textContent = 'Asking the browser for permission… choose Allow.'; $('#micStatus').className = 'status';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { Dev.mic.state = 'error'; setBadge('#micBadge', 'Unsupported', 'bad'); $('#micStatus').textContent = 'This browser cannot access the microphone. Use Edge or Chrome on http://localhost.'; $('#micStatus').className = 'status bad'; updateDevNext(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true }, video: false });
    if (!stream.getAudioTracks().length) throw new DOMException('No audio track', 'NotFoundError');
    Dev.mic.stream = stream; Dev.mic.state = 'connected';
    const AC = window.AudioContext || window.webkitAudioContext, ctx = new AC(), an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(an); Dev.mic.ctx = ctx;
    const buf = new Float32Array(an.fftSize); let peak = 0;
    const loop = () => { an.getFloatTimeDomainData(buf); let s = 0; for (const x of buf) s += x * x; const lvl = Math.sqrt(s / buf.length);
      $('#micLevel').style.width = Math.min(100, lvl * 600) + '%'; peak = Math.max(peak, lvl);
      if (Dev.mic.state === 'connected' && peak > 0.02) { Dev.mic.state = 'ok'; setBadge('#micBadge', 'Working', 'ok'); $('#micStatus').textContent = 'Sound detected — the microphone is working.'; $('#micStatus').className = 'status ok'; updateDevNext(); }
      Dev.mic.raf = requestAnimationFrame(loop); };
    loop();
    setBadge('#micBadge', 'Listening…', 'warn'); $('#micStatus').textContent = 'Microphone connected. Say a sentence — it is marked Working only after sound is detected.'; $('#micStatus').className = 'status';
    $('#btnRecTest').disabled = typeof MediaRecorder === 'undefined';
  } catch (e) { Dev.mic.state = 'error'; setBadge('#micBadge', 'Error', 'bad'); $('#micStatus').textContent = friendlyMediaError(e); $('#micStatus').className = 'status bad'; }
  $('#btnMic').textContent = 'Retry Microphone Test'; updateDevNext();
}
function recordTest() {
  if (!Dev.mic.stream || typeof MediaRecorder === 'undefined') return;
  const rec = new MediaRecorder(Dev.mic.stream), chunks = []; Dev.mic.recorder = rec;
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.onstop = () => { if (Dev.mic.url) URL.revokeObjectURL(Dev.mic.url); Dev.mic.url = URL.createObjectURL(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }));
    const a = $('#micPlayback'); a.src = Dev.mic.url; a.hidden = false; $('#btnRecTest').disabled = false; $('#btnRecTest').textContent = 'Record 4-second test';
    $('#micStatus').textContent = 'Recorded. Press play to hear yourself (kept only in this page, never uploaded).'; };
  rec.start(); $('#btnRecTest').disabled = true; $('#btnRecTest').textContent = 'Recording…'; setTimeout(() => rec.state !== 'inactive' && rec.stop(), 4000);
}
function readMode() { const c = document.querySelector('input[name="inputMode"]:checked'); S.mode = c ? c.value : 'typed'; }
function updateDevNext() {
  readMode();
  const sp = $('#sttSupport');
  sp.textContent = sttSupported() ? 'Speech recognition is available in this browser (Edge/Chrome send audio to their speech service, so internet is needed).' : 'This browser has no speech recognition. Use typed answers, or switch to Edge or Chrome.';
  let ok = true, note = '';
  if (S.mode === 'spoken') {
    if (!sttSupported()) { ok = false; note = 'Spoken answers need browser speech recognition, which is not available here.'; }
    else if (Dev.mic.state !== 'ok') { ok = false; note = 'Spoken answers need a verified microphone: press “Start Microphone Test” and speak until it shows Working.'; }
  } else note = 'Typed answers need no devices. The camera is optional.';
  $('#ttsSupport').textContent = TTS ? 'Uses your browser\'s built-in voice. Press Test voice to check the volume.' : 'This browser has no text-to-speech, so questions will be shown as text only.';
  $('#voiceOn').disabled = !TTS; if (!TTS) $('#voiceOn').checked = false;
  $('#devNote').textContent = note; $('#devNext').disabled = !ok;
}
function releaseDevices(keepCam) {
  stopMicMeter(); stopStream(Dev.mic.stream); Dev.mic.stream = null; Dev.mic.state = 'idle';
  if (!keepCam) { stopStream(Dev.cam.stream); Dev.cam = { state: 'idle', stream: null }; $('#devVideo').parentElement.classList.remove('on'); }
}

/* ------------------------------------------------------------ 5. review */
function renderReview() {
  const { n_tech, n_hr } = counts();
  const rows = [['Role', S.role], ['Interview type', S.itype], ['Difficulty', S.diff + ' (estimated from wording)'], ['Questions', `${n_tech} technical + ${n_hr} HR = ${n_tech + n_hr}`],
    ['Estimated duration', `about ${n_tech * S.opts.minutes_per_tech + n_hr * S.opts.minutes_per_hr} minutes`], ['Answer input', S.mode === 'spoken' ? 'Spoken (speech-to-text)' : 'Typed'],
    ['Camera', Dev.cam.state === 'ok' ? 'On (self-view only)' : 'Off'], ['Voice interviewer', $('#voiceOn').checked && TTS ? 'On (questions read aloud)' : 'Off']];
  if (S.name) rows.push(['Session label', S.name]);
  $('#reviewList').innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  $('#revSpoken').hidden = S.mode !== 'spoken'; $('#beginError').hidden = true;
}

/* ------------------------------------------------------------ 6. interview */
async function beginInterview() {
  const btn = $('#btnBegin'); btn.disabled = true; $('#beginError').hidden = true;
  try {
    const d = await api('/api/start', { role: S.role, interview_type: S.itype, difficulty: S.diff, ...counts(), candidate_name: S.name, input_mode: S.mode });
    Object.assign(S, { sid: d.session_id, questions: d.questions, config: d.config, idx: 0, records: [], viewingHistory: false });
    if (d.difficulty_note) toast(d.difficulty_note);
  } catch (e) { const b = $('#beginError'); b.textContent = e.message; b.hidden = false; btn.disabled = false; return; }
  btn.disabled = false;
  releaseDevices(true);
  $('#roleChip').textContent = `${S.role} · ${S.itype}`;
  const camOn = Dev.cam.state === 'ok' && Dev.cam.stream;
  $('#camCard').hidden = !camOn;
  if (camOn) { const v = $('#liveVideo'); v.srcObject = Dev.cam.stream; v.play().catch(() => {}); $('#liveWrap').classList.add('on'); $('#btnCamHide').textContent = 'Hide'; }
  S.voice = TTS && $('#voiceOn').checked; $('#voiceLive').checked = S.voice; $('#voiceLive').disabled = !TTS; $('#btnReplay').disabled = !TTS;
  $('#typedBox').hidden = S.mode === 'spoken'; $('#spokenBox').hidden = S.mode !== 'spoken';
  $('#submitHint').textContent = S.mode === 'spoken' ? 'Stop recording, then submit.' : 'Ctrl + Enter to submit';
  $('#savedList').textContent = 'None yet.';
  S.startedAt = performance.now(); clearInterval(S.timer); S.timer = setInterval(() => ($('#timer').textContent = mmss(performance.now() - S.startedAt)), 1000); $('#timer').textContent = '00:00';
  show('interview'); renderQuestion();
}
function renderQuestion() {
  const q = S.questions[S.idx];
  $('#qCount').textContent = `Question ${S.idx + 1} of ${S.questions.length}`;
  const p = Math.round((S.idx / S.questions.length) * 100); $('#progressBar').style.width = p + '%'; $('#progress').setAttribute('aria-valuenow', p);
  $('#qTag').textContent = q.is_hr ? '🗣️ HR / Behavioral' : `💻 Technical · ${nice(q.skill)}${q.difficulty ? ' · ' + q.difficulty : ''}`;
  $('#qText').textContent = q.question;
  $('#answer').value = ''; $('#answer').disabled = false; $('#btnSubmit').disabled = true; $('#evalBusy').hidden = true;
  resetTranscript(); S.firstAt = null; S.shownAt = performance.now(); S.submitting = false;
  if (S.mode === 'typed') $('#answer').focus();
  speakCurrentQuestion();
}
function currentAnswer() { return S.mode === 'spoken' ? (S.transcriptFinal + ' ' + S.transcriptInterim).trim() : $('#answer').value.trim(); }
function refreshSubmit() { $('#btnSubmit').disabled = S.submitting || !currentAnswer(); }
function renderTranscript() {
  const t = $('#transcript'), f = S.transcriptFinal.trim(), i = S.transcriptInterim.trim();
  t.innerHTML = f || i ? `${esc(f)}${i ? ` <span class="muted">${esc(i)}</span>` : ''}` : '<span class="muted">Your transcript will appear here.</span>';
}
function resetTranscript() { stopRecognition(); S.transcriptFinal = ''; S.transcriptInterim = ''; S.recMs = 0; renderTranscript(); setRec(false); }
function setRec(on, msg) { $('#btnRecStart').disabled = on; $('#btnRecStop').disabled = !on; const p = $('#recStatus'); p.textContent = msg || (on ? '● Recording…' : 'Not recording'); p.className = 'pill' + (on ? ' rec' : ''); }
function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition; if (!SR) { toast('Speech recognition is not supported in this browser.'); return; }
  const r = new SR(); r.lang = 'en-IN'; r.continuous = true; r.interimResults = true; S.recog = r; S.listening = true; S.recStart = performance.now();
  r.onresult = e => { let fin = '', interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) { const t = e.results[i][0].transcript; if (e.results[i].isFinal) fin += t + ' '; else interim += t; }
    if (fin) S.transcriptFinal += fin; S.transcriptInterim = interim; renderTranscript(); refreshSubmit(); };
  r.onerror = e => { if (e.error === 'not-allowed') toast('Microphone permission was denied for speech recognition.'); else if (e.error === 'network') toast('Speech recognition needs an internet connection.'); else if (e.error !== 'no-speech' && e.error !== 'aborted') toast('Speech recognition error: ' + e.error); };
  r.onend = () => { if (S.listening) { try { r.start(); } catch (_) {} } };
  try { r.start(); setRec(true); } catch (_) { S.listening = false; toast('Could not start speech recognition.'); }
}
function stopRecognition() {
  if (S.listening) { S.recMs += performance.now() - S.recStart; }
  S.listening = false; if (S.recog) { try { S.recog.stop(); } catch (_) {} S.recog = null; }
  if (S.transcriptInterim) { S.transcriptFinal += S.transcriptInterim + ' '; S.transcriptInterim = ''; renderTranscript(); }
  setRec(false); refreshSubmit();
}
async function submitAnswer() {
  if (S.submitting) return; stopSpeaking(); if (S.listening) stopRecognition();
  const text = currentAnswer(); if (!text) return;
  S.submitting = true; $('#btnSubmit').disabled = true; $('#answer').disabled = true; $('#evalBusy').hidden = false;
  const payload = { session_id: S.sid, index: S.idx, answer: text, input_mode: S.mode, latency: ((S.firstAt ?? performance.now()) - S.shownAt) / 1000 };
  if (S.mode === 'spoken') payload.speech_seconds = S.recMs / 1000;
  try {
    const res = await api('/api/answer', payload);
    S.records.push(res.record); $('#evalBusy').hidden = true;
    $('#savedList').innerHTML = S.records.map(r => `<div>Q${r.index + 1} saved</div>`).join('');
    if (res.done) await finish(); else { S.idx++; renderQuestion(); }
  } catch (e) { toast(e.message); S.submitting = false; $('#evalBusy').hidden = true; $('#answer').disabled = false; refreshSubmit(); }
}
async function finish() {
  clearInterval(S.timer); stopSpeaking(); stopRecognition(); stopStream(Dev.cam.stream); Dev.cam = { state: 'idle', stream: null }; $('#devVideo').parentElement.classList.remove('on');
  try { const r = await api(`/api/report?session_id=${S.sid}`); S.lastReport = r; S.viewingHistory = false; renderResults(r); show('results'); }
  catch (e) { toast(e.message); show('welcome'); loadWelcome(); }
}

/* ------------------------------------------------------------ 7. results */
const DIM_LABEL = { correctness: 'Technical correctness', relevance: 'Relevance', completeness: 'Completeness of key concepts', explanation: 'Explanation & reasoning',
  specificity: 'Specificity & clarity', example: 'Real example', action: 'Actions you took', outcome: 'Outcome / learning' };
const bars = (obj, weights) => Object.keys(obj).filter(k => num(obj[k]) !== null).map(k => `<div class="brow"><span>${esc(DIM_LABEL[k] || k)}${weights && weights[k] ? ` <small class="muted">(${Math.round(weights[k] * 100)}%)</small>` : ''}</span><div class="bar"><i class="${barClass(obj[k])}" style="width:${Math.max(2, obj[k])}%"></i></div><b>${sc(obj[k])}</b></div>`).join('');
const list = (items, empty) => (items && items.length ? `<ul class="recs">${items.map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : `<p class="muted">${esc(empty)}</p>`);

function renderResults(r) {
  const cfg = r.config || {}, d = r.delivery || {}, sw = (r.scoring && r.scoring.tech_weights) || {}, hw = (r.scoring && r.scoring.hr_weights) || {};
  const overall = num(r.overall_score), verdict = overall === null ? 'No score' : overall >= 75 ? 'Strong performance' : overall >= 55 ? 'Developing: a few gaps to close' : 'Needs more practice';
  const circ = 326.7, off = circ * (1 - (overall || 0) / 100);
  const tiles = [['Technical answer quality', sc(r.technical_quality), r.technical_quality === null ? 'No technical questions answered' : 'average of per-question quality /100'],
    ['HR / behavioral quality', sc(r.hr_quality), r.hr_quality === null ? 'No HR questions answered' : 'separate behavioral rubric /100'],
    ['Estimated delivery', d.available ? sc(d.score) : '—', d.available ? 'spoken answers only; not a measure of confidence' : (d.reason || 'Unavailable')],
    ['Answered', `${r.answered}/${r.total}`, r.answered < r.total ? 'interview ended early' : 'all questions answered']];
  let html = `<div class="card report-hero"><div class="ring-wrap"><svg viewBox="0 0 120 120" class="ring" aria-hidden="true"><circle cx="60" cy="60" r="52" class="ring-bg"/><circle cx="60" cy="60" r="52" class="ring-fg" style="stroke-dashoffset:${off};stroke:${beliefColor((overall || 0) / 100)}"/></svg><div class="ring-text"><b>${sc(overall)}</b><span>overall /100</span></div></div>
    <div class="report-text"><h1>${esc(verdict)}</h1><p class="muted">${esc(cfg.role || '')} · ${esc(cfg.interview_type || '')} · ${esc(cfg.difficulty || '')}${cfg.candidate_name ? ' · ' + esc(cfg.candidate_name) : ''}</p>
    <p class="caveat">Scores are rubric-based estimates from a local evaluator. They have not been validated against human expert ratings${r.low_evidence_count ? `; ${r.low_evidence_count} technical question(s) had no reference concepts, so their correctness and completeness are capped estimates` : ''}.</p>${r.difficulty_note ? `<p class="caveat">${esc(r.difficulty_note)}</p>` : ''}</div></div>`;
  html += `<div class="tiles">${tiles.map(t => `<div class="tile"><span>${esc(t[0])}</span><b>${esc(t[1])}</b><small>${esc(t[2])}</small></div>`).join('')}</div>`;
  html += `<div class="grid2 even"><div class="card"><h3>Technical score breakdown</h3>${bars(r.dimension_averages || {}, sw) || '<p class="muted">No technical answers.</p>'}${r.technical_quality !== null ? `<div class="formula">Quality = ${Object.keys(sw).map(k => `${sw[k]}×${k}`).join(' + ')}</div>` : ''}
    ${Object.values(r.hr_dimension_averages || {}).some(v => num(v) !== null) ? `<h3 class="gap">HR score breakdown</h3>${bars(r.hr_dimension_averages, hw)}` : ''}</div>
    <div class="card"><h3>Delivery &amp; filler words</h3>${d.available ? `<div class="brow"><span>Filler words</span><div class="bar"><i class="${barClass(100 - Math.min(100, d.filler_rate * 10))}" style="width:${Math.max(2, 100 - Math.min(100, d.filler_rate * 10))}%"></i></div><b>${d.filler_count}</b></div>
      <p>${d.filler_count} detected filler word(s) in ${d.words} spoken words = <b>${d.filler_rate.toFixed(1)}%</b> filler rate. Delivery estimate: <b>${sc(d.score)}/100</b>.</p>
      <div class="formula">Filler rate = fillers ÷ spoken words × 100<br>Delivery = 0.50×filler + 0.25×pace + 0.25×repetition (available parts only)</div>` : `<p class="muted">${esc(d.reason || 'Delivery metrics are unavailable.')}</p>`}
      <p class="small muted">Delivery uses the speech-recognition transcript and is reported separately from answer quality. Camera and voice signals are not used to judge emotion or confidence.</p></div></div>`;
  html += `<div class="grid2 even"><div class="card"><h3>Strengths</h3>${list(r.strengths, 'No standout strengths were detected yet.')}</div><div class="card"><h3>Areas for improvement</h3>${list(r.improvements, 'Nothing major to flag.')}</div></div>`;
  html += `<div class="card"><h3>Suggested topics to revise</h3>${list(r.revise_topics, 'No weak topics detected in the skills you were asked about.')}</div>`;
  html += `<div class="grid2 wide"><div class="card"><h3>Skill map <span class="muted small">Bayesian network (hand-set priors)</span></h3><svg id="bigGraph" class="graph" role="img" aria-label="Skill map"></svg><div class="legend"><i class="g"></i>strong <i class="a"></i>unsure <i class="r"></i>weak · solid ring = asked directly · dashed = inferred</div></div>
    <div class="card"><h3>Interview-state journey <span class="muted small">Markov chain</span></h3><div class="timeline">${(r.state_history || []).map((s, i) => `<span class="state ${esc(s)}" title="after answer ${i}">${esc(s)}</span>`).join('<span class="muted">→</span>')}</div>
    <h3 class="gap">What if you fix a weak skill?</h3>${whatIf(r.what_if)}</div></div>`;
  html += `<div class="card"><h3>Question by question</h3>${(r.records || []).map(qDetail).join('')}</div>`;
  html += `<div class="card"><h3>How scores are calculated</h3><div class="formula">Technical quality = ${Object.keys(sw).map(k => `${sw[k]}×${k}`).join(' + ')} (each 0-100)<br>HR quality = ${Object.keys(hw).map(k => `${hw[k]}×${k}`).join(' + ')} (each 0-100)<br>Pass threshold for skill-map evidence = ${(r.scoring || {}).pass_threshold}</div>
    <p class="small muted">Weights are initial rubric settings, configurable in scoring_config.py, and not scientifically validated. Keyword presence alone never earns correctness. Without a reference answer the evaluator cannot verify facts, so such scores are capped. Experimental RBM latent score (synthetic-trained, not part of any score above): ${r.experimental_latent_score === null ? '—' : sc(r.experimental_latent_score * 100)}.</p></div>`;
  html += `<details class="card" id="indepBox" ${S.sid && !S.viewingHistory ? '' : 'hidden'}><summary><b>Independence explorer</b> <span class="muted small">Unit I: are two skills dependent?</span></summary><div class="indep"><label>Skill A <select id="indA"></select></label><label>Skill B <select id="indB"></select></label><div><span class="muted small">Observed (given)</span><div id="indGiven" class="chips"></div></div></div><p id="indResult" class="indres"></p></details>`;
  $('#resultsRoot').innerHTML = html;
  if (r.graph && r.graph.nodes.length) drawGraph($('#bigGraph'), r.graph);
  if (S.sid && !S.viewingHistory && r.graph) setupIndependence(r.graph);
  $('#btnRetake').hidden = false;
}
function whatIf(w) {
  if (!w || !w.items || !w.items.length) return '<p class="muted">No weak skills to simulate.</p>';
  return '<ul class="recs">' + w.items.map(x => `<li>If you master <b>${niceH(x.skill)}</b> (now ${pct(x.belief_now)}): average skill ${pct(x.avg_skill_before)} → ${pct(x.avg_skill_after)}; chance of the <b>Confident</b> state ${w.horizon} answers ahead ${pct(x.confident_before)} → <b>${pct(x.confident_after)}</b></li>`).join('') +
    '</ul><p class="muted small">Model-based estimate from hand-set priors (belief treated as the chance of answering correctly), not a measured outcome.</p>';
}
function qDetail(x) {
  const fb = x.feedback || { strengths: [], improvements: [] }, dm = x.dimensions || {}, wts = x.rubric === 'hr' ? null : null;
  const ev = x.evidence_level === 'curated' ? 'reference concepts' : x.evidence_level === 'keywords' ? 'keyword evidence' : x.evidence_level === 'heuristic' ? 'heuristic rubric' : 'no reference answer';
  return `<details class="q"><summary><span class="tag ${x.is_hr ? 'hr' : ''}">${x.is_hr ? 'HR' : niceH(x.skill)}</span><b>Q${x.index + 1}</b><span>${esc(x.question.slice(0, 110))}</span><span class="qsc">${sc(x.quality)}/100</span></summary><div class="qb">
    <p><b>Your answer</b> <span class="muted small">(${x.input_mode === 'spoken' ? 'spoken, transcribed' : 'typed'})</span></p><div class="ans">${esc(x.answer)}</div>
    <p class="muted small">${esc(fb.summary || '')} · evidence: ${esc(ev)}</p>${bars(dm)}
    ${(x.matched || []).length || (x.missing || []).length ? `<div class="chips">${(x.matched || []).map(k => `<span class="c ok">✓ ${esc(k)}</span>`).join('')}${(x.missing || []).map(k => `<span class="c no">✗ ${esc(k)}</span>`).join('')}</div>` : ''}
    ${(x.misconceptions || []).length ? `<div class="warnbox">${x.misconceptions.map(m => `<div>⚠ ${esc(m)}</div>`).join('')}</div>` : ''}
    ${(fb.strengths || []).length ? `<p><b>What was good</b></p><ul class="fbl">${fb.strengths.map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : ''}
    ${(fb.improvements || []).length ? `<p><b>What to improve</b></p><ul class="fbl">${fb.improvements.map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : ''}
    ${(x.key_concepts || []).length ? `<p><b>Key concepts to include:</b> ${x.key_concepts.map(esc).join(', ')}</p>` : ''}
    ${x.model_answer ? `<p><b>Model answer</b></p><div class="ans">${esc(x.model_answer)}</div>` : ''}
    ${x.reference && !x.model_answer ? `<p><b>Source notes</b> <span class="muted small">(from the question source repository; not a verified answer)</span></p><div class="ans">${esc(x.reference)}</div>` : ''}
    ${x.is_hr && x.star ? `<p class="small muted">STAR elements detected: ${Object.entries(x.star).map(([k, v]) => `${k} ${v ? '✓' : '✗'}`).join(' · ')} (a guide, not a requirement)</p>` : ''}
    ${(x.delivery_feedback || []).length ? `<p><b>Delivery</b></p><ul class="fbl">${x.delivery_feedback.map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : ''}</div></details>`;
}
function setupIndependence(g) {
  const ids = g.nodes.map(n => n.id), A = $('#indA'), B = $('#indB'), G = $('#indGiven');
  [A, B].forEach(s => (s.innerHTML = ids.map(i => `<option value="${esc(i)}">${niceH(i)}</option>`).join('')));
  A.value = g.edges[0] ? g.edges[0][0] : ids[0]; B.value = g.edges.length ? g.edges[g.edges.length - 1][1] : ids[Math.min(1, ids.length - 1)];
  const refresh = async () => {
    const a = A.value, b = B.value, given = [...G.querySelectorAll('input:checked')].map(i => i.value);
    G.innerHTML = ids.filter(i => i !== a && i !== b).map(i => `<label class="gv"><input type="checkbox" value="${esc(i)}" ${given.includes(i) ? 'checked' : ''}>${niceH(i)}</label>`).join('');
    G.querySelectorAll('input').forEach(i => (i.onchange = refresh));
    const res = $('#indResult'); if (a === b) { res.textContent = 'Pick two different skills.'; return; }
    const cur = [...G.querySelectorAll('input:checked')].map(i => i.value);
    try { const d = await api('/api/independence', { session_id: S.sid, a, b, given: cur });
      res.innerHTML = `<b>${niceH(a)}</b> and <b>${niceH(b)}</b> are <b>${d.dependent ? 'dependent' : 'independent (d-separated)'}</b> ${cur.length ? 'given ' + cur.map(niceH).join(', ') : 'when nothing else is observed'}.`; }
    catch (e) { res.textContent = e.message; }
  };
  A.onchange = B.onchange = refresh; refresh();
}

/* ------------------------------------------------------------ navigation / wiring */
function goHome() { clearInterval(S.timer); stopSpeaking(); stopRecognition(); releaseDevices(false); S.sid = null; loadWelcome(); }
function openDevices() { show('devices'); updateDevNext(); }
document.addEventListener('DOMContentLoaded', () => {
  initRoles().catch(e => toast('Could not reach the server: ' + e.message));
  loadWelcome();
  $('#navHome').onclick = () => { if (!$('#view-interview').hidden && !confirmLeave()) return; goHome(); };
  $('#btnNew').onclick = () => { S.viewingHistory = false; show('role'); };
  $('#roleBack').onclick = goHome; $('#roleNext').onclick = () => { if (S.role) { syncConfig(); show('config'); } };
  $('#nTech').oninput = $('#nHr').oninput = $('#candName').oninput = syncConfig;
  $('#cfgBack').onclick = () => show('role'); $('#cfgNext').onclick = () => { syncConfig(); if (!configErrors().length) openDevices(); };
  $('#btnCam').onclick = startCamera; $('#btnMic').onclick = startMic; $('#btnRecTest').onclick = recordTest;
  document.querySelectorAll('input[name="inputMode"]').forEach(i => (i.onchange = updateDevNext));
  $('#devBack').onclick = () => { releaseDevices(false); show('config'); };
  $('#devNext').onclick = () => { readMode(); updateDevNext(); if (!$('#devNext').disabled) { renderReview(); show('review'); } };
  $('#revBack').onclick = openDevices; $('#btnBegin').onclick = beginInterview;
  $('#answer').addEventListener('input', () => { if (S.firstAt === null) S.firstAt = performance.now(); refreshSubmit(); });
  $('#answer').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submitAnswer(); });
  $('#btnSubmit').onclick = submitAnswer;
  $('#btnVoiceTest').onclick = () => speak('Hello. I will be your interviewer today. Let us begin when you are ready.');
  $('#btnReplay').onclick = () => { const was = S.voice; S.voice = true; speakCurrentQuestion(); S.voice = was; };
  $('#voiceLive').onchange = () => { S.voice = $('#voiceLive').checked; if (!S.voice) stopSpeaking(); else speakCurrentQuestion(); };
  $('#btnRecStart').onclick = () => { if (S.firstAt === null) S.firstAt = performance.now(); startRecognition(); };
  $('#btnRecStop').onclick = stopRecognition; $('#btnClearTr').onclick = resetTranscript;
  $('#btnCamHide').onclick = () => { const w = $('#liveWrap'), hidden = w.hidden = !w.hidden; $('#btnCamHide').textContent = hidden ? 'Show' : 'Hide'; };
  $('#btnEnd').onclick = () => ($('#endModal').hidden = false); $('#endCancel').onclick = () => ($('#endModal').hidden = true);
  $('#endConfirm').onclick = async () => { $('#endModal').hidden = true; if (!S.records.length) { toast('No answers were submitted, so there is nothing to score.'); goHome(); } else await finish(); };
  $('#btnRetake').onclick = () => { if (!S.config) { show('role'); return; } S.role = S.config.role; S.itype = S.config.interview_type; S.diff = S.config.difficulty; S.mode = S.config.input_mode || 'typed'; S.name = S.config.candidate_name || ''; S.viewingHistory = false; selectRole(S.role); buildSegments(); syncConfig(); renderReview(); show('review'); };
  $('#btnHome').onclick = goHome;
});
function confirmLeave() { return window.confirm ? window.confirm('Leave the interview? Your progress in this session will be kept only if you finish it.') : true; }
window.__IP = { S, Dev };   // exposed for the automated UI test
})();
