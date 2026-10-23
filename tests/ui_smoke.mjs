// Optional UI flow test. Needs Node 18+ and:  npm i jsdom   then start the server and run:  node tests/ui_smoke.mjs http://127.0.0.1:5055
// It drives the real page in jsdom (no real camera/mic/speech engine), so it checks navigation, validation, state and rendering only.
import { JSDOM } from 'jsdom';
const BASE = process.argv[2] || 'http://127.0.0.1:5055';
let pass = 0, fail = 0;
const ok = (c, m) => { (c ? pass++ : fail++); console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); };
const wait = ms => new Promise(r => setTimeout(r, ms));
const html = await (await fetch(BASE + '/')).text();
const dom = new JSDOM(html, { url: BASE + '/', runScripts: 'dangerously', pretendToBeVisual: true,
  beforeParse(w) {
    w.fetch = (u, o) => fetch(new URL(u, BASE), o); w.confirm = () => true; w.scrollTo = () => {};
    w.HTMLElement.prototype.scrollIntoView = () => {}; w.HTMLMediaElement.prototype.play = () => Promise.resolve();
    w.SpeechSynthesisUtterance = class { constructor(t) { this.text = t; } };
    w.speechSynthesis = { spoken: [], speak(u) { this.spoken.push(u.text); setTimeout(() => u.onend && u.onend(), 0); }, cancel() {}, getVoices() { return []; } };
    const err = new Error('denied'); err.name = 'NotAllowedError';
    Object.defineProperty(w.navigator, 'mediaDevices', { value: { getUserMedia: () => Promise.reject(err) }, configurable: true });
  } });
const w = dom.window, d = w.document, $ = s => d.querySelector(s), vis = id => !$('#view-' + id).hidden;
const src = await (await fetch(BASE + '/static/app.js')).text(); w.eval(src); d.dispatchEvent(new w.Event('DOMContentLoaded'));
await wait(800);
const click = s => $(s).dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
const fire = (s, ev = 'input') => $(s).dispatchEvent(new w.Event(ev, { bubbles: true }));
const answers = ['A model learns the training data including noise because it is too complex, so it performs poorly on unseen data and regularization or more data helps.',
  'Cross-validation splits data into folds, trains on some and validates on the held-out fold, then averages results to estimate generalization.',
  'Precision is true positives over predicted positives and recall is true positives over actual positives; F1 is their harmonic mean.'];

ok(vis('welcome') && !vis('role'), 'welcome dashboard shows first, setup is not on the same screen');
ok($('#roles').children.length === 5, 'role cards exist (5) but only on the role screen');
click('#btnNew'); ok(vis('role') && !vis('welcome'), 'Start New Interview opens the role screen');
ok($('#roleNext').disabled, 'Continue is disabled until a role is selected');
click('#roles .role[data-role="Data Scientist"]'); click('#roles .role[data-role="ML Engineer"]');
ok(d.querySelectorAll('#roles .role.sel').length === 1 && !$('#roleNext').disabled, 'exactly one role can be selected; Continue enables');
click('#roleNext'); ok(vis('config'), 'configuration is a separate screen');
$('#typeSeg input[value="Technical"]').checked = true; fire('#typeSeg input[value="Technical"]', 'change');
ok($('#lblHr').hidden && !$('#lblTech').hidden, 'Technical type hides the HR-question slider');
$('#candName').value = 'x'.repeat(41); fire('#candName'); ok($('#cfgNext').disabled && !$('#cfgErrors').hidden, 'invalid configuration blocks Continue and shows the error');
$('#candName').value = 'Cherry'; fire('#candName'); $('#nTech').value = 3; fire('#nTech'); ok(!$('#cfgNext').disabled, 'valid configuration re-enables Continue');
click('#cfgNext'); ok(vis('devices'), 'device setup is its own screen');
$('input[name="inputMode"][value="spoken"]').checked = true; fire('input[name="inputMode"][value="spoken"]', 'change');
ok($('#devNext').disabled, 'spoken mode cannot continue without a verified microphone');
click('#btnCam'); await wait(200); ok(/blocked/i.test($('#camStatus').textContent) && $('#camBadge').classList.contains('bad'), 'camera permission denial is shown clearly (not reported as working)');
click('#btnMic'); await wait(200); ok($('#micBadge').classList.contains('bad') && $('#devNext').disabled, 'microphone error is shown and retry label offered');
$('input[name="inputMode"][value="typed"]').checked = true; fire('input[name="inputMode"][value="typed"]', 'change');
ok(!$('#devNext').disabled, 'typed mode can continue without devices');
ok($('#voiceOn').checked && !$('#voiceOn').disabled, 'voice interviewer option is available and on by default');
click('#btnVoiceTest'); ok(w.speechSynthesis.spoken.length === 1, 'Test voice speaks');
click('#devNext'); ok(vis('review') && /ML Engineer/.test($('#reviewList').textContent) && /3 technical/.test($('#reviewList').textContent), 'review screen summarises the chosen setup');
ok(!vis('interview'), 'interview does not start automatically');
click('#btnBegin'); await wait(600); ok(vis('interview') && /Question 1 of 3/.test($('#qCount').textContent), 'Begin Interview starts the session');
ok($('#btnSubmit').disabled, 'Submit is disabled for an empty answer');
const sid1 = w.__IP.S.sid; await wait(50);
ok(/^Question 1\. /.test(w.speechSynthesis.spoken[1] || '') && (w.speechSynthesis.spoken[1] || '').includes(w.__IP.S.questions[0].question), 'the interviewer reads question 1 aloud');
const n0 = w.speechSynthesis.spoken.length; click('#btnReplay'); ok(w.speechSynthesis.spoken.length === n0 + 1, 'Replay speaks the question again');
$('#voiceLive').checked = false; fire('#voiceLive', 'change');
for (let i = 0; i < 3; i++) {
  $('#answer').value = answers[i]; fire('#answer'); ok(!$('#btnSubmit').disabled, `Q${i + 1}: Submit enables after typing`);
  click('#btnSubmit'); await wait(700);
  if (i < 2) ok(/Question/.test($('#qCount').textContent) && w.__IP.S.idx === i + 1 && w.__IP.S.records.length === i + 1, `Q${i + 1}: answer saved, moved to next, previous records kept`);
}
await wait(500); ok(vis('results'), 'finishing opens the results screen');
const txt = $('#resultsRoot').textContent;
ok(/overall/.test(txt) && /Technical score breakdown/.test(txt) && /Delivery/.test(txt) && /Question by question/.test(txt), 'results show overall, breakdown, delivery and per-question sections');
ok(/Unavailable|unavailable/i.test(txt), 'delivery is labelled unavailable for typed answers (no fabricated filler stats)');
ok(d.querySelectorAll('details.q').length === 3, 'one detail panel per question');
click('#btnRetake'); ok(vis('review'), 'Retake Interview returns to the final check with the same setup');
click('#btnBegin'); await wait(600); ok(w.__IP.S.sid !== sid1 && w.__IP.S.records.length === 0 && /Question 1 of 3/.test($('#qCount').textContent), 'retake creates a NEW session with empty records');
click('#btnEnd'); ok(!$('#endModal').hidden, 'End interview asks for confirmation'); click('#endCancel'); ok($('#endModal').hidden, 'cancel keeps the interview going');
click('#navHome'); await wait(500); ok(vis('welcome'), 'dashboard reachable');
ok(/1 interviews|interviews completed/.test($('#statsRow').textContent) && d.querySelectorAll('#historyList .hist').length >= 1, 'previous results appear from real saved data');
click('#historyList [data-hid]'); await wait(500); ok(vis('results') && /Question by question/.test($('#resultsRoot').textContent), 'a previous result can be reopened');
console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail ? 1 : 0);
