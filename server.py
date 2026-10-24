"""
InterviewIQ server.   Run:  python server.py     (opens http://localhost:5055)

The browser handles camera, microphone, speech-to-text and read-aloud (no OpenCV / PortAudio needed).
This server holds the explainable core: grading -> Bayesian skill network -> Markov chain -> RBM confidence.
"""
import os, threading, uuid, webbrowser
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from interview_engine import (InterviewSession, INTERVIEW_TYPES, DIFFICULTIES, estimated_minutes,
                              MINUTES_PER_TECH, MINUTES_PER_HR)
import history
import scoring_config as C
from skill_network import ROLE_SKILL_GRAPHS
from confidence import ConfidenceModel
from paths import QUESTION_BANK

app = Flask(__name__, static_folder='static', static_url_path='/static')
BANK = pd.read_csv(QUESTION_BANK)
CONF = ConfidenceModel()
SESSIONS = {}          # session_id -> InterviewSession (in memory)
MAX_SESSIONS = 200     # online use: drop the oldest sessions so memory cannot grow without limit
HISTORY_ENABLED = os.environ.get('DISABLE_HISTORY') != '1'   # set DISABLE_HISTORY=1 on a public server: history is shared, not per-user


def graph_payload(sess):
    beliefs, evidence = sess.skill_beliefs()
    return {'nodes': [{'id': n, 'belief': round(float(beliefs[n]), 4), 'asked': n in evidence}
                      for n in sess.net.model.nodes()],
            'edges': [list(e) for e in sess.net.edges]}


def get_session():
    data = request.get_json(force=True, silent=True) or {}
    sid = data.get('session_id') or request.args.get('session_id')
    if sid not in SESSIONS:
        return None, data, (jsonify(error='Unknown session - start a new interview.'), 404)
    return SESSIONS[sid], data, None


@app.get('/')
def index():
    return send_from_directory('static', 'index.html')


@app.get('/api/roles')
def roles():
    out = []
    for role, cfg in ROLE_SKILL_GRAPHS.items():
        n = int(((BANK.role == role) & (~BANK.is_hr)).sum())
        skills = sorted(set(cfg['category_map'].values()))
        out.append({'name': role, 'questions': n, 'skills': len(skills), 'skill_areas': [x.replace('_', ' ') for x in skills]})
    return jsonify(roles=out, hr_questions=int(BANK.is_hr.sum()))


@app.get('/api/options')
def options():
    return jsonify(interview_types=list(INTERVIEW_TYPES), difficulties=list(DIFFICULTIES), max_tech=12, max_hr=5,
                   minutes_per_tech=MINUTES_PER_TECH, minutes_per_hr=MINUTES_PER_HR, hr_questions=int(BANK.is_hr.sum()),
                   pass_threshold=C.PASS_THRESHOLD, tech_weights=C.TECH_WEIGHTS, hr_weights=C.HR_WEIGHTS,
                   scoring_version=C.SCORING_VERSION)


def _parse_config(d):
    """Validate an interview configuration. Returns (config, errors)."""
    errors = []
    role = d.get('role')
    if role not in ROLE_SKILL_GRAPHS: errors.append('Choose one of the five roles.')
    itype = d.get('interview_type', 'Mixed')
    if itype not in INTERVIEW_TYPES: errors.append('Unknown interview type.')
    diff = d.get('difficulty', 'Intermediate')
    if diff not in DIFFICULTIES: errors.append('Unknown difficulty.')
    try:
        n_tech, n_hr = int(d.get('n_tech', 6)), int(d.get('n_hr', 2))
    except (TypeError, ValueError):
        return None, ['Question counts must be whole numbers.']
    if not 0 <= n_tech <= 12 or not 0 <= n_hr <= 5: errors.append('Question counts are out of range (technical 0-12, HR 0-5).')
    if itype == 'Technical': n_hr = 0
    if itype == 'HR/Behavioral': n_tech = 0
    if itype == 'Technical' and n_tech < 1: errors.append('Choose at least one technical question.')
    if itype == 'HR/Behavioral' and n_hr < 1: errors.append('Choose at least one HR question.')
    if itype == 'Mixed' and (n_tech < 1 or n_hr < 1): errors.append('A mixed interview needs at least one technical and one HR question.')
    name = str(d.get('candidate_name') or '').strip()
    if len(name) > 40: errors.append('Name must be 40 characters or fewer.')
    mode = d.get('input_mode', 'typed')
    if mode not in ('typed', 'spoken'): errors.append('Input mode must be typed or spoken.')
    return {'role': role, 'interview_type': itype, 'difficulty': diff, 'n_tech': n_tech, 'n_hr': n_hr,
            'candidate_name': name, 'input_mode': mode}, errors


@app.post('/api/start')
def start():
    cfg, errors = _parse_config(request.get_json(force=True, silent=True) or {})
    if errors:
        return jsonify(error=' '.join(errors), errors=errors), 400
    sess = InterviewSession(cfg['role'], BANK, n_tech=cfg['n_tech'], n_hr=cfg['n_hr'], confidence_model=CONF,
                            interview_type=cfg['interview_type'], difficulty=cfg['difficulty'], candidate_name=cfg['candidate_name'])
    sess.input_mode = cfg['input_mode']
    if not sess.questions:
        return jsonify(error='No questions are available for that configuration.'), 400
    sid = uuid.uuid4().hex
    SESSIONS[sid] = sess
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    qs = [{'index': i, 'question': q['question'], 'is_hr': bool(q['is_hr']), 'skill': q.get('skill'),
           'category': q['category'], 'difficulty': q.get('difficulty')} for i, q in enumerate(sess.questions)]
    return jsonify(session_id=sid, config=cfg, questions=qs, graph=graph_payload(sess), difficulty_note=sess.difficulty_note,
                   estimated_minutes=estimated_minutes(cfg['n_tech'], cfg['n_hr']))


@app.post('/api/answer')
def answer():
    sess, d, err = get_session()
    if err: return err
    idx = int(d.get('index', -1))
    if idx != len(sess.records) or idx >= len(sess.questions):
        return jsonify(error='Question out of order.'), 400
    text = str(d.get('answer', '')).strip()
    if not text:
        return jsonify(error='Empty answer.'), 400
    if len(text) > 6000:
        return jsonify(error='Answer is too long (6000 characters maximum).'), 400
    mode = d.get('input_mode', 'typed')
    if mode not in ('typed', 'spoken'): return jsonify(error='Invalid input mode.'), 400
    try:
        speech = float(d.get('speech_seconds')) if d.get('speech_seconds') is not None else None
        if speech is not None and not 0 <= speech <= 3600: speech = None
    except (TypeError, ValueError):
        speech = None
    rec = sess.submit(idx, text, float(d.get('latency') or 0.0), d.get('sensors') or None, input_mode=mode, speech_seconds=speech)
    return jsonify(record=rec, state=sess.chain.state, state_history=sess.chain.history, graph=graph_payload(sess),
                   answered=len(sess.records), total=len(sess.questions), done=len(sess.records) >= len(sess.questions))


@app.get('/api/report')
def report():
    sess, _, err = get_session()
    if err: return err
    if not sess.records:
        return jsonify(error='No answers were submitted yet.'), 400
    r = sess.report()
    r['graph'] = graph_payload(sess)
    r['what_if'] = sess.what_if()
    r['config'] = {'role': sess.role, 'interview_type': sess.interview_type, 'difficulty': sess.difficulty,
                   'candidate_name': sess.candidate_name, 'input_mode': getattr(sess, 'input_mode', 'typed')}
    r['scoring'] = {'version': C.SCORING_VERSION, 'tech_weights': C.TECH_WEIGHTS, 'hr_weights': C.HR_WEIGHTS,
                    'pass_threshold': C.PASS_THRESHOLD, 'validated': False}
    sid = request.args.get('session_id')
    if HISTORY_ENABLED: history.save_result(r, {'role': sess.role, 'interview_type': sess.interview_type, 'difficulty': sess.difficulty,
                            'candidate_name': sess.candidate_name, 'answered': r['answered'], 'total': r['total'],
                            'overall_score': r['overall_score']}, session_id=sid)
    return jsonify(r)


@app.get('/api/history')
def history_list():
    return jsonify(items=history.list_history() if HISTORY_ENABLED else [])


@app.get('/api/history/<entry_id>')
def history_get(entry_id):
    e = history.get_result(entry_id) if HISTORY_ENABLED else None
    if not e:
        return jsonify(error='Result not found.'), 404
    return jsonify(e)


@app.post('/api/independence')
def independence():
    sess, d, err = get_session()
    if err: return err
    nodes = set(sess.net.model.nodes())
    a, b, given = d.get('a'), d.get('b'), [g for g in (d.get('given') or []) if g in nodes]
    if a not in nodes or b not in nodes or a == b:
        return jsonify(error='Pick two different skills.'), 400
    return jsonify(dependent=bool(sess.net.check_independence(a, b, given)), given=given)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5055))
    if 'PORT' in os.environ:          # running on a hosting service
        app.run(host='0.0.0.0', port=port, debug=False)
    else:                             # running on your own computer
        url = f'http://localhost:{port}'
        print(f'\nInterview Prep running at {url}   (Ctrl+C to stop)\nUse Microsoft Edge or Google Chrome for camera/mic/voice features.\n')
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        app.run(host='127.0.0.1', port=port, debug=False)
