"""Run this FIRST on your machine:  python check_setup.py
Camera / microphone / speech are handled by your BROWSER (Edge or Chrome), so they are tested inside the app
with the 'Test camera & mic' button - not here."""
import importlib, os, socket, sys, warnings
warnings.filterwarnings('ignore')
OK, WARN, BAD = '[ OK ]', '[WARN]', '[FAIL]'
failed = False

print(f'Python {sys.version.split()[0]}  (3.10 - 3.12 recommended)')
print('\n1. Libraries')
for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'requests', 'pgmpy', 'flask']:
    try:
        importlib.import_module(m); print(OK, m)
    except Exception as e:
        failed = True; print(BAD, m, '->', e, '  (run: pip install -r requirements.txt)')

print('\n2. Project files')
from paths import QUESTION_BANK, RBM_WEIGHTS, RBM_ANCHORS
for p, fix in [(QUESTION_BANK, 'python build_question_bank.py'), (RBM_WEIGHTS, 'python train_rbm.py'), (RBM_ANCHORS, 'python train_rbm.py')]:
    if os.path.exists(p): print(OK, p.name)
    else: failed = True; print(BAD, p.name, f'missing -> run:  {fix}')
here = os.path.dirname(os.path.abspath(__file__))
for f in ['index.html', 'style.css', 'app.js', 'sensors.js']:
    ok = os.path.exists(os.path.join(here, 'static', f))
    failed |= not ok; print(OK if ok else BAD, 'static/' + f)

if not failed:
    print('\n3. Engine self-test (all 5 roles)')
    import pandas as pd
    from interview_engine import InterviewSession
    from skill_network import ROLE_SKILL_GRAPHS, check_category_coverage
    qb = pd.read_csv(QUESTION_BANK)
    print(OK if not check_category_coverage(qb) else BAD, 'every question category maps to a skill node')
    for role in ROLE_SKILL_GRAPHS:
        s = InterviewSession(role, qb, n_tech=4, n_hr=1, seed=1)
        for i, q in enumerate(s.questions):
            s.submit(i, 'x ' * 45 if q['is_hr'] else q['expected_keywords'].replace(',', ' '), 10)
        print(OK, f"{role}: engine ran, keyword-only answers scored {s.report()['overall_score']:.0f}/100")

print('\n4. Port 5055 (used by the app)')
sock = socket.socket(); sock.settimeout(0.5)
busy = sock.connect_ex(('127.0.0.1', 5055)) == 0; sock.close()
print(WARN if busy else OK, 'port 5055 is ' + ('already in use - close the other copy of the app' if busy else 'free'))

print('\n' + ('Fix the [FAIL] items above, then re-run.' if failed else 'All required checks passed. Start the app with:  python server.py'))
