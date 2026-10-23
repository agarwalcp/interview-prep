"""Persistent interview history (data/history.json). Append-only and non-destructive: a corrupt file is renamed, never deleted."""
import json, os, tempfile, threading, time, uuid
from paths import DATA_DIR

HISTORY_FILE = DATA_DIR / 'history.json'
_lock = threading.Lock()


def _load(path=None):
    path = path or HISTORY_FILE
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        os.replace(path, f'{path}.corrupt-{int(time.time())}')      # keep the unreadable file for inspection
        return []


def _write(items, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(items, f)
    os.replace(tmp, path)


def save_result(report, meta, session_id=None, path=None):
    """Store one finished interview. Saving the same session_id again replaces that entry (idempotent)."""
    path = str(path or HISTORY_FILE)
    with _lock:
        items = _load(path)
        entry = {'id': session_id or uuid.uuid4().hex, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), **meta, 'report': report}
        items = [e for e in items if e.get('id') != entry['id']] + [entry]
        _write(items, path)
    return entry['id']


def list_history(path=None):
    items = _load(str(path or HISTORY_FILE))
    keys = ('id', 'saved_at', 'role', 'interview_type', 'difficulty', 'candidate_name', 'answered', 'total', 'overall_score')
    return [{k: e.get(k) for k in keys} for e in reversed(items)]


def get_result(entry_id, path=None):
    for e in _load(str(path or HISTORY_FILE)):
        if e.get('id') == entry_id:
            return e
    return None
