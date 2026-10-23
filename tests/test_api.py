import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import history
history.HISTORY_FILE = os.path.join(tempfile.mkdtemp(), 'history.json')       # never touch the real history in tests
import server

GOOD = ('Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on unseen data. '
        'Underfitting is when the model is too simple. To prevent overfitting you can use regularization, dropout, early stopping or more data.')


class ApiFlow(unittest.TestCase):
    def setUp(self): self.c = server.app.test_client()

    def start(self, **kw):
        body = {'role': 'Data Scientist', 'interview_type': 'Mixed', 'difficulty': 'Intermediate', 'n_tech': 2, 'n_hr': 1, 'input_mode': 'typed'}; body.update(kw)
        return self.c.post('/api/start', json=body)

    def test_invalid_configurations_rejected(self):
        for bad in [{'role': 'Astronaut'}, {'interview_type': 'Mixed', 'n_tech': 0}, {'interview_type': 'Mixed', 'n_hr': 0}, {'interview_type': 'Technical', 'n_tech': 0},
                    {'n_tech': 99}, {'difficulty': 'Impossible'}, {'candidate_name': 'x' * 41}, {'input_mode': 'telepathy'}, {'n_tech': 'abc'}]:
            self.assertEqual(self.start(**bad).status_code, 400, bad)

    def test_types_respect_counts(self):
        t = self.start(interview_type='Technical', n_tech=3, n_hr=4).get_json(); self.assertTrue(all(not q['is_hr'] for q in t['questions'])); self.assertEqual(len(t['questions']), 3)
        h = self.start(interview_type='HR/Behavioral', n_tech=5, n_hr=2).get_json(); self.assertTrue(all(q['is_hr'] for q in h['questions'])); self.assertEqual(len(h['questions']), 2)

    def test_one_role_only(self):
        d = self.start(role='Frontend Developer', interview_type='Technical', n_tech=8).get_json()
        sess = server.SESSIONS[d['session_id']]; self.assertTrue(all(q['role'] == 'Frontend Developer' for q in sess.questions))

    def test_full_session_persists_and_report(self):
        d = self.start().get_json(); sid = d['session_id']
        for i, q in enumerate(d['questions']):
            r = self.c.post('/api/answer', json={'session_id': sid, 'index': i, 'answer': 'um ' + GOOD if i == 0 else 'I once led a team project at college, I organized tasks and we finished 2 days early, so I learned planning.', 'input_mode': 'spoken' if i == 0 else 'typed', 'speech_seconds': 20})
            self.assertEqual(r.status_code, 200); j = r.get_json(); self.assertEqual(j['answered'], i + 1); self.assertEqual(j['done'], i == len(d['questions']) - 1)
        rep = self.c.get(f'/api/report?session_id={sid}').get_json()
        self.assertEqual(rep['answered'], 3); self.assertEqual(len(rep['records']), 3); self.assertTrue(0 <= rep['overall_score'] <= 100)
        self.assertTrue(rep['delivery']['available']); self.assertEqual(rep['delivery']['filler_count'], 1)
        self.assertIsNone(rep['records'][1]['delivery']['filler_count'])        # typed answer: speech metrics unavailable
        items = self.c.get('/api/history').get_json()['items']; self.assertTrue(any(i['id'] == sid for i in items))
        self.assertEqual(self.c.get(f'/api/history/{sid}').get_json()['report']['answered'], 3)

    def test_retake_creates_new_session_and_keeps_old(self):
        a = self.start().get_json()['session_id']; b = self.start().get_json()['session_id']; self.assertNotEqual(a, b)
        self.c.post('/api/answer', json={'session_id': a, 'index': 0, 'answer': GOOD}); self.assertEqual(len(server.SESSIONS[a].records), 1); self.assertEqual(len(server.SESSIONS[b].records), 0)
        self.c.get(f'/api/report?session_id={a}'); self.assertIsNotNone(history.get_result(a))

    def test_answer_validation(self):
        sid = self.start().get_json()['session_id']
        self.assertEqual(self.c.post('/api/answer', json={'session_id': sid, 'index': 0, 'answer': '   '}).status_code, 400)
        self.assertEqual(self.c.post('/api/answer', json={'session_id': sid, 'index': 1, 'answer': 'x'}).status_code, 400)       # out of order
        self.assertEqual(self.c.post('/api/answer', json={'session_id': sid, 'index': 0, 'answer': 'x' * 7000}).status_code, 400)
        self.assertEqual(self.c.post('/api/answer', json={'session_id': 'nope', 'index': 0, 'answer': 'x'}).status_code, 404)
        self.assertEqual(self.c.get(f'/api/report?session_id={sid}').status_code, 400)                                       # nothing answered yet

    def test_early_end_scores_answered_only(self):
        d = self.start(n_tech=3).get_json(); sid = d['session_id']
        self.c.post('/api/answer', json={'session_id': sid, 'index': 0, 'answer': GOOD})
        rep = self.c.get(f'/api/report?session_id={sid}').get_json(); self.assertEqual((rep['answered'], rep['total']), (1, 4))

    def test_history_survives_corrupt_file(self):
        p = history.HISTORY_FILE; open(p, 'w').write('{not json'); self.assertEqual(history.list_history(p), [])
        self.assertTrue(any(f.startswith('history.json.corrupt') for f in os.listdir(os.path.dirname(p))))


if __name__ == '__main__':
    unittest.main()
