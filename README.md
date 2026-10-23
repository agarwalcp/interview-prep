# InterviewIQ — Explainable Mock-Interview Analyzer (v2: browser interface)

A candidate picks a role, answers real technical + HR questions (typed or **spoken**, camera/mic optional), and the system
**explains** its result: a live skill map, the interview-state journey, a latent "confidence" score, and *why* for every number.
For your own practice — **not** for screening candidates.

**What changed in v2:** camera, microphone, speech-to-text and read-aloud now run **inside your browser** (Edge/Chrome), with a
live camera preview, meters and a visual skill graph. OpenCV, PortAudio, librosa, pyttsx3 and Streamlit are no longer needed,
which removes the install problems of v1.


## v3 changes (read this first)
Flow: Dashboard -> Role (one role) -> Configure (type, difficulty, counts, name) -> Camera/Mic check -> Final check -> Interview -> Results.
Run from the folder that contains server.py:  python check_setup.py  then  python server.py

Scoring (backend only: scoring.py, scoring_config.py, delivery.py)
- Technical quality = 0.40 x correctness + 0.25 x relevance + 0.20 x completeness + 0.15 x explanation (each 0-100; weights in scoring_config.py).
- HR quality = 0.20 x each of relevance, specificity, example, action, outcome.
- Delivery (spoken answers only) = 0.50 x filler + 0.25 x pace + 0.25 x repetition, over the parts that are available. Filler rate = fillers / spoken words x 100; no penalty up to 1%, zero credit at 10%.
- Delivery never lowers the technical score. Camera and voice signals are not used to judge emotion or confidence. The old RBM score is kept only as an experimental figure on the results page.
- data/rubric_overrides.json holds hand-written concepts, misconceptions and model answers for 17 patterns (24 questions). Other questions have no reference answer, so correctness/completeness are capped at 65 and flagged. Add entries to that file to make more questions fully gradable.
- Not validated against human ratings. Build a labelled set (answers scored 0-100 by 2+ people), compare with Scorer.evaluate_*, then tune scoring_config.py.

Voice interviewer: the browser reads each question aloud (Web Speech synthesis); in spoken mode recording starts automatically after the question finishes. Turn it off on the Devices screen or during the interview.

Tests: python -m unittest discover -s tests -v   (38 tests). Optional UI flow test: tests/ui_smoke.mjs (Node + jsdom; see its header).
History: finished interviews are appended to data/history.json (never deleted; a corrupt file is renamed).

## 1. Folder layout
```
interview_iq/                 <- open THIS folder in VS Code (the one containing server.py)
├── server.py                 <- START HERE:  python server.py   (opens http://localhost:5055)
├── static/                   <- the web interface
│   ├── index.html, style.css, app.js
│   └── sensors.js            <- camera stillness + voice-steadiness maths (runs in the browser)
├── interview_engine.py       <- question selection + grading + Bayesian net + Markov chain + RBM score
├── skill_network.py          <- per-role Bayesian Networks + Markov chain                (Unit I, IV)
├── rbm.py, confidence.py     <- RBM + latent confidence score                            (Units II-IV)
├── grading.py                <- explainable keyword grading
├── build_question_bank.py    <- fetch real questions from 6 GitHub repos -> data/question_bank.csv
├── train_rbm.py              <- train the RBM -> data/rbm_weights.npz, rbm_anchors.npz
├── check_setup.py            <- run first: tells you what to fix
├── paths.py, requirements.txt
├── data/                     <- question_bank.csv, rbm_weights.npz, rbm_anchors.npz (pre-built)
└── notebooks/                <- 00_question_bank, 01_skill_bn_markov, 02_rbm_confidence
```
(If your unzip tool adds an extra outer folder, open the **inner** one — the one that contains `server.py`.)

## 2. One-time setup (Windows commands; Mac/Linux in brackets)
1. Install **Python 3.10–3.12** (tick "Add Python to PATH") and **VS Code** + its *Python* and *Jupyter* extensions.
2. VS Code → *File → Open Folder* → the folder containing `server.py`. Then *Terminal → New Terminal*.
3. If you still have the old `venv` from v1 you can reuse it. Otherwise create one:
   ```
   python -m venv venv
   venv\Scripts\activate          [source venv/bin/activate]
   ```
4. Install (a few minutes, once):
   ```
   pip install -r requirements.txt
   ```
5. Check:
   ```
   python check_setup.py
   ```

## 3. Run
```
python server.py
```
Your browser opens `http://localhost:5055`. **Use Microsoft Edge or Google Chrome.** Stop with `Ctrl+C` in the terminal.

Screen 1 — pick a role, number of questions, and (optional) tick Camera / Microphone / Speak my answers / Read aloud.
Press **Test camera & mic**: the browser asks permission (click **Allow**), the live preview appears and the meters move when you
move and talk. Then **Start interview**.
Screen 2 — read/hear the question, type or press **Speak answer**, submit. You immediately see which concepts you hit/missed and
how your skill map changed (and why). Screen 3 — report with confidence ring, skill graph, state timeline, per-question breakdown,
recommendations and an **Independence explorer** (Unit I demo).

(Optional rebuild of the data, needs internet: `python build_question_bank.py` then `python train_rbm.py`.)

## 4. How the sensing works (say this in your viva)
| Signal | How it is computed (in the browser, no ML model, nothing uploaded) |
|---|---|
| Stillness | camera image of the centre of the frame is shrunk to 64×48 grey pixels; mean absolute change between frames every 150 ms → `calm = 1 − 5×motion` |
| Voice steadiness | microphone → autocorrelation pitch estimate on loud frames → standard deviation of pitch (octave outliers removed) → `steady = 1 − std/100 Hz`; needs ≥ 1.5 s of clear speech, else "not measured" |
| Speech-to-text | the browser's built-in SpeechRecognition (Edge/Chrome send the audio to their speech service → internet needed) |
| Read aloud | the browser's built-in speechSynthesis |
Only 3 numbers per answer (motion, pitch spread, voiced ratio) are sent to the local server on your own computer.

## 5. Syllabus mapping
| Unit / topic | File | Use |
|---|---|---|
| I – joint distributions, Bayesian networks, causal & evidential reasoning, independencies | `skill_network.py`, notebook 01 | one skill network per role; answering one skill updates related skills; d-separation explorer |
| II – latent variables | `rbm.py`, `confidence.py` | "confidence" is a hidden variable inferred from 4 signals |
| III – RBMs, unsupervised learning | `rbm.py`, notebook 02 | the confidence model |
| IV – Markov chains, Gibbs sampling, Contrastive Divergence | `skill_network.py`, `rbm.py` | interview-state tracking; RBM training (CD-1) |
| Not used | – | Markov networks (only mentioned) and Unit V VAE/GAN — they don't fit this project |

## 6. HONEST STATUS — read before your viva
**Real & tested:** ~750 real questions fetched live from 6 public GitHub repos; Bayesian inference (pgmpy), Markov chain, RBM training,
grading, the web server (all endpoints, all 5 roles) and the web interface (clicked through end-to-end in a simulated browser with a
synthetic microphone and camera). The in-browser pitch and motion maths were tested on synthetic voices and frames
(pitch within 0.2 %, shaky voice clearly separated from steady).

**Added after v2:** a `reference` column in the question bank (Backend only: 190 of 211 questions, text taken from the source repo's
own description, not a verified model answer), a reference panel after each answer, and a *What if you fix a weak skill?* section in the
report (Markov-chain forward projection + Bayesian re-inference). The what-if numbers are model-based estimates from hand-set priors,
not measurements. Re-run `python build_question_bank.py` if you want to rebuild the CSV.

**Hand-set / synthetic — say so:**
- Bayesian Network edges and probability tables are **hand-set priors**, not learned from data.
- The RBM is trained on **synthetic** strong/weak profiles. "Confidence" has no objective ground truth; it is a latent score from 4
  proxy signals and is **not validated** against human ratings (future work: MIT Interview Dataset / ChaLearn First Impressions v2).
- **Keyword grading is weakest for ML, Data Scientist, Frontend and Software Developer questions** (their repos give only titles, so
  keywords come from the question itself; repeating the question's words can score). **Backend** has real answer text. Edit the
  `expected_keywords` column in `data/question_bank.csv` for questions you will demo.
- HR "substance" is only answer length (capped) — HR answers have no correct answer.
- Camera motion uses the whole centre of the frame, not a detected face (so background movement counts too).
- Sensors off → neutral training-mean values are used.
- **Not tested on a real camera/microphone by the author** (none in the dev sandbox) — only simulated. Use the *Test camera & mic* button on your laptop before any demo.

## 7. Troubleshooting
| Problem | Fix |
|---|---|
| `requirements.txt` not found | you are in the outer folder: `cd` into the folder that contains `server.py` |
| Camera preview stays black / "Permission was blocked" | click the camera or lock icon in the address bar → Allow → press Test again; Windows: Settings → Privacy & security → Camera → allow desktop/browser access |
| "camera/microphone is being used by another app" | close Zoom / Teams / Camera app / other browser tabs using it |
| Microphone meter doesn't move | Windows Settings → System → Sound → Input: choose the right mic; allow microphone for the browser |
| "Speak answer" greyed out | browser lacks speech recognition — use Edge or Chrome |
| Speech recognition error "network" | needs internet; otherwise type the answer |
| Voice steadiness shows "not measured" | you spoke < 1.5 s or very quietly; speak a full sentence near the mic |
| Port 5055 in use | close the other copy of the app (or restart the terminal) |
| `ModuleNotFoundError` | activate the venv (`(venv)` must show in the prompt) and `pip install -r requirements.txt` |
