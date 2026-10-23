"""
Step 1: builds data/question_bank.csv by fetching REAL questions live from 6 public GitHub repos.
Run once:  python build_question_bank.py
Same logic as notebooks/00_question_bank.ipynb.
"""
import re, os, requests
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EXTRA = {'did','does','doing','explain','describe','tell','give','example',
         'difference','differentiate','what','which','how','why','when'}
STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA)

def extract_keywords(text, top_n=5):
    words = re.findall(r'[a-zA-Z][a-zA-Z\-]{2,}', text.lower())
    content = [w for w in words if w not in STOPWORDS]
    top = set(w for w, _ in Counter(content).most_common(top_n))
    seen = []
    for w in content:
        if w in top and w not in seen:
            seen.append(w)
    return seen or list(top)

def fetch(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

def parse_ml_repo(content):
    skip = {"Prepared and maintained by the **Founder** of [Outcome School](https://outcomeschool.com): Amit Shekhar",
            "Follow Amit Shekhar", "Follow Outcome School", "License"}
    section, out = None, []
    for line in content.split('\n'):
        line = line.rstrip()
        if line.startswith('### '):
            section = line[4:].strip()
        elif line.startswith('- ') and section and section not in skip:
            t = line[2:].strip()
            if re.match(r'^\[.+\]\(http', t) or t.startswith('Answer:') or len(t) < 10:
                continue
            out.append({'category': section, 'question': t})
    return out

def parse_backend_repo(content):
    sections = re.split(r"### \[\[↑\]\]\(#toc\) <a name=.+?>(.+?):</a>", content)
    out = []
    for i in range(1, len(sections), 2):
        cat = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ''
        items = re.split(r'#### (.+)', body)
        for j in range(1, len(items), 2):
            title = items[j].strip()
            text = ' '.join((items[j + 1] if j + 1 < len(items) else '').split())
            if title and len(title) > 2:
                out.append({'category': cat, 'question': title, 'answer_text': text})
    return out

def parse_swe_topics(content):
    out, cur = [], None
    section = content.split('## Data Structures')[1].split('## Video Lectures')[0]
    for line in section.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            cur = line[3:].strip()
        elif line.startswith('### ') or line.startswith('#### '):
            topic = line.lstrip('#').strip()
            if len(topic) > 2:
                out.append({'category': cur or 'Data Structures & Algorithms',
                            'question': f'Explain how {topic} works, its time complexity, and a common use case.',
                            'topic_name': topic})
    return out

def main(out=None):
    from paths import QUESTION_BANK
    out = str(out or QUESTION_BANK)
    print("Fetching real sources from GitHub...")
    ml_all = parse_ml_repo(fetch('https://raw.githubusercontent.com/amitshekhariitbhu/machine-learning-interview-questions/main/README.md'))
    ML_CATS = {'Algorithms','Deep Learning','Optimization','NLP','Computer Vision','Large Language Model','AI Engineering','Coding'}
    DS_CATS = {'Probability and Statistics','Model Evaluation','Data Preprocessing and Feature Engineering','Fundamentals of Machine Learning'}
    ml_q = [q for q in ml_all if q['category'] in ML_CATS]
    ds_q = [q for q in ml_all if q['category'] in DS_CATS]

    fe_files = {'general-questions':'General','html-questions':'HTML','css-questions':'CSS',
                'javascript-questions':'JavaScript','testing-questions':'Testing','performance-questions':'Performance'}
    fe_q = []
    for fname, cat in fe_files.items():
        c = fetch(f'https://raw.githubusercontent.com/h5bp/Front-end-Developer-Interview-Questions/main/src/questions/{fname}.md')
        for line in c.split('\n'):
            line = line.strip()
            if line.startswith('* ') or line.startswith('  * '):
                t = line.lstrip('* ').strip()
                if len(t) > 10 and not t.startswith('---'):
                    fe_q.append({'category': cat, 'question': t})

    be_q = parse_backend_repo(fetch('https://raw.githubusercontent.com/arialdomartini/Back-End-Developer-Interview-Questions/master/README.md'))
    swe_q = parse_swe_topics(fetch('https://raw.githubusercontent.com/kdn251/interviews/master/README.md'))
    for q in be_q:
        if q['category'] in ('Questions about logic and algorithms', 'Questions about Languages'):
            swe_q.append({'category': q['category'], 'question': q['question']})
    hr_q = re.findall(r'<summary><b>(.+?)</b></summary>',
                      fetch('https://raw.githubusercontent.com/ashishps1/awesome-behavioral-interviews/main/README.md'))

    rows = []
    def add(role, items, answer_field=None):
        for q in items:
            src = q.get(answer_field) if answer_field and q.get(answer_field) else q.get('topic_name', q['question'])
            ref = q.get('answer_text', '') if answer_field else ''
            if len(ref) < 40:                        # too short to be a real reference paragraph (e.g. 'Interview me.')
                ref = ''
            if len(ref) > 600:                       # trim to a readable paragraph, ending on a word boundary
                ref = ref[:600].rsplit(' ', 1)[0] + ' ...'
            rows.append({'role': role, 'category': q['category'], 'question': q['question'],
                         'expected_keywords': ', '.join(extract_keywords(src)), 'is_hr': False,
                         'reference': ref})
    add('ML Engineer', ml_q); add('Data Scientist', ds_q); add('Frontend Developer', fe_q)
    add('Backend Developer', be_q, 'answer_text'); add('Software Developer', swe_q)
    for q in hr_q:
        rows.append({'role': 'ALL (HR/Behavioral)', 'category': 'Behavioral', 'question': q,
                     'expected_keywords': '', 'is_hr': True, 'reference': ''})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df.to_csv(out, index=False)
    print('Saved', out, ' total =', len(df))
    print(df['role'].value_counts())
    return df

if __name__ == '__main__':
    main()
