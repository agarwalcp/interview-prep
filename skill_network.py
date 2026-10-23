"""
Per-role Bayesian skill networks (Unit I) + interview-state Markov chain (Unit IV).
Skill nodes come from the REAL category names in data/question_bank.csv.
Edges (prerequisites) and CPD numbers are hand-set priors, not learned from data.
"""
import itertools
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

ROLE_SKILL_GRAPHS = {
    'Frontend Developer': {
        'edges': [('html','css'), ('css','javascript'), ('javascript','testing'), ('javascript','performance')],
        'nodes': [],
        'category_map': {'HTML':'html','CSS':'css','JavaScript':'javascript','Testing':'testing',
                         'Performance':'performance','General':'javascript'},
    },
    'Backend Developer': {
        'edges': [('languages','design_patterns'), ('design_patterns','code_design'),
                  ('databases','distributed_systems'), ('design_patterns','architecture'),
                  ('architecture','distributed_systems'), ('architecture','security'),
                  ('distributed_systems','concurrency')],
        'nodes': [],
        'category_map': {
            'Questions about Languages':'languages', 'Questions about Design Patterns':'design_patterns',
            'Questions about Code Design':'code_design', 'Questions about Databases':'databases',
            'Questions about NoSQL':'databases', 'Questions about Distributed Systems':'distributed_systems',
            'Questions about Software Architecture':'architecture',
            'Questions about Service Oriented Architecture and Microservices':'architecture',
            'Questions about Web development':'architecture', 'Questions about Security':'security',
            'Questions about Concurrency':'concurrency', 'Questions about code versioning':'code_design',
            'Questions about Software Lifecycle and Team Management':'code_design',
            'Questions about logic and algorithms':'languages', 'General Questions':'architecture',
            'Open Questions':'architecture', 'Questions about snippets of code':'code_design',
            'Bill Gates Style Questions':'architecture'},
    },
    'ML Engineer': {
        'edges': [('algorithms','optimization'), ('optimization','deep_learning'),
                  ('deep_learning','nlp'), ('deep_learning','computer_vision'), ('deep_learning','llm')],
        'nodes': [],
        'category_map': {'Algorithms':'algorithms','Optimization':'optimization','Deep Learning':'deep_learning',
                         'NLP':'nlp','Computer Vision':'computer_vision','Large Language Model':'llm',
                         'AI Engineering':'llm','Coding':'algorithms'},
    },
    'Data Scientist': {
        'edges': [('fundamentals','probability_stats'), ('probability_stats','data_preprocessing'),
                  ('data_preprocessing','model_evaluation')],
        'nodes': [],
        'category_map': {'Fundamentals of Machine Learning':'fundamentals','Probability and Statistics':'probability_stats',
                         'Data Preprocessing and Feature Engineering':'data_preprocessing','Model Evaluation':'model_evaluation'},
    },
    'Software Developer': {
        'edges': [('data_structures','algorithms'), ('algorithms','runtime_analysis')],
        'nodes': ['oop'],   # independent root: no claimed prerequisite link
        'category_map': {'Data Structures & Algorithms':'data_structures','Algorithms':'algorithms',
                         'Runtime Analysis':'runtime_analysis','Questions about logic and algorithms':'algorithms',
                         'Questions about Languages':'oop'},
    },
}


class RoleSkillNetwork:
    def __init__(self, edges, extra_nodes=()):
        self.edges = edges
        self.model = DiscreteBayesianNetwork(edges)
        for n in extra_nodes:
            self.model.add_node(n)
        self._build_cpds()
        self.infer = VariableElimination(self.model)

    def _build_cpds(self):
        parents_of = {n: [] for n in self.model.nodes()}
        for p, c in self.edges:
            parents_of[c].append(p)
        cpds = []
        for node, parents in parents_of.items():
            if not parents:
                cpds.append(TabularCPD(node, 2, values=[[0.5], [0.5]]))
            else:
                combos = list(itertools.product([0, 1], repeat=len(parents)))
                pm = [0.2 + 0.6 * (sum(c) / len(c)) for c in combos]   # 0.2 none mastered .. 0.8 all
                cpds.append(TabularCPD(node, 2, values=[[1 - p for p in pm], pm],
                                       evidence=parents, evidence_card=[2] * len(parents)))
        self.model.add_cpds(*cpds)
        assert self.model.check_model()

    def query(self, target, evidence=None):
        return float(self.infer.query([target], evidence=evidence or {}, show_progress=False).values[1])

    def beliefs(self, evidence=None):
        evidence = evidence or {}
        return {n: (float(evidence[n]) if n in evidence else self.query(n, evidence))
                for n in self.model.nodes()}

    def check_independence(self, a, b, given=None):
        """True = d-connected (dependent), False = d-separated (independent)."""
        return self.model.is_dconnected(a, b, observed=given or [])


class InterviewMarkovChain:
    STATES = ['Confident', 'Struggling', 'Recovering']
    ON_CORRECT = {'Confident': [0.85, 0.05, 0.10], 'Struggling': [0.10, 0.20, 0.70], 'Recovering': [0.60, 0.10, 0.30]}
    ON_WRONG   = {'Confident': [0.30, 0.60, 0.10], 'Struggling': [0.05, 0.85, 0.10], 'Recovering': [0.10, 0.70, 0.20]}

    def __init__(self, start='Confident'):
        self.state, self.history = start, [start]

    def step(self, correct):
        probs = (self.ON_CORRECT if correct else self.ON_WRONG)[self.state]
        self.state = str(np.random.choice(self.STATES, p=probs))
        self.history.append(self.state)
        return self.state

    @classmethod
    def project(cls, p_correct, steps, start):
        """Analytic forward simulation: distribution over states after `steps` more answers,
        if each answer is correct with probability p_correct. Returns {state: probability}."""
        T = np.array([[p_correct * c + (1 - p_correct) * w for c, w in zip(cls.ON_CORRECT[st], cls.ON_WRONG[st])]
                      for st in cls.STATES])
        v = np.zeros(len(cls.STATES)); v[cls.STATES.index(start)] = 1.0
        for _ in range(steps):
            v = v @ T
        return dict(zip(cls.STATES, map(float, v)))


def build_all_networks():
    return {role: RoleSkillNetwork(cfg['edges'], cfg['nodes']) for role, cfg in ROLE_SKILL_GRAPHS.items()}


def check_category_coverage(question_bank):
    """Every real category in the question bank must map to a skill node."""
    problems = []
    for role, cfg in ROLE_SKILL_GRAPHS.items():
        cats = set(question_bank[question_bank.role == role].category)
        missing = cats - set(cfg['category_map'])
        nodes = set(n for e in cfg['edges'] for n in e) | set(cfg['nodes'])
        bad = set(cfg['category_map'].values()) - nodes
        if missing: problems.append((role, 'unmapped categories', missing))
        if bad: problems.append((role, 'maps to unknown node', bad))
    return problems


if __name__ == '__main__':
    import pandas as pd
    from paths import QUESTION_BANK
    qb = pd.read_csv(QUESTION_BANK)
    print('Coverage problems:', check_category_coverage(qb) or 'none')
    nets = build_all_networks()
    for role, n in nets.items():
        print(role, 'nodes:', len(n.model.nodes()))
    fe = nets['Frontend Developer']
    print('P(javascript) prior      :', round(fe.query('javascript'), 3))
    print('P(javascript | html ok)  :', round(fe.query('javascript', {'html': 1}), 3))
    print('testing vs performance, nothing known   (dependent?):', fe.check_independence('testing', 'performance'))
    print('testing vs performance, given javascript (dependent?):', fe.check_independence('testing', 'performance', ['javascript']))
