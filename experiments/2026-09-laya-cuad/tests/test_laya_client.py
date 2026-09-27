from jev_bench.models.laya_client import QID, LayaClient, instructions_for
from jev_bench.runner import predict_chunks


class FakeAgent:
    def __init__(self, probs):
        self.probs = probs
        self.calls = []

    def predict_batch(self, states, questions, batch_size=None, max_len=None):
        self.calls.append((states, questions))
        return [{"answers": {QID: {"noul": p}}} for p in self.probs]


def make_client(probs, threshold=0.5):
    c = LayaClient.__new__(LayaClient)  # skip model download
    c.name, c.threshold, c.batch_size, c.max_len = "laya:test", threshold, 32, None
    c.agent = FakeAgent(probs)
    return c


def test_instructions_strip_generative_suffix():
    q = 'Does this contract contain a "X" clause? CUAD definition: d Answer Yes or No.'
    assert instructions_for(q) == 'Does this contract contain a "X" clause? CUAD definition: d'


def test_answer_many_thresholds_and_batches():
    c = make_client([0.2, 0.7, 0.5])
    preds = c.answer_many(["a", "b", "c"], "Q Answer Yes or No.")
    assert [p.answer for p in preds] == [False, True, True]
    assert len(c.agent.calls) == 1
    assert c.agent.calls[0][1][QID] == {"type": "noul", "instructions": "Q"}


def test_predict_chunks_uses_batch_path():
    c = make_client([0.1, 0.9])
    assert [p.answer for p in predict_chunks(c, ["a", "b"], "Q")] == [False, True]
