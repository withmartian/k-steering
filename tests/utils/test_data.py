import pytest

from k_steering.data.task_prompts import debates_prompts, tones_prompts

# Import the function under test
from k_steering.utils.data import load_task


class MockDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]
    

@pytest.fixture
def mock_dataset():
    return MockDataset(
        [
            {"id": "q1", "text": "What is AI?"},
            {"id": "q2", "text": "Explain transformers."},
        ]
    )


def test_load_task_tones(monkeypatch, mock_dataset):
    # Mock external deps
    monkeypatch.setattr(
        "k_steering.utils.data.load_dataset",
        lambda *args, **kwargs: mock_dataset,
    )

    dataset, labels, eval_prompts = load_task("tones")

    # ---- labels ----
    assert labels == sorted({t for t, _ in tones_prompts()})

    # ---- eval prompts ----
    assert eval_prompts == ["What is AI?", "Explain transformers."]

    # ---- dataset size ----
    # 2 questions × 5 tones
    assert len(dataset) == 10

    # ---- dataset schema ----
    sample = dataset[0]
    assert set(sample.keys()) == {
        "id",
        "original_question",
        "text",
        "label",
    }

    # ---- content correctness ----
    assert sample["original_question"] in eval_prompts
    assert sample["label"] in labels


def test_load_task_debates(monkeypatch, mock_dataset):
    monkeypatch.setattr(
        "k_steering.utils.data.load_dataset",
        lambda *args, **kwargs: mock_dataset,
    )

    dataset, labels, eval_prompts = load_task("debates")

    assert labels == sorted({t for t, _ in debates_prompts()})
    assert len(dataset) == 20
    assert eval_prompts == ["What is AI?", "Explain transformers."]


def test_load_task_with_max_samples(monkeypatch, mock_dataset):
    monkeypatch.setattr(
        "k_steering.utils.data.load_dataset",
        lambda *args, **kwargs: mock_dataset,
    )

    dataset, labels, eval_prompts = load_task("tones", max_samples=2)

    assert len(dataset) == 2
    assert len(eval_prompts) == 2
    assert labels == sorted({t for t, _ in tones_prompts()})


def test_load_task_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        load_task("unknown_task")
