"""Tests of wb.gpt: Holmes as character ids, batches, the sliding validation loss, the reference mini-GPT and the
single checkpoint of the trained model (bonus chapter B3)."""

import importlib
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from wb import gpt as G  # noqa: E402
from wb import text as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _reference(module: str):
    """A module of the reference library (solutions/mylearn_ref), imported by its own path (learner mode has no
    solutions/ on sys.path)."""
    if str(ROOT / "solutions") not in sys.path:
        sys.path.insert(0, str(ROOT / "solutions"))
    return importlib.import_module(f"mylearn_ref.{module}")


class Bigram(nn.Module):
    """A model whose logits at position t depend only on the token at t: any window gives the same loss."""

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.table(idx)


def test_char_data_is_the_frozen_split_and_the_vocabulary_of_build_vocab():
    data = G.char_data("holmes")
    train, val = W.train_val("holmes")
    assert data.train_text == train and data.val_text == val
    assert len(data.itos) == 88 == G.MINI_CONFIG["vocab_size"], "expected the 88 characters of Holmes (ch. 22)"
    stoi, itos = _reference("sequences").build_vocab(train + val)
    assert itos == data.itos and stoi == data.stoi, "expected the vocabulary of build_vocab on the whole text"
    assert data.train_ids.dtype == torch.long and len(data.train_ids) == len(train) and len(data.val_ids) == len(val)
    assert data.decode(data.val_ids[:500]) == val[:500] and data.decode(data.encode("Holmes")) == "Holmes"
    assert G.char_data("holmes") is data, "expected the data to be built once per session"
    assert data.fingerprint == G.split_fingerprint(train, val) and len(data.fingerprint) == 16
    with pytest.raises(ValueError, match="outside the vocabulary"):
        data.encode("€")


def test_get_batch_is_the_one_of_the_fiche():
    x, y = G.get_batch(torch.arange(1000), 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and torch.equal(y, x + 1)
    assert x[0].tolist() == list(range(204, 212)), "expected the output printed in the fiche (§203.8)"


@pytest.mark.parametrize("n, block, stride", [(129, 128, 32), (300, 128, 32), (1000, 128, 32), (1000, 16, 16),
                                              (517, 64, 1), (200, 128, 128)])
def test_sliding_windows_score_every_target_once_with_enough_context(n, block, stride):
    windows = G.sliding_windows(n, block, stride)
    scored = [p for _, first, last in windows for p in range(first, last + 1)]
    assert scored == list(range(block, n)), "expected each target from block_size to n - 1 exactly once, in order"
    for start, first, last in windows:
        assert 0 <= start and start + block <= n - 1, "expected the inputs and the targets inside the text"
        assert block - stride + 1 <= first - start and last - start <= block, "expected the context of the contract"


def test_sliding_windows_refuse_a_short_text_or_a_bad_stride():
    with pytest.raises(ValueError):
        G.sliding_windows(128, 128, 32)
    with pytest.raises(ValueError):
        G.sliding_windows(500, 128, 0)
    with pytest.raises(ValueError):
        G.sliding_windows(500, 128, 129)


def test_char_loss_equals_a_direct_computation_and_restores_the_mode():
    torch.manual_seed(0)
    model = Bigram(11).train()
    ids = torch.randint(0, 11, (700,))
    with torch.no_grad():
        direct = nn.functional.cross_entropy(model(ids[None, 127:-1])[0], ids[128:]).item()
    for stride in (1, 7, 32):
        assert math.isclose(G.char_loss(model, ids, 128, stride, batch_size=5), direct, rel_tol=1e-6)
    assert model.training, "expected the model back in train mode"


class Counter(nn.Module):
    """A model that always predicts the token after the last one (modulo V), with a small context."""

    def __init__(self, vocab_size, block_size=4):
        super().__init__()
        self.vocab_size, self.block_size, self.seen = vocab_size, block_size, []
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        self.seen.append(idx.shape[1])
        return nn.functional.one_hot((idx + 1) % self.vocab_size, self.vocab_size).float() * 10


def test_generate_is_greedy_crops_the_context_and_samples_with_the_generator():
    model = Counter(7).train()
    out = G.generate(model, torch.tensor([[0, 1], [5, 6]]), 6)
    assert out.tolist() == [[0, 1, 2, 3, 4, 5, 6, 0], [5, 6, 0, 1, 2, 3, 4, 5]], "expected greedy decoding"
    assert max(model.seen) == 4, "expected the context cropped to block_size"
    assert model.training, "expected the model back in train mode"
    assert torch.equal(G.generate(model, torch.tensor([[3]]), 0), torch.tensor([[3]]))
    torch.manual_seed(0)
    gpt = G.ReferenceGPT(vocab_size=11, block_size=8, n_layer=1, n_head=1, d_model=8)
    a = G.generate(gpt, torch.zeros(2, 1, dtype=torch.long), 20, 1.0, torch.Generator().manual_seed(3))
    b = G.generate(gpt, torch.zeros(2, 1, dtype=torch.long), 20, 1.0, torch.Generator().manual_seed(3))
    assert torch.equal(a, b) and a.shape == (2, 21), "expected the draws of the generator"
    assert len(set(a[0].tolist())) > 3, "expected random draws at temperature 1 from a nearly uniform model"
    with pytest.raises(ValueError, match="temperature"):
        G.generate(gpt, torch.zeros(1, 1, dtype=torch.long), 2, -1.0)


def test_reference_gpt_follows_the_contract():
    torch.manual_seed(0)
    model = G.ReferenceGPT(**G.MINI_CONFIG)
    assert sum(p.numel() for p in model.parameters()) == G.count_params() == 820_992
    shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    assert shapes == G.expected_state_shapes(), "expected the names and shapes of the contract of Ex B3.19"
    assert model.head.weight is model.tok_emb.weight
    assert not any(isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer)) for m in model.modules())
    data = G.char_data("holmes")
    x, y = G.get_batch(data.train_ids, 8, 128, torch.Generator().manual_seed(1))
    with torch.no_grad():
        loss = G.lm_loss(model(x), y).item()
        assert abs(loss - math.log(88)) < 0.05, f"expected an initial loss close to log V, got {loss:.4f}"
        changed = x.clone()
        changed[:, 64] = (changed[:, 64] + 1) % 88
        assert torch.allclose(model(x)[:, :64], model(changed)[:, :64]), "expected a causal model"
        with pytest.raises(ValueError, match="block_size"):
            model(torch.zeros(1, 129, dtype=torch.long))


def test_lr_at_is_cosine_lr_of_chapter_19():
    cosine_lr = _reference("optim").cosine_lr
    for it in (0, 1, 25, 50, 51, 300, 599, 600, 700):
        assert math.isclose(G.lr_at(it, 600, 2e-3, 50, 2e-4), cosine_lr(it, 600, 2e-3, 50, 2e-4), abs_tol=1e-15)


def test_configure_optimizer_decays_the_matrices_only():
    model = G.ReferenceGPT(**G.MINI_CONFIG)
    opt = G.configure_optimizer(model, 1e-3, 0.1)
    decay = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert set(decay) == {id(p) for p in model.parameters()} and len(decay) == len(list(model.parameters()))
    for p in model.parameters():
        assert decay[id(p)] == (0.1 if p.dim() >= 2 else 0.0)


def test_train_records_its_history():
    config = {"vocab_size": 88, "block_size": 16, "n_layer": 1, "n_head": 2, "d_model": 8, "dropout": 0.0}
    settings = G.train_settings(True) | {"iters": 7, "warmup": 2, "eval_interval": 3, "eval_batches": 2, "batch_size": 4}
    torch.manual_seed(0)
    model = G.ReferenceGPT(**config)
    data = G.char_data("holmes")
    history = G.train(model, data.train_ids, data.val_ids, settings, torch.Generator().manual_seed(0), block_size=16)
    assert len(history["train_loss"]) == len(history["lr"]) == 7 and history["val_iter"] == [0, 3, 6, 7]
    assert history["lr"][0] == 0.0 and math.isclose(history["lr"][2], settings["lr"])
    assert all(isinstance(v, float) for v in history["val_loss"])


def test_checkpoint_round_trip_and_refusals(tmp_path):
    config = {"vocab_size": 88, "block_size": 16, "n_layer": 1, "n_head": 2, "d_model": 8, "dropout": 0.0}
    torch.manual_seed(0)
    model = G.ReferenceGPT(**config)
    key = G.checkpoint_key(config, {"iters": 1}, fast=True, source="learner")
    assert G.load_checkpoint(key, tmp_path) is None
    history = {"train_loss": [4.5], "val_iter": [0], "val_loss": [4.4]}
    path = G.save_checkpoint(key, model, history, G.char_data().itos, tmp_path, extra={"char_loss": 4.3})
    assert path == G.checkpoint_file(key, tmp_path) and [p.name for p in tmp_path.iterdir()] == [path.name]
    ckpt = G.load_checkpoint(key, tmp_path)
    assert ckpt["config"] == config and ckpt["history"] == history and ckpt["extra"] == {"char_loss": 4.3}
    again = G.build_from_checkpoint(G.ReferenceGPT, ckpt)
    x = torch.randint(0, 88, (2, 16))
    with torch.no_grad():
        assert torch.equal(again(x), model.eval()(x)) and again.head.weight is again.tok_emb.weight
    other = G.checkpoint_key(config, {"iters": 2}, fast=True, source="learner")
    assert G.checkpoint_file(other, tmp_path) != path and G.load_checkpoint(other, tmp_path) is None
    assert G.checkpoint_file(G.checkpoint_key(config, {"iters": 1}, fast=False, source="learner"), tmp_path) != path
    path.write_bytes(path.read_bytes()[:200])                     # a truncated file: as if absent
    assert G.load_checkpoint(key, tmp_path) is None


def test_keys_are_complete():
    fast, full = G.learner_key(True), G.learner_key(False)
    assert fast != full and G.reference_key(True) != fast
    for key in (fast, full):
        assert {"config", "settings", "fast", "source", "split", "vocab"} <= set(key)
        assert key["settings"]["seed"] == G.SEED and key["split"] == G.char_data().fingerprint
    assert fast["settings"]["iters"] == 600 and full["settings"]["iters"] == 5000
    assert fast["config"]["dropout"] == 0.0 and full["config"]["dropout"] == 0.2
    assert {k: v for k, v in full["config"].items() if k != "dropout"} == {k: v for k, v in G.MINI_CONFIG.items() if k != "dropout"}


def test_reference_model_is_read_from_its_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(1)
    model = G.ReferenceGPT(**G.MINI_CONFIG)
    G.save_checkpoint(G.reference_key(True), model, {"val_loss": [1.0]}, G.char_data().itos, tmp_path)
    monkeypatch.setattr(G, "train", lambda *a, **k: pytest.fail("expected no training when the checkpoint exists"))
    again, ckpt = G.reference_model(True, tmp_path, verbose=False)
    assert isinstance(again, G.ReferenceGPT) and not again.training and ckpt["history"] == {"val_loss": [1.0]}
    x = torch.randint(0, 88, (1, 32))
    with torch.no_grad():
        assert torch.equal(again(x), model.eval()(x))


def test_the_solution_and_the_learner_never_share_a_checkpoint(tmp_path):
    learner, solution = G.learner_key(True), G.learner_key(True, source="solution")
    assert learner != solution and G.checkpoint_file(learner, tmp_path) != G.checkpoint_file(solution, tmp_path)
    torch.manual_seed(0)
    model = G.ReferenceGPT(**(G.MINI_CONFIG | {"n_layer": 1, "d_model": 8, "n_head": 2}))
    G.save_checkpoint(solution, model, {"train_loss": []}, G.char_data("holmes").itos, tmp_path)
    assert G.load_checkpoint(learner, tmp_path) is None, "expected the learner's key not to read the solution's model"
    assert G.delete_checkpoint(learner, tmp_path) is False
    assert G.delete_checkpoint(solution, tmp_path) is True and G.load_checkpoint(solution, tmp_path) is None
