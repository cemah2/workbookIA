"""The long training histories of chapter 24 (notebook exercises), too long to run in the notebook.

    python tools/chapters/history_ch24.py                 # every part (≈ 1 h 10 min on 2 CPUs)
    python tools/chapters/history_ch24.py cnn21 aug23     # some parts only; the others are kept

Writes tools/chapters/history_ch24.json, one entry per part:
- ``mlp14`` (📈 24.14): the MLP 784-32-32-10 with dropout 0.2 after each hidden layer and a max-norm of 3 on the
  hidden layers, Adam(1e-3), mini-batches of 256, 30 epochs on the 54 000 training images of the stratified split
  of chapter 23 (test_size=0.1, stratify=y, random_state=0), standardised with the training mean and std. Per
  epoch: the training loss and accuracy averaged during the epoch (train mode, what Keras prints), the same
  measured on the whole training set at the end of the epoch in train mode (dropout on) and in eval mode, and
  the validation loss and accuracy (eval mode). About 2 s per epoch.
- ``cnn21`` (🔬 24.21): the deep CNN of 📦 24.20 (two 3x3 convolutions and a 2x2 max pooling per stage, widths
  16, 32, 64, then Flatten and Linear(1024, 10)) in four variants: plain, dropout 0.25 after each pooling,
  batchnorm after each convolution, both. CIFAR-10: the first 20 000 training images, validation on the last
  5 000 of the training file; per-channel standardisation with the fiche's constants; Adam(1e-3), mini-batches of
  128, 20 epochs. About 15 s per epoch.
- ``aug23`` (📦 24.23): the deep CNN with batchnorm, without and with augmentation (RandomCrop(32, padding=4),
  RandomHorizontalFlip), same data, 25 epochs. About 15 to 20 s per epoch.
- ``frugal31`` (🏆 24.31): the reference recipe of the challenge in full mode: a small ResNet (a 3x3 stem of 16
  channels, three residual blocks of 16, 32 and 64 channels, the last two with stride 2, global average pooling,
  Linear(64, 10)), 78 042 parameters; augmentation (crop and mirror); SGD with Nesterov momentum 0.9, weight decay
  5e-4, OneCycleLR up to 0.1; label smoothing 0.1; mini-batches of 128; 30 epochs on the first 45 000 training
  images, validation on the last 5 000, then the test accuracy on the 10 000 test images. About 1 min per epoch.
- ``text30`` (📦 24.30): a character model (Embedding(V, 32), LSTM(128), Linear) and a word model (vocabulary of
  the 2 000 most frequent tokens plus <unk>, Embedding(V, 64), LSTM(128), Linear) on the whole Holmes text, the
  last 10 % of the tokens for validation, sequences of 64 characters or 32 tokens, Adam(3e-3), mini-batches of 64,
  gradient clipping at 1, 20 epochs each; per epoch the training and validation loss (nats per token); then
  three samples of 300 characters or 60 words at temperatures 0.5, 1 and 1.5. About 5 s per epoch.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import v2

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import wb  # noqa: E402
from mylearn_ref.sequences import build_vocab, encode, make_sequence_pairs, sample_from_logits  # noqa: E402

OUT = pathlib.Path(__file__).with_name("history_ch24.json")
MEAN, STD = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# Models (the same definitions as the reference solutions of the notebook)
# ---------------------------------------------------------------------------
def make_mlp(hidden=(784,), n_classes=10, p_drop=0.0):
    layers, n_in = [nn.Flatten()], 28 * 28
    for n_out in hidden:
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        n_in = n_out
    layers.append(nn.Linear(n_in, n_classes))
    return nn.Sequential(*layers)


def max_norm(model, c=3.0):
    linears = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][:-1]
    with torch.no_grad():
        for layer in linears:
            layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=c))


def make_deep_cnn(batchnorm=False, p_drop=0.0, widths=(16, 32, 64)):
    layers, c_in = [], 3
    for width in widths:
        for c in (c_in, width):
            layers.append(nn.Conv2d(c, width, 3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        c_in = width
    layers += [nn.Flatten(), nn.Linear(c_in * 4 * 4, 10)]
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return torch.relu(self.f(x) + self.shortcut(x))


def make_frugal_resnet(widths=(16, 32, 64), n_classes=10):
    layers = [nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU()]
    c_in = widths[0]
    for i, width in enumerate(widths):
        layers.append(ResidualBlock(c_in, width, stride=1 if i == 0 else 2))
        c_in = width
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, n_classes)]
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data and loops
# ---------------------------------------------------------------------------
def cifar_tensor(images):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    return ((x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)).contiguous()


class Images(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


AUGMENT = v2.Compose([v2.ToImage(), v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(),
                      v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


def evaluate(model, x, y, loss_fn, batch_size=1000):
    model.eval()
    total, correct = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            logits = model(x[i:i + batch_size])
            total += loss_fn(logits, y[i:i + batch_size]).item() * len(logits)
            correct += (logits.argmax(dim=1) == y[i:i + batch_size]).sum().item()
    return total / len(x), correct / len(x)


def train(model, loader, x_val, y_val, epochs, optimizer, loss_fn, schedule=None, after_step=None, label=""):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "seconds": []}
    eval_loss = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total, correct, count = 0.0, 0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                schedule.step()
            if after_step is not None:
                after_step(model)
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(xb)
        val_loss, val_acc = evaluate(model, x_val, y_val, eval_loss)
        seconds = time.perf_counter() - start
        for key, value in [("train_loss", total / count), ("train_acc", correct / count), ("val_loss", val_loss),
                           ("val_acc", val_acc), ("seconds", seconds)]:
            history[key].append(round(value, 5 if key != "seconds" else 1))
        log(f"{label} epoch {epoch:2d} · train {total / count:.4f} {correct / count:.4f} · val {val_loss:.4f} "
            f"{val_acc:.4f} · {seconds:.0f} s")
    return history


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def part_mlp14():
    X, y = wb.datasets.load_fashion_mnist("train", seed=None)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    mean, std = float((X_tr / 255.0).mean()), float((X_tr / 255.0).std())

    def prep(images):
        return torch.from_numpy(((images.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)).unsqueeze(1)

    x_tr, t_tr, x_val, t_val = prep(X_tr), torch.from_numpy(y_tr), prep(X_val), torch.from_numpy(y_val)
    torch.manual_seed(2414)
    model = make_mlp((32, 32), p_drop=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, t_tr), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(2414))
    keys = ["running_loss", "running_acc", "end_train_mode_loss", "end_train_mode_acc", "end_eval_mode_loss",
            "end_eval_mode_acc", "val_loss", "val_acc"]
    history = {key: [] for key in keys}
    for epoch in range(1, 31):
        model.train()
        total, correct, count = 0.0, 0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            max_norm(model, 3.0)
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(xb)
        model.train()                                     # the final weights, dropout on (no update)
        with torch.no_grad():
            torch.manual_seed(10_000 + epoch)
            logits = torch.cat([model(x_tr[i:i + 4096]) for i in range(0, len(x_tr), 4096)])
            tm_loss, tm_acc = loss_fn(logits, t_tr).item(), (logits.argmax(1) == t_tr).float().mean().item()
        em_loss, em_acc = evaluate(model, x_tr, t_tr, loss_fn, 4096)
        val_loss, val_acc = evaluate(model, x_val, t_val, loss_fn, 4096)
        for key, value in zip(keys, [total / count, correct / count, tm_loss, tm_acc, em_loss, em_acc, val_loss,
                                     val_acc]):
            history[key].append(round(value, 6))
        log(f"mlp14 epoch {epoch:2d} · running {correct / count:.4f} · end train mode {tm_acc:.4f} · eval mode "
            f"{em_acc:.4f} · val {val_acc:.4f}")
    meta = {"model": "MLP 784-32-32-10, dropout 0.2 after each hidden layer, max-norm 3 on the hidden layers",
            "optimizer": "Adam(lr=1e-3)", "batch_size": 256, "epochs": 30, "seed": 2414, "n_train": len(X_tr),
            "n_val": len(X_val)}
    return {"meta": meta, **history}


def cifar_split(n_train):
    X, y = wb.datasets.load_cifar10("train", seed=None)
    return X[:n_train], y[:n_train].astype(np.int64), X[45_000:], y[45_000:].astype(np.int64)


def part_cnn21():
    X_tr, y_tr, X_val, y_val = cifar_split(20_000)
    x_tr, t_tr, x_val, t_val = cifar_tensor(X_tr), torch.from_numpy(y_tr), cifar_tensor(X_val), torch.from_numpy(y_val)
    out = {"meta": {"model": "make_deep_cnn_20 (widths 16, 32, 64)", "optimizer": "Adam(lr=1e-3)", "batch_size": 128,
                    "epochs": 20, "seed": 2421, "n_train": 20_000, "n_val": 5_000, "p_drop": 0.25}}
    for name, kwargs in [("plain", {}), ("dropout", {"p_drop": 0.25}), ("batchnorm", {"batchnorm": True}),
                         ("both", {"batchnorm": True, "p_drop": 0.25})]:
        torch.manual_seed(2421)
        model = make_deep_cnn(**kwargs)
        loader = DataLoader(TensorDataset(x_tr, t_tr), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(2421))
        out[name] = train(model, loader, x_val, t_val, 20, torch.optim.Adam(model.parameters(), lr=1e-3),
                          nn.CrossEntropyLoss(), label=f"cnn21 {name}")
    return out


def part_aug23():
    X_tr, y_tr, X_val, y_val = cifar_split(20_000)
    x_tr, t_tr, x_val, t_val = cifar_tensor(X_tr), torch.from_numpy(y_tr), cifar_tensor(X_val), torch.from_numpy(y_val)
    out = {"meta": {"model": "make_deep_cnn_20(batchnorm=True)", "optimizer": "Adam(lr=1e-3)", "batch_size": 128,
                    "epochs": 25, "seed": 2423, "n_train": 20_000, "n_val": 5_000,
                    "augmentation": "RandomCrop(32, padding=4), RandomHorizontalFlip()"}}
    for name in ("without", "with"):
        torch.manual_seed(2423)
        model = make_deep_cnn(batchnorm=True)
        data = TensorDataset(x_tr, t_tr) if name == "without" else Images(X_tr, y_tr, AUGMENT)
        loader = DataLoader(data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(2423))
        out[name] = train(model, loader, x_val, t_val, 25, torch.optim.Adam(model.parameters(), lr=1e-3),
                          nn.CrossEntropyLoss(), label=f"aug23 {name}")
    return out


def part_frugal31(epochs=30):
    X, y = wb.datasets.load_cifar10("train", seed=None)
    X_test, y_test = wb.datasets.load_cifar10("test", seed=None)
    X_tr, y_tr = X[:45_000], y[:45_000].astype(np.int64)
    x_val, t_val = cifar_tensor(X[45_000:]), torch.from_numpy(y[45_000:].astype(np.int64))
    torch.manual_seed(2431)
    model = make_frugal_resnet()
    loader = DataLoader(Images(X_tr, y_tr, AUGMENT), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(2431))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=epochs * len(loader))
    history = train(model, loader, x_val, t_val, epochs, optimizer, nn.CrossEntropyLoss(label_smoothing=0.1),
                    schedule=schedule, label="frugal31")
    test_loss, test_acc = evaluate(model, cifar_tensor(X_test), torch.from_numpy(y_test.astype(np.int64)),
                                   nn.CrossEntropyLoss())
    log(f"frugal31 · {n_params(model)} parameters · test accuracy {test_acc:.4f}")
    meta = {"model": "small ResNet (stem 16, residual blocks 16, 32/2, 64/2, global average pooling)",
            "params": n_params(model), "optimizer": "SGD(momentum=0.9, nesterov=True, weight_decay=5e-4), "
            "OneCycleLR(max_lr=0.1)", "loss": "CrossEntropyLoss(label_smoothing=0.1)", "batch_size": 128,
            "epochs": epochs, "seed": 2431, "n_train": 45_000, "n_val": 5_000,
            "augmentation": "RandomCrop(32, padding=4), RandomHorizontalFlip()"}
    return {"meta": meta, **history, "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4)}


WORD_RE = re.compile(r"[a-z]+(?:['’][a-z]+)*|[^\sa-z]")


def tokenize_words(text):
    return WORD_RE.findall(text.lower())


class LM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, ids, state=None):
        output, state = self.lstm(self.emb(ids), state)
        return self.head(output), state


def lm_loss(model, X, Y, V, batch_size=256):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            logits, _ = model(X[i:i + batch_size])
            total += nn.functional.cross_entropy(logits.reshape(-1, V), Y[i:i + batch_size].reshape(-1),
                                                 reduction="sum").item()
    return total / Y.numel()


def generate(model, prompt, n, temperature, seed):
    rng = np.random.default_rng(seed)
    model.eval()
    out = list(prompt)
    with torch.inference_mode():
        logits, state = model(torch.tensor([prompt]))
        for _ in range(n):
            nxt = int(sample_from_logits(logits[0, -1].numpy(), temperature=temperature, rng=rng))
            out.append(nxt)
            logits, state = model(torch.tensor([[nxt]]), state)
    return out[len(prompt):]


def train_lm(ids, V, emb_dim, hidden, seq_len, label, epochs=20):
    n_val = len(ids) // 10
    X_tr, Y_tr = (torch.from_numpy(a.astype(np.int64)) for a in make_sequence_pairs(ids[:-n_val], seq_len))
    X_val, Y_val = (torch.from_numpy(a.astype(np.int64)) for a in make_sequence_pairs(ids[-n_val:], seq_len))
    torch.manual_seed(2430)
    model = LM(V, emb_dim, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(2430))
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            logits, _ = model(xb)
            loss = nn.functional.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * yb.numel()
            count += yb.numel()
        history["train_loss"].append(round(total / count, 5))
        history["val_loss"].append(round(lm_loss(model, X_val, Y_val, V), 5))
        log(f"{label} epoch {epoch:2d} · train {total / count:.4f} · val {history['val_loss'][-1]:.4f}")
    return model, history, len(X_tr) * seq_len, len(X_val) * seq_len


def part_text30():
    text = wb.datasets.load_holmes()
    out = {"meta": {"text": "load_holmes() (whole text)", "n_chars": len(text), "optimizer": "Adam(lr=3e-3)",
                    "batch_size": 64, "epochs": 20, "seed": 2430, "clip": 1.0, "val": "last 10 % of the tokens"}}
    stoi, itos = build_vocab(text)
    char_ids = encode(text, stoi)
    model, history, n_tr, n_val = train_lm(char_ids, len(itos), 32, 128, 64, "char")
    prompt = encode("Holmes", stoi).tolist()
    history["samples"] = {str(t): "Holmes" + "".join(itos[i] for i in generate(model, prompt, 300, t, 2430))
                          for t in (0.5, 1.0, 1.5)}
    out["char"] = {"vocab_size": len(itos), "emb_dim": 32, "hidden": 128, "seq_len": 64, **history}
    tokens = tokenize_words(text)
    stoi, itos = build_vocab(tokens, max_size=2001, unk_token="<unk>")
    word_ids = encode(tokens, stoi, unk_token="<unk>")
    model, history, n_tr, n_val = train_lm(word_ids, len(itos), 64, 128, 32, "word")
    prompt = encode(["holmes"], stoi, unk_token="<unk>").tolist()
    history["samples"] = {str(t): "holmes " + " ".join(itos[i] for i in generate(model, prompt, 60, t, 2430))
                          for t in (0.5, 1.0, 1.5)}
    out["word"] = {"vocab_size": len(itos), "n_tokens": len(tokens), "chars_per_token": round(len(text) / len(tokens), 4),
                   "unk_rate": round(float(np.mean(word_ids == 0)), 4), "emb_dim": 64, "hidden": 128, "seq_len": 32,
                   **history}
    return out


PARTS = {"mlp14": part_mlp14, "cnn21": part_cnn21, "aug23": part_aug23, "frugal31": part_frugal31,
         "text30": part_text30}


def main(argv) -> int:
    torch.set_num_threads(2)
    chosen = argv or list(PARTS)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in chosen:
        start = time.perf_counter()
        data[name] = PARTS[name]()
        data[name].setdefault("meta", {})["minutes"] = round((time.perf_counter() - start) / 60, 1)
        data[name]["meta"]["torch"] = torch.__version__
        data[name]["meta"]["generated_by"] = "tools/chapters/history_ch24.py"
        OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"✅ {name} written to {OUT.relative_to(ROOT)} ({(time.perf_counter() - start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
