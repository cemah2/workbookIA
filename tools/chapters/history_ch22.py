"""The training history of 📈 Ex 22.26 (chapter 22), too long to run in the notebook.

The character generator of Ex 22.25 in its full mode: the vocabulary of the whole Holmes text, windows of 40
characters every 3 characters (many-to-one: the target is the next character), two LSTM layers of 128 values,
RMSprop(lr=0.01, alpha=0.9) as the book's Keras code, gradient clipping at 1.0, mini-batches of 128. Training on the
first 90 % of the text, validation on the last 10 %. After each epoch: the mean training loss (as reported during the
epoch), the validation loss (evaluation mode), and three texts generated from the same seed at the temperatures 0, 0.5
and 1. About 90 seconds per epoch on 2 CPUs.

    python tools/chapters/history_ch22.py          # writes tools/chapters/history_ch22.json
"""

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import wb  # noqa: E402
from mylearn_ref import sequences as S  # noqa: E402

OUT = pathlib.Path(__file__).with_name("history_ch22.json")
EPOCHS = 20
SEED_TEXT = "To Sherlock Holmes she is always the wom"          # 40 characters, the start of the first story


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, ids):
        out, _ = self.lstm(F.one_hot(ids, self.vocab_size).float())
        return self.head(out[:, -1])


def generate(model, stoi, itos, n_chars, temperature, seed=0):
    rng = np.random.default_rng(seed)
    window = list(S.encode(SEED_TEXT, stoi))
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_chars):
            logits = model(torch.tensor([window[-40:]]))[0].numpy().astype(float)
            nxt = S.sample_from_logits(logits, temperature=temperature, rng=rng)
            out.append(nxt)
            window.append(nxt)
    return S.decode(np.array(out), itos)


def main() -> None:
    torch.set_num_threads(2)
    text = wb.datasets.load_holmes()
    stoi, itos = S.build_vocab(text)
    ids = S.encode(text, stoi)
    cut = int(0.9 * len(ids))
    X, y = S.make_windows(ids[:cut], 40, horizon=1, stride=3)
    Xv, yv = S.make_windows(ids[cut:], 40, horizon=1, stride=3)
    X, y, Xv, yv = (torch.from_numpy(X[:, :, 0]), torch.from_numpy(y), torch.from_numpy(Xv[:, :, 0]),
                    torch.from_numpy(yv))
    torch.manual_seed(2226)
    model = CharLSTM(len(itos))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
    generator = torch.Generator().manual_seed(2226)
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for i in range(0, len(X), 128):
            batch = order[i:i + 128]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = sum(F.cross_entropy(model(Xv[i:i + 1000]), yv[i:i + 1000], reduction="sum").item()
                      for i in range(0, len(Xv), 1000)) / len(Xv)
        samples = {str(t): generate(model, stoi, itos, 200, t) for t in (0.0, 0.5, 1.0)}
        history.append({"epoch": epoch, "train_loss": round(total / len(X), 4), "val_loss": round(val, 4),
                        "samples": samples})
        print(epoch, history[-1]["train_loss"], history[-1]["val_loss"], f"{time.perf_counter() - start:.0f} s",
              repr(samples["0.5"][:60]), flush=True)
        OUT.write_text(json.dumps({"vocab_size": len(itos), "n_train_windows": len(X), "n_val_windows": len(Xv),
                                   "history": history}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
