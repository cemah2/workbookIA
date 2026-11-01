"""Code cells of the chapter B3 notebook, second half (parts C to E: B3.21 to B3.30), shared by build_chB3.py (through
chB3_parts2.py) and history_chB3.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
A few GIVEN_* cells hold the placeholders «NAME», filled with the precomputed results of history_chB3.json by
chB3_parts2.py.
"""

# --- The protocols of the experiments (data, not cells: chB3_parts2.py writes them into the cells, history_chB3.py runs
# them with the solution) ------------------------------------------------------------------------------------------------
_FAST_SETTINGS = {"iters": 600, "batch_size": 32, "lr": 5e-3, "min_lr": 5e-4, "warmup": 50, "weight_decay": 0.1,
                  "beta2": 0.99, "clip": 1.0, "eval_interval": 1000, "eval_batches": 2, "seed": 2320, "eval_seed": 2321}
SMALL_CONFIG = {"vocab_size": 88, "block_size": 64, "n_layer": 2, "n_head": 4, "d_model": 64, "dropout": 0.0}
MINI = {"vocab_size": 88, "block_size": 128, "n_layer": 4, "n_head": 4, "d_model": 128, "dropout": 0.0}
PROTOCOLS = {
    "25": {"fast": {"config": SMALL_CONFIG, "settings": _FAST_SETTINGS | {"iters": 200, "warmup": 20},
                    "seeds": [2501, 2502], "val_chars": 20_000},
           "full": {"config": MINI, "settings": dict(_FAST_SETTINGS), "seeds": [2501, 2502, 2503], "val_chars": 56_237}},
    "26": {"fast": {"config": SMALL_CONFIG, "settings": _FAST_SETTINGS | {"iters": 200, "warmup": 20},
                    "seeds": [2601, 2602], "val_chars": 20_000, "char_block": 64},
           "full": {"config": MINI, "settings": dict(_FAST_SETTINGS), "seeds": [2320, 2321, 2322], "val_chars": 56_237,
                    "char_block": 128}},
}


# --- Part C: tools ----------------------------------------------------------------------------------------------------
TOOLS_C = r'''# Tools of part C: the mini-GPT trained by B3.20, reloaded from its checkpoint (never retrained here)
import re
import unicodedata

REFERENCE_FALLBACK = False   # True: if your model of B3.20 is missing, use the reference mini-GPT of wb.gpt (trained
                             # once with the recipe of B3.20, 2 to 3 min on CPU in fast mode, then reloaded)
N_HEAD = MINI_CONFIG["n_head"]
CKPT = wbgpt.load_checkpoint(wbgpt.learner_key(FAST_MODE))
GPT = None
if CKPT is not None:
    try:
        GPT = wbgpt.build_from_checkpoint(MiniGPT_19, CKPT)
        print(f"ℹ️ ton mini-GPT de B3.20, relu de {CKPT['path'].name} (wbgpt.checkpoints_dir()).")
    except Exception:  # noqa: BLE001 - MiniGPT_19 not defined in this session (a restarted kernel), or not the contract
        GPT = wbgpt.build_from_checkpoint(wbgpt.ReferenceGPT, CKPT)
        print(f"ℹ️ ton mini-GPT de B3.20, relu de {CKPT['path'].name} dans la classe ReferenceGPT de wb.gpt (ta classe "
              "MiniGPT_19 n'est pas définie dans cette session : mêmes noms, mêmes poids).")
elif REFERENCE_FALLBACK:
    GPT, CKPT = wbgpt.reference_model(FAST_MODE)
    print("ℹ️ ton mini-GPT de B3.20 manque : le mini-GPT de référence de wb.gpt le remplace dans la partie C.")
else:
    print("⏳ aucune sauvegarde de ton mini-GPT de B3.20 (" + ("mode rapide" if FAST_MODE else "mode complet") + ") : "
          "exécute les cellules de B3.19 et B3.20, ou mets REFERENCE_FALLBACK = True dans cette cellule pour prendre le "
          "mini-GPT de référence (2 à 3 min la première fois). Sur Colab, la sauvegarde s'efface avec la session.")


def need_gpt(ex_id):
    """Stop a check cell with ⏳ while no trained mini-GPT is loaded."""
    if GPT is None:
        print(f"⏳ Ex {ex_id} : cet exercice attend ton mini-GPT entraîné (B3.20), ou REFERENCE_FALLBACK = True dans la "
              "cellule des outils de la partie C.")
        raise StopCheck


def need(ex_id, *names):
    """Stop a check cell with ⏳ while a function or a class of the exercise is not defined yet."""
    for name in names:
        if globals().get(name) is None:
            print(f"⏳ Ex {ex_id} : définis d'abord {name}.")
            raise StopCheck'''

# --- 📈 B3.21 ----------------------------------------------------------------------------------------------------------
GIVEN_21 = r'''# Two full-mode trainings of the reference mini-GPT (5 000 iterations, precomputed by tools/chapters/history_chB3.py):
# the training loss averaged over blocks of 50 iterations, and the validation loss estimated every 250 iterations
FULL_21 = «FULL_21»


def block_means_21(values, width):
    """(centres, means) of consecutive blocks of `width` values: a smoothed training curve."""
    values = np.asarray(values, dtype=float)
    n = len(values) // width
    means = values[:n * width].reshape(n, width).mean(axis=1)
    return (np.arange(n) * width + width / 2), means


fig_21, axes_21 = plt.subplots(1, 3, figsize=(17, 4.4))
ax = axes_21[0]
if CKPT is None:
    ax.text(0.5, 0.5, "⏳ ton entraînement (B3.20)", ha="center", va="center", transform=ax.transAxes)
else:
    h_21 = CKPT["history"]
    x_21, m_21 = block_means_21(h_21["train_loss"], 10)
    ax.plot(x_21, m_21, color="tab:blue", label="entraînement (moyenne de 10 itérations)")
    ax.plot(h_21["val_iter"], h_21["val_loss"], "o-", color="tab:orange", label="validation (estimée)")
    lr_ax = ax.twinx()
    lr_ax.plot(np.arange(len(h_21["lr"])), h_21["lr"], ":", color="gray")
    lr_ax.set_ylabel("learning rate (pointillés)")
    ax.legend(loc="upper right")
ax.set(title="(1) ton entraînement (B3.20)", xlabel="itération", ylabel="loss (nats par caractère)")
ax.grid(alpha=0.3)
for name, color in (("sans dropout", "tab:red"), ("dropout 0,2", "tab:green")):
    run = FULL_21[name]
    axes_21[1].plot(run["train_iter"], run["train"], color=color, alpha=0.6, label=f"{name} : entraînement")
    axes_21[1].plot(run["val_iter"], run["val"], "o-", color=color, ms=3, label=f"{name} : validation")
    later = [i for i, it in enumerate(run["val_iter"]) if it >= 1000]
    axes_21[2].plot([run["val_iter"][i] for i in later], [run["val"][i] for i in later], "o-", color=color, ms=4,
                    label=f"{name} : validation")
axes_21[1].set(title="(2) mode complet, 5 000 itérations", xlabel="itération", ylabel="loss (nats par caractère)",
               ylim=(0.7, 2.6))
axes_21[2].set(title="(3) les deux validations, à partir de 1 000 itérations", xlabel="itération", ylabel="loss")
for ax in axes_21[1:]:
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()'''

TODO_21 = r'''vocab_21 = ...    # a) the loss at iteration 0 (graph 1) is close to log V: what is V? (a whole number)
fast_21 = ...     # b) your training (graph 1): "A" the validation loss is still going down at the end, the model could
                  #    learn more; "B" overfitting, the validation loss goes up again; "C" unstable, the loss jumps or diverges
best_21 = ...     # c) full mode without dropout: the iteration of the lowest validation loss: 1000, 2500, 4000 or 5000
early_21 = ...    # d) True or False: stopped at its best iteration, the run WITHOUT dropout would beat the run with
                  #    dropout at 5 000 iterations (a lower validation loss)
above_21 = ...    # e) True or False: in the run WITH dropout, the training curve ends above the validation curve'''

CHECK_21 = r'''for letter, answer in zip("abcde", [vocab_21, fast_21, best_21, early_21, above_21]):
    wb.check(f"B3.21{letter}", answer)'''

# --- 🔮 B3.22 ----------------------------------------------------------------------------------------------------------
GIVEN_22 = r'''VERNE = wbgpt.char_data("verne")             # Verne, « Le Tour du monde en quatre-vingts jours » (B2, ch. 22)
_TO_ENGLISH_22 = {"'": "’", "«": "“", "»": "”"}


def to_holmes_22(text):
    """The text in the vocabulary of Holmes: the apostrophe and the French quotes become those of Holmes, an accented
    letter that Holmes never uses loses its accent (ê -> e, ç -> c), any other unknown character is dropped."""
    out = []
    for ch in text:
        ch = _TO_ENGLISH_22.get(ch, ch)
        if ch not in DATA.stoi:
            ch = unicodedata.normalize("NFD", ch)[0]
        if ch in DATA.stoi:
            out.append(ch)
    return "".join(out)


PROMPT_FR_22 = to_holmes_22("Toutefois le vent ne fraîchit pas autant qu'on aurait pu le craindre. Ce ne fut pas un de "
                            "ces ouragans qui passent avec une vitesse de quatre-vingt-dix milles à l'heure. Il ")
PROMPT_EN_22 = ("“You have erred, perhaps,” he observed, taking up a glowing cinder with the tongs and lighting with it "
                "the long cherry-wood pipe. He ")
FRENCH_22 = DATA.encode(to_holmes_22(VERNE.val_text))      # the validation of Verne, in the vocabulary of Holmes
_words_h = set(re.findall(r"[^\W\d_]+", DATA.train_text.lower()))
_words_v = set(re.findall(r"[^\W\d_]+", to_holmes_22(VERNE.train_text).lower()))
ENGLISH_ONLY_22, FRENCH_ONLY_22 = _words_h - _words_v, _words_v - _words_h   # words seen in one novel only


def languages_22(text):
    """(English words, French words) of a text: the words seen in the training text of Holmes only, and those seen
    in the training text of Verne only (a word of both novels, such as « a » or « Fogg », counts for neither)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return sum(w in ENGLISH_ONLY_22 for w in words), sum(w in FRENCH_ONLY_22 for w in words)


@torch.no_grad()
def loss_by_position_22(model, ids, block_size=128, n_windows=64):
    """The mean loss (nats per character) of `model` at the positions 0 to 15, 16 to 63 and 64 to 127 of
    `n_windows` consecutive windows of `ids`: how the loss changes as the context of the window grows."""
    x = torch.stack([ids[i * block_size:(i + 1) * block_size] for i in range(n_windows)])
    y = torch.stack([ids[i * block_size + 1:(i + 1) * block_size + 1] for i in range(n_windows)])
    nll = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none").view(n_windows, -1)
    return [nll[:, a:b].mean().item() for a, b in ((0, 16), (16, 64), (64, block_size))]


print(f"amorce française ({len(PROMPT_FR_22)} caractères) : {PROMPT_FR_22!r}")
print(f"Verne en caractères de Holmes : {len(FRENCH_22):_} caractères de validation ; mots propres à Holmes "
      f"{len(ENGLISH_ONLY_22):_}, propres à Verne {len(FRENCH_ONLY_22):_}.".replace("_", " "))'''

TODO_22 = r'''french_22 = ...   # a) True or False: of 10 continuations of 120 characters drawn after the French prompt, MORE
                  #    THAN HALF have more French words than English words
uniform_22 = ...  # b) True or False: on the French text of Verne, the loss per character of your model of Holmes is LOWER
                  #    than that of a uniform prediction over the 88 characters (log 88 nats per character)'''

EXPERIMENT_22 = r'''with checked("B3.22"):
    need_gpt("B3.22")
    rows_22 = {}
    for name, prompt in (("amorce française", PROMPT_FR_22), ("amorce anglaise (témoin)", PROMPT_EN_22)):
        start_22 = DATA.encode(prompt)[None].repeat(10, 1)                       # 10 copies, 10 different draws
        out_22 = wbgpt.generate(GPT, start_22, 120, temperature=1.0, generator=torch.Generator().manual_seed(22))
        texts_22 = [DATA.decode(row[len(prompt):]) for row in out_22]
        counts_22 = [languages_22(t) for t in texts_22]
        rows_22[name] = sum(fr_ > en for en, fr_ in counts_22)
        print(f"{name} : {rows_22[name]} continuation(s) sur 10 ont plus de mots français que de mots anglais "
              f"(en moyenne {np.mean([c[0] for c in counts_22]):.1f} mots anglais et "
              f"{np.mean([c[1] for c in counts_22]):.1f} mots français par continuation). Les deux premières :")
        for t in texts_22[:2]:
            print("   ", repr(t[:160]))
    fr_loss_22 = wbgpt.char_loss(GPT, FRENCH_22)
    en_loss_22 = (CKPT.get("extra") or {}).get("char_loss") or wbgpt.char_loss(GPT, DATA.val_ids)
    print(f"\nloss sur toute la validation : Holmes {fr(en_loss_22, 3)}, Verne {fr(fr_loss_22, 3)} nats par caractère "
          f"(uniforme sur les {len(DATA.itos)} caractères : {fr(math.log(len(DATA.itos)), 3)})")
    for name, ids in (("Holmes", DATA.val_ids), ("Verne", FRENCH_22)):
        a, b, c = loss_by_position_22(GPT, ids)
        print(f"{name:>6} : loss aux positions 0-15 {fr(a, 3)}, 16-63 {fr(b, 3)}, 64-127 {fr(c, 3)}")
    wb.check("B3.22a", french_22)
    wb.check("B3.22b", uniform_22)'''

# --- 📦 B3.23 ----------------------------------------------------------------------------------------------------------
TODO_23 = r'''@torch.no_grad()
def generate_23(model, idx, n_new):
    """Greedy generation WITHOUT cache (contract of Ex B3.23): idx (N, T0) int64 -> (N, T0 + n_new)."""
    raise NotImplementedError("generate_23")


@torch.no_grad()
def generate_cached_23(model, idx, n_new, n_head):
    """The same tokens as generate_23, with a KV cache (contract of Ex B3.23); T0 + n_new <= model.block_size."""
    raise NotImplementedError("generate_cached_23")'''

SOLUTION_23 = r'''@torch.no_grad()
def generate_23(model, idx, n_new):
    """Greedy generation WITHOUT cache (contract of Ex B3.23): idx (N, T0) int64 -> (N, T0 + n_new)."""
    model.eval()
    for _ in range(n_new):
        logits = model(idx[:, -model.block_size:])[:, -1]           # the whole window again, at every step
        idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return idx


@torch.no_grad()
def generate_cached_23(model, idx, n_new, n_head):
    """The same tokens as generate_23, with a KV cache (contract of Ex B3.23); T0 + n_new <= model.block_size."""
    N, T0 = idx.shape
    if T0 + n_new > model.block_size:
        raise ValueError(f"{T0} + {n_new} tokens: more than block_size = {model.block_size} learned positions")
    model.eval()
    cache = [None] * len(model.blocks)                                # (keys, values) of each block, (N, H, t, d_h)

    def forward_new(tokens, start):
        """Logits of the last new token; the keys and values of the new tokens are appended to the cache."""
        T = tokens.shape[1]
        x = model.tok_emb(tokens) + model.pos_emb(torch.arange(start, start + T, device=tokens.device))
        for i, block in enumerate(model.blocks):
            D = x.shape[-1]
            q, k, v = block.attn.qkv(block.ln1(x)).split(D, dim=2)
            q, k, v = (t.view(N, T, n_head, D // n_head).transpose(1, 2) for t in (q, k, v))
            if cache[i] is not None:
                k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
            cache[i] = (k, v)
            S = k.shape[2]                                            # the prompt: causal; then 1 query, every key
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + block.attn.proj(y.transpose(1, 2).reshape(N, T, D))
            x = x + block.mlp(block.ln2(x))
        return model.head(model.ln_f(x[:, -1]))

    out = [idx]
    logits = forward_new(idx, 0)                                      # the prefill: the whole prompt at once
    for j in range(n_new):
        nxt = logits.argmax(dim=-1, keepdim=True)
        out.append(nxt)
        if j < n_new - 1:
            logits = forward_new(nxt, T0 + j)                         # one token, at its position
    return torch.cat(out, dim=1)'''

CHECK_23 = r'''cache_ok_23 = False
with checked("B3.23"):
    # 1. a small model of wb.gpt (not your B3.19), with a large token table: well-separated logits, no near tie
    torch.manual_seed(23)
    small_23 = wbgpt.ReferenceGPT(vocab_size=88, block_size=32, n_layer=2, n_head=2, d_model=16).eval()
    nn.init.normal_(small_23.tok_emb.weight, std=1.0)
    prompt_23 = torch.randint(0, 88, (3, 5), generator=torch.Generator().manual_seed(23))
    plain_23 = generate_23(small_23, prompt_23.clone(), 40)
    if not returned("B3.23", "generate_23", plain_23):
        raise StopCheck
    want_23 = wbgpt.generate(small_23, prompt_23.clone(), 40)
    plain_ok_23 = isinstance(plain_23, torch.Tensor) and tuple(plain_23.shape) == (3, 45) and torch.equal(plain_23, want_23)
    verdict("B3.23", plain_ok_23,
            "generate_23 : 40 tokens gloutons après 5, le contexte coupé aux 32 derniers tokens (block_size).",
            f"generate_23(model, idx (3, 5), 40) doit rendre les ids (3, 45) du tirage glouton (l'argmax des logits de la "
            f"dernière position, sur les block_size = 32 derniers tokens) ; reçu "
            f"{tuple(plain_23.shape) if isinstance(plain_23, torch.Tensor) else type(plain_23).__name__}"
            + (", d'autres tokens" if isinstance(plain_23, torch.Tensor) and tuple(plain_23.shape) == (3, 45) else "") + ".")
    # 2. the cache: the same tokens, and a cache really used (qkv sees the prompt once, then one token at a time)
    calls_23, hooks_23 = {"qkv": [], "whole": 0}, []
    for block in small_23.blocks:
        hooks_23.append(block.attn.qkv.register_forward_hook(lambda m, inp, out: calls_23["qkv"].append(inp[0].shape[1])))
        for module in (block, block.attn):
            hooks_23.append(module.register_forward_hook(lambda m, inp, out: calls_23.__setitem__("whole", calls_23["whole"] + 1)))
    hooks_23.append(small_23.register_forward_hook(lambda m, inp, out: calls_23.__setitem__("whole", calls_23["whole"] + 1)))
    try:
        cached_23 = generate_cached_23(small_23, prompt_23.clone(), 27, 2)
    finally:
        for hook in hooks_23:
            hook.remove()
    if not returned("B3.23", "generate_cached_23", cached_23):
        raise StopCheck
    want_c_23 = wbgpt.generate(small_23, prompt_23.clone(), 27)
    same_23 = isinstance(cached_23, torch.Tensor) and tuple(cached_23.shape) == (3, 32) and torch.equal(cached_23, want_c_23)
    verdict("B3.23", same_23, "generate_cached_23 : les mêmes 27 tokens que sans cache (contexte plein : 32 tokens).",
            "generate_cached_23(model, idx (3, 5), 27, 2) doit rendre les mêmes ids (3, 32) que le tirage glouton sans "
            "cache ; reçu " + (f"{tuple(cached_23.shape)}" if isinstance(cached_23, torch.Tensor) else type(cached_23).__name__)
            + (", d'autres tokens : vérifie les positions des nouveaux tokens et le masque du prefill"
               if isinstance(cached_23, torch.Tensor) and tuple(cached_23.shape) == (3, 32) else "") + ".")
    want_calls_23 = [(sorted([5] * 2 + [1] * 2 * steps)) for steps in (26, 27)]   # per block: the prompt, then 1 token
    used_23 = calls_23["whole"] == 0 and sorted(calls_23["qkv"]) in want_calls_23     # per step (the last one optional)
    verdict("B3.23", used_23,
            "le cache sert : chaque attn.qkv voit le prompt une fois (5 tokens), puis un seul token par pas ; le modèle, "
            "les blocs et les modules d'attention ne sont jamais appelés sur la séquence.",
            f"avec un KV cache, chaque attn.qkv doit voir le prompt une fois (5 tokens), puis 1 token à chacun des 26 pas "
            f"suivants, et ni le modèle, ni un bloc, ni un module d'attention ne doivent être appelés ; vu {calls_23['whole']} "
            f"appel(s) de modèle, de bloc ou d'attention, et des appels de qkv sur {sorted(set(calls_23['qkv']))} tokens.")
    refuses("B3.23", "5 + 28 tokens pour un contexte de 32", lambda: generate_cached_23(small_23, prompt_23, 28, 2))
    cache_ok_23 = plain_ok_23 and same_23 and used_23'''

EXPERIMENT_23 = r'''with checked("B3.23"):
    need_gpt("B3.23")
    if not cache_ok_23:
        print("⏳ Ex B3.23 : la mesure attend les contrôles de la cellule précédente.")
        raise StopCheck
    starts_23 = DATA.val_ids[1000:1000 + 4 * 32].view(4, 32)                 # 4 prompts of 32 characters
    rows_23 = []
    for n_new in (16, 32, 64, 96):
        timings_23 = {}
        for name, fn in (("sans cache", lambda: generate_23(GPT, starts_23.clone(), n_new)),
                         ("avec cache", lambda: generate_cached_23(GPT, starts_23.clone(), n_new, N_HEAD))):
            best_23, ids_23 = math.inf, None
            for _ in range(3):                                                # the best of three runs
                t0_23 = time.perf_counter()
                ids_23 = fn()
                best_23 = min(best_23, time.perf_counter() - t0_23)
            timings_23[name] = (best_23, ids_23)
        same_ids_23 = torch.equal(timings_23["sans cache"][1], timings_23["avec cache"][1])
        rows_23.append((n_new, timings_23["sans cache"][0], timings_23["avec cache"][0], same_ids_23))
    print(f"{'tokens générés':>15} {'sans cache':>12} {'avec cache':>12} {'gain':>6}  mêmes tokens")
    for n_new, plain, cached, same_ids in rows_23:
        print(f"{n_new:>15} {fr(1000 * plain, 0) + ' ms':>12} {fr(1000 * cached, 0) + ' ms':>12} "
              f"{fr(plain / cached, 1):>6}  {same_ids}")
    verdict("B3.23", all(r[3] for r in rows_23), "avec et sans cache, les mêmes tokens à toutes les longueurs.",
            "à une longueur au moins, les tokens générés avec et sans cache diffèrent : relis les contrôles ci-dessus.")
    verdict("B3.23", rows_23[-1][1] > rows_23[-1][2],
            f"à {rows_23[-1][0]} tokens, le cache va {fr(rows_23[-1][1] / rows_23[-1][2], 1)} fois plus vite (une mesure : "
            "elle dépend de la machine).",
            f"à {rows_23[-1][0]} tokens, la génération avec cache n'est pas plus rapide : la machine est-elle chargée ? "
            "Relance la cellule.")
    plt.figure(figsize=(6, 3.8))
    plt.plot([r[0] for r in rows_23], [1000 * r[1] for r in rows_23], "o-", label="sans cache")
    plt.plot([r[0] for r in rows_23], [1000 * r[2] for r in rows_23], "o-", label="avec cache")
    plt.xlabel("tokens générés (après 32 tokens d'amorce)")
    plt.ylabel("temps (ms, 4 séquences)")
    plt.title("Génération gloutonne du mini-GPT sur CPU")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    print("génération gloutonne, amorce 0 :", repr(DATA.decode(timings_23["avec cache"][1][0])))'''

# --- 📈 B3.24 ----------------------------------------------------------------------------------------------------------
GIVEN_24 = r'''SENTENCE_24 = "“To the man who loves art for its own sake,” remarked Sherlock Holmes."
WINDOWS_24 = DATA.val_ids[:16 * 128].view(16, 128)        # 16 windows of the validation, for the measures


@torch.no_grad()
def attention_maps_24(model, ids, n_head=N_HEAD):
    """The attention weights of every head of every block: ids (N, T) -> (N, n_layer, n_head, T, T). The attention
    module calls SDPA, which does not return its weights: they are recomputed from the input of each attention (the
    output of ln1, read by a hook), attn.qkv (queries, keys), the scale 1/sqrt(d_h) and the causal mask."""
    inputs, hooks = {}, []
    for i, block in enumerate(model.blocks):
        hooks.append(block.ln1.register_forward_hook(lambda m, inp, out, i=i: inputs.__setitem__(i, out)))
    try:
        model.eval()(ids)
    finally:
        for hook in hooks:
            hook.remove()
    N, T = ids.shape
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    maps = []
    for i, block in enumerate(model.blocks):
        D = inputs[i].shape[-1]
        q, k, _ = block.attn.qkv(inputs[i]).split(D, dim=2)
        q, k = (t.view(N, T, n_head, D // n_head).transpose(1, 2) for t in (q, k))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D // n_head)
        maps.append(torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1))
    return torch.stack(maps, dim=1)


def plot_maps_24(maps, text):
    """The (n_layer, n_head) grid of the attention maps of one sequence: a row per query, a column per key."""
    n_layer, n_head = maps.shape[:2]
    fig, axes = plt.subplots(n_layer, n_head, figsize=(3.6 * n_head, 3.6 * n_layer))
    chars = [c if c != " " else "␣" for c in text]
    for l in range(n_layer):
        for h in range(n_head):
            ax = axes[l, h]
            ax.imshow(maps[l, h].numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"bloc {l}, tête {h}", fontsize=10)
            ax.set_xticks(range(len(chars)))
            ax.set_xticklabels(chars, fontsize=5)
            ax.set_yticks(range(len(chars)))
            ax.set_yticklabels(chars, fontsize=5)
    plt.tight_layout()
    plt.show()


if GPT is None:
    print("⏳ Ex B3.24 : les cartes attendent ton mini-GPT entraîné (B3.20), ou REFERENCE_FALLBACK = True dans la cellule "
          "des outils de la partie C.")
else:
    ids_24 = DATA.encode(SENTENCE_24)[None]
    plot_maps_24(attention_maps_24(GPT, ids_24)[0], SENTENCE_24)'''

TODO_24 = r'''previous_24 = ...   # a) (block, head), both counted from 0: the head that looks the most at the PREVIOUS character
diagonal_24 = ...   # b) (block, head): the head that looks the most at the character ITSELF
spread_24 = ...     # c) (block, head): the head whose attention is the most SPREAD over the keys (the least peaked)'''

CHECK_24 = r'''with checked("B3.24"):
    need_gpt("B3.24")
    maps_24 = attention_maps_24(GPT, WINDOWS_24)                             # (16, n_layer, H, 128, 128)
    rows_24 = torch.arange(1, 128)
    scores_24 = {
        "previous": maps_24[..., rows_24, rows_24 - 1].mean(dim=(0, -1)),     # (n_layer, H): weight on j = i - 1
        "diagonal": maps_24.diagonal(dim1=-2, dim2=-1)[..., 1:].mean(dim=(0, -1)),   # weight on j = i
        "spread": -(maps_24 * maps_24.clamp_min(1e-12).log()).sum(-1)[..., 32:].mean(dim=(0, -1)),   # entropy, rows >= 32
    }
    n_layer_24, n_head_24 = scores_24["previous"].shape
    what_24 = {"previous": ("a)", "de poids moyen au caractère précédent"),
               "diagonal": ("b)", "de poids moyen au caractère lui-même"),
               "spread": ("c)", "nats d'entropie moyenne (une attention uniforme sur 64 clés : 4,16)")}
    claims_24 = {"previous": previous_24, "diagonal": diagonal_24, "spread": spread_24}
    good_24 = 0
    for key, claim in claims_24.items():
        letter, unit = what_24[key]
        if claim is ... or claim is None:
            print(f"⏳ Ex B3.24 {letter} : écris ton choix (bloc, tête).")
            continue
        try:
            l_24, h_24 = (int(v) for v in claim)
            if not (0 <= l_24 < n_layer_24 and 0 <= h_24 < n_head_24):
                raise ValueError
        except (TypeError, ValueError):
            print(f"❌ Ex B3.24 {letter} : attendu un couple (bloc, tête) d'entiers, bloc de 0 à {n_layer_24 - 1} et tête de "
                  f"0 à {n_head_24 - 1} ; reçu {claim!r}.")
            continue
        s_24, best_24 = scores_24[key][l_24, h_24].item(), scores_24[key].max().item()
        ok_24 = s_24 >= 0.9 * best_24
        good_24 += ok_24
        verdict("B3.24", ok_24,
                f"{letter} la tête ({l_24}, {h_24}) : {fr(s_24, 2)} {unit} sur 16 fenêtres de la validation, la plus forte des "
                f"{n_layer_24 * n_head_24} (à 10 % près).",
                f"{letter} la tête ({l_24}, {h_24}) : {fr(s_24, 2)} {unit} sur 16 fenêtres de la validation ; une autre tête "
                "en a nettement plus : relis les cartes.")
    if good_24 == 3:
        print("\nLes trois mesures, tête par tête (lignes : blocs ; colonnes : têtes) :")
        for key, (letter, unit) in what_24.items():
            print(f"{letter} {unit} :")
            print(np.round(scores_24[key].numpy(), 2))'''

# --- 🔬 B3.25 ----------------------------------------------------------------------------------------------------------
GIVEN_25 = r'''VARIANTS_25 = ("reference", "no_position", "no_residual", "post_ln", "one_head")
LABELS_25 = {"reference": "référence", "no_position": "sans position", "no_residual": "sans résiduel",
             "post_ln": "post-LN", "one_head": "une seule tête"}
# The protocol of this notebook: fast mode, a small model (2 blocks, d = 64, contexts of 64 characters), 200 iterations,
# 2 seeds (≈ 1 min); full mode (🚀), the mini-GPT of B3.20, 600 iterations, 3 seeds (≈ 35 min on CPU)
PROTOCOL_25 = («PROTOCOL_FAST_25» if FAST_MODE else «PROTOCOL_FULL_25»)
# The full protocol, precomputed by tools/chapters/history_chB3.py with the solution: nats per character on the whole
# validation of Holmes, one value per seed
FULL_25 = «FULL_25»'''

TODO_25 = r'''def make_variant_25(name, config):
    """The mini-GPT of B3.19 with one ablation (contract of Ex B3.25): name in VARIANTS_25, config a dict like
    MINI_CONFIG; returns a new model, nanoGPT initialisation, forward(idx) -> logits (N, T, vocab_size)."""
    raise NotImplementedError("make_variant_25")'''

SOLUTION_25 = r'''class AblatedBlock_25(Block_19):
    """Block_19 (pre-LN, residual), post-LN (residual after each sub-layer, then the layer norm), or without residual."""

    def __init__(self, d_model, n_head, residual=True, pre_ln=True):
        super().__init__(d_model, n_head)
        self.residual, self.pre_ln = residual, pre_ln

    def forward(self, x):
        if not self.pre_ln:                                  # post-LN (2017): the layer norm after the addition
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.mlp(x))
        if not self.residual:                                # each sub-layer replaces its input
            x = self.attn(self.ln1(x))
            return self.mlp(self.ln2(x))
        return super().forward(x)


class AblatedGPT_25(nn.Module):
    """MiniGPT_19 with optional learned positions and AblatedBlock_25 blocks."""

    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, positions=True, residual=True, pre_ln=True,
                 dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model) if positions else None    # None: no parameter at all
        self.blocks = nn.ModuleList([AblatedBlock_25(d_model, n_head, residual, pre_ln) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        for module in self.modules():                                              # nanoGPT initialisation
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx):
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"{T} tokens, more than block_size = {self.block_size}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_variant_25(name, config):
    """The mini-GPT of B3.19 with one ablation (contract of Ex B3.25): name in VARIANTS_25, config a dict like
    MINI_CONFIG; returns a new model, nanoGPT initialisation, forward(idx) -> logits (N, T, vocab_size)."""
    if name not in VARIANTS_25:
        raise ValueError(f"unknown variant {name!r}: expected one of {VARIANTS_25}")
    c = {k: config[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "d_model")}
    if name == "one_head":
        c["n_head"] = 1
    return AblatedGPT_25(**c, positions=name != "no_position", residual=name != "no_residual",
                         pre_ln=name != "post_ln")'''

CHECK_25 = r'''variants_ok_25 = False
SMALL_25 = {"vocab_size": 88, "block_size": 16, "n_layer": 2, "n_head": 2, "d_model": 16, "dropout": 0.0}
with checked("B3.25"):
    need("B3.25", "make_variant_25")
    x_25 = torch.randn(2, 6, 16)
    ids_25 = torch.randint(0, 88, (2, 12), generator=torch.Generator().manual_seed(25))
    ok_25 = []
    for name in VARIANTS_25:
        torch.manual_seed(25)
        model = make_variant_25(name, SMALL_25)
        if not isinstance(model, nn.Module):
            print(f"❌ Ex B3.25 : make_variant_25({name!r}, …) doit rendre un nn.Module ; reçu {type(model).__name__}.")
            ok_25.append(False)
            continue
        model.eval()
        want_25 = wbgpt.expected_state_shapes(SMALL_25)
        if name == "no_position":
            want_25.pop("pos_emb.weight")
        got_25 = {n: tuple(t.shape) for n, t in model.state_dict().items()}
        names_ok_25 = got_25 == want_25
        problems_25 = [] if names_ok_25 else [f"state_dict : manquants {sorted(set(want_25) - set(got_25))[:3]}, en trop "
                                              f"{sorted(set(got_25) - set(want_25))[:3]}"]
        if forbidden_in(model):
            problems_25.append(f"modules interdits {forbidden_in(model)}")
        with torch.no_grad():
            if names_ok_25:
                b = model.blocks[0]
                if name == "post_ln":
                    h = b.ln1(x_25 + b.attn(x_25))
                    want_b, rule = b.ln2(h + b.mlp(h)), "h = ln1(x + attn(x)), puis ln2(h + mlp(h))"
                elif name == "no_residual":
                    want_b, rule = b.mlp(b.ln2(b.attn(b.ln1(x_25)))), "mlp(ln2(attn(ln1(x)))), sans addition"
                else:
                    h = x_25 + b.attn(b.ln1(x_25))
                    want_b, rule = h + b.mlp(b.ln2(h)), "h = x + attn(ln1(x)), puis h + mlp(ln2(h))"
                if not torch.allclose(b(x_25), want_b, atol=1e-6):
                    problems_25.append(f"un bloc doit calculer {rule}")
                heads_25 = 1 if name == "one_head" else 2                       # the attention, recomputed from qkv
                q, k, v = b.attn.qkv(x_25).split(16, dim=2)
                q, k, v = (t.view(2, 6, heads_25, 16 // heads_25).transpose(1, 2) for t in (q, k, v))
                ref_attn = b.attn.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 6, 16))
                if not torch.allclose(b.attn(x_25), ref_attn, atol=1e-6):
                    problems_25.append(f"l'attention doit avoir {heads_25} tête(s)")
            logits = model(ids_25)
            if tuple(logits.shape) != (2, 12, 88):
                problems_25.append(f"logits {tuple(logits.shape)} au lieu de (2, 12, 88)")
            else:
                rep = model(torch.full((1, 12), 7))[0]
                flat = (rep[1:] - rep[:1]).abs().max().item() < 1e-5
                if name == "no_position" and not flat:
                    problems_25.append("sans position, un même caractère répété doit donner les mêmes logits partout")
                if name != "no_position" and flat:
                    problems_25.append("les positions apprises doivent servir")
                changed = ids_25.clone()
                changed[:, 8] = (changed[:, 8] + 1) % 88
                if (model(changed)[:, :8] - logits[:, :8]).abs().max().item() != 0:
                    problems_25.append("le modèle voit le futur")
        if abs(model.tok_emb.weight.std().item() / 0.02 - 1) > 0.15 or model.head.weight is not model.tok_emb.weight:
            problems_25.append("tête liée et initialisation de nanoGPT attendues")
        verdict("B3.25", not problems_25, f"{LABELS_25[name]} ({name!r}) : conforme au contrat.",
                f"{LABELS_25[name]} ({name!r}) : " + " ; ".join(problems_25) + ".")
        ok_25.append(not problems_25)
    variants_ok_25 = all(ok_25)'''

EXPERIMENT_25 = r'''results_25 = None
if not globals().get("variants_ok_25"):
    print("⏳ Ex B3.25 : l'expérience attend tes cinq variantes (la cellule précédente).")
else:
    with checked("B3.25"):
        train_25 = train_20 if globals().get("probes_ok_20") else wbgpt.train
        if train_25 is wbgpt.train:
            print("ℹ️ ton train_20 (B3.20) n'a pas passé ses contrôles dans cette session : la recette de wb.gpt le remplace.")
        P = PROTOCOL_25
        results_25 = {name: [] for name in VARIANTS_25}
        started_25 = time.perf_counter()
        for seed in P["seeds"]:
            for name in VARIANTS_25:
                torch.manual_seed(seed)
                model = make_variant_25(name, P["config"])
                train_25(model, DATA.train_ids, DATA.val_ids, P["settings"], torch.Generator().manual_seed(seed),
                         block_size=P["config"]["block_size"])
                loss = wbgpt.char_loss(model, DATA.val_ids[:P["val_chars"]], block_size=P["config"]["block_size"])
                results_25[name].append(loss)
        print(f"{len(P['seeds']) * len(VARIANTS_25)} entraînements en {time.perf_counter() - started_25:.0f} s "
              f"({P['settings']['iters']} itérations chacun).\n")
        print(f"{'variante':<16} {'ce notebook : loss':>20} {'écart à la référence':>22}   {'protocole complet (3 graines)':>30}")
        base_25, base_full_25 = np.mean(results_25["reference"]), np.mean(FULL_25["reference"])
        for name in VARIANTS_25:
            mine, full = np.array(results_25[name]), np.array(FULL_25[name])
            gap = mine - np.array(results_25["reference"])
            gap_full = full - np.array(FULL_25["reference"])
            print(f"{LABELS_25[name]:<16} {fr(mine.mean(), 3):>12} ± {fr(mine.std(), 3):<5} {fr(gap.mean(), 3):>14} ± "
                  f"{fr(gap.std(), 3):<5}   {fr(full.mean(), 3):>12} ± {fr(full.std(), 3):<5} écart {fr(gap_full.mean(), 3)} "
                  f"± {fr(gap_full.std(), 3)}")
        fig_25, axes_25 = plt.subplots(1, 2, figsize=(13, 3.8), sharey=False)
        for ax, res, title in ((axes_25[0], results_25, f"ce notebook ({len(P['seeds'])} graines)"),
                               (axes_25[1], FULL_25, "protocole complet (3 graines, précalculé)")):
            gaps = [np.array(res[n]) - np.array(res["reference"]) for n in VARIANTS_25[1:]]
            ax.bar(range(4), [g.mean() for g in gaps], yerr=[g.std() for g in gaps], capsize=4, color="tab:purple")
            ax.set_xticks(range(4))
            ax.set_xticklabels([LABELS_25[n] for n in VARIANTS_25[1:]])
            ax.axhline(0, color="k", lw=0.8)
            ax.set(title=title, ylabel="écart de loss à la référence (nats/car.)")
            ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()'''

TODO_25B = r'''worst_25 = ...     # a) in the full protocol, the ablation that loses the most: "P" no position, "R" no residual,
                   #    "L" post-LN, "H" one head
mildest_25 = ...   # b) in the full protocol, the ablation that loses the least: "P", "R", "L" or "H"
clear_25 = ...     # c) in the full protocol, how many of the four ablations lose more than twice the dispersion of
                   #    their gap (mean gap > 2 standard deviations)? a whole number from 0 to 4'''

CHECK_25B = r'''for letter, answer in zip("abc", [worst_25, mildest_25, clear_25]):
    wb.check(f"B3.25{letter}", answer)'''

# --- 🔬 B3.26 ----------------------------------------------------------------------------------------------------------
GIVEN_26 = r'''# The tokenizer: YOUR BPETokenizer of B2 (byte level, pre-tokenizer 'gpt2', 512 tokens), learned on the training text of
# Holmes; its merges come from the library tokenizers (Rust, the rule of your fit, B2.19), your encode does the rest
VOCAB_26 = 512
try:
    _tok_class_26 = mylearn.tokenizer.BPETokenizer
    _probe_26 = _tok_class_26(vocab_size=258, pre_tokenizer="gpt2").fit("aa aa")
    _probe_26.encode("aa aa")
except Exception:  # noqa: BLE001 - B2 not done (not written, or no tokenizer.py): the reference library
    _tok_class_26 = wb.load_mylearn("ref").tokenizer.BPETokenizer
    print("ℹ️ ton BPETokenizer (B2.16, B2.17) n'est pas écrit : celui de la référence le remplace pour cet exercice.")


def bpe_26(text, vocab_size, tok_class=_tok_class_26):
    """A byte-level BPETokenizer(vocab_size, pre_tokenizer='gpt2') of tok_class with the merges of wbtext.byte_bpe_merges
    (those of fit, 50 to 200 times faster): its four learned attributes set as in B2.20."""
    tok = tok_class(vocab_size=vocab_size, pre_tokenizer="gpt2")
    tok.n_base_, tok.merges_, tok.special_ids_ = 256, wbtext.byte_bpe_merges(text, vocab_size), {}
    tok.vocab_ = {i: bytes([i]) for i in range(256)}
    for r, (a, b) in enumerate(tok.merges_):
        tok.vocab_[256 + r] = tok.vocab_[a] + tok.vocab_[b]
    return tok


from wb import text as wbtext  # noqa: E402

TOK_26 = bpe_26(DATA.train_text, VOCAB_26)
_ref_ids_26 = bpe_26(DATA.train_text, VOCAB_26, wb.load_mylearn("ref").tokenizer.BPETokenizer).encode(DATA.val_text[:3000])
if TOK_26.encode(DATA.val_text[:3000]) != _ref_ids_26:
    print("❌ Ex B3.26 : ton encode (B2.17) ne donne pas les ids de la référence pour les mêmes fusions : relance les tests "
          "de B2 ; la référence le remplace pour cet exercice.")
    TOK_26 = bpe_26(DATA.train_text, VOCAB_26, wb.load_mylearn("ref").tokenizer.BPETokenizer)
# The protocol of this notebook: fast mode, the small model of B3.25 (contexts of 64 characters), 200 iterations,
# 2 seeds; full mode (🚀), the mini-GPT of B3.20 (contexts of 128 characters), 600 iterations, 3 seeds
PROTOCOL_26 = («PROTOCOL_FAST_26» if FAST_MODE else «PROTOCOL_FULL_26»)
FULL_26 = «FULL_26»       # the full protocol, precomputed: nats per character on the validation, one value per seed'''

TODO_26 = r'''def chars_per_token_26(tok, text):
    """The mean number of characters per token of `text` encoded by `tok` (a float)."""
    raise NotImplementedError("chars_per_token_26")


def budget_block_26(chars_per_token, char_block):
    """The context in BPE tokens that covers about char_block characters: an int (equal budget, contract of B3.26)."""
    raise NotImplementedError("budget_block_26")


def nats_per_char_26(model, ids, tok, block_size):
    """The loss of a BPE model in nats per CHARACTER on ids (1-D int64): wbgpt.char_loss(model, ids, block_size) is a
    mean per token over the targets ids[block_size:]; convert it with the characters of those same targets."""
    raise NotImplementedError("nats_per_char_26")'''

SOLUTION_26 = r'''def chars_per_token_26(tok, text):
    """The mean number of characters per token of `text` encoded by `tok` (a float)."""
    return len(text) / len(tok.encode(text))


def budget_block_26(chars_per_token, char_block):
    """The context in BPE tokens that covers about char_block characters: an int (equal budget, contract of B3.26)."""
    return round(char_block / chars_per_token)


def nats_per_char_26(model, ids, tok, block_size):
    """The loss of a BPE model in nats per CHARACTER on ids (1-D int64): wbgpt.char_loss(model, ids, block_size) is a
    mean per token over the targets ids[block_size:]; convert it with the characters of those same targets."""
    per_token = wbgpt.char_loss(model, ids, block_size=block_size, stride=max(1, block_size // 4))
    targets = ids[block_size:]
    n_chars = len(tok.decode(targets.tolist()))           # the characters that these tokens write
    return per_token * len(targets) / n_chars'''

CHECK_26 = r'''conv_ok_26 = False
with checked("B3.26"):
    text_26 = DATA.val_text[:5000]
    cpt_26 = chars_per_token_26(TOK_26, text_26)
    want_cpt_26 = len(text_26) / len(TOK_26.encode(text_26))
    verdict("B3.26", isinstance(cpt_26, float) and math.isclose(cpt_26, want_cpt_26, rel_tol=1e-9),
            f"chars_per_token_26 : {fr(cpt_26, 3)} caractères par token sur 5 000 caractères de la validation.",
            f"chars_per_token_26 doit rendre un float, le nombre de caractères du texte divisé par son nombre de tokens ; "
            f"reçu {cpt_26!r}.")
    blocks_26 = {(2.0, 64): 32, (2.6, 128): 49, (3.9, 128): 33}
    got_blocks_26 = {k: budget_block_26(*k) for k in blocks_26}
    verdict("B3.26", all(type(v) is int for v in got_blocks_26.values()) and got_blocks_26 == blocks_26,
            "budget_block_26 : le contexte en tokens qui couvre autant de caractères (un entier).",
            f"budget_block_26(caractères par token, contexte en caractères) doit rendre l'entier le plus proche du rapport "
            f"des deux : attendu {blocks_26}, reçu {got_blocks_26}.")
    # a model whose loss is known exactly: uniform logits, log V nats per token
    class Uniform_26(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size, self.dummy = 8, nn.Parameter(torch.zeros(1))

        def forward(self, idx):
            return torch.zeros(*idx.shape, VOCAB_26)

    ids_26 = torch.tensor(TOK_26.encode(text_26))
    got_26 = nats_per_char_26(Uniform_26(), ids_26, TOK_26, 8)
    want_26 = math.log(VOCAB_26) * (len(ids_26) - 8) / len(TOK_26.decode(ids_26[8:].tolist()))
    verdict("B3.26", isinstance(got_26, float) and math.isclose(got_26, want_26, rel_tol=1e-6),
            f"nats_per_char_26 : un modèle uniforme, log 512 = {fr(math.log(VOCAB_26), 3)} nats par token, fait "
            f"{fr(want_26, 3)} nats par caractère.",
            f"un modèle uniforme fait log 512 = {fr(math.log(VOCAB_26), 3)} nats par token, donc {fr(want_26, 3)} nats par "
            f"caractère sur ce texte (multiplié par le nombre de tokens notés, divisé par leurs caractères) ; reçu {got_26!r}.")
    conv_ok_26 = math.isclose(got_26, want_26, rel_tol=1e-6) and got_blocks_26 == blocks_26
    cpt_val_26 = chars_per_token_26(TOK_26, DATA.val_text)
    print(f"sur toute la validation de Holmes : {fr(cpt_val_26, 4)} caractères par token BPE (512 tokens)")
    wb.check("B3.26a", cpt_val_26, computed=True)'''

EXPERIMENT_26 = r'''results_26 = None
if not globals().get("conv_ok_26"):
    print("⏳ Ex B3.26 : l'expérience attend tes trois fonctions (la cellule précédente).")
else:
    with checked("B3.26"):
        P = PROTOCOL_26
        cpt_26 = chars_per_token_26(TOK_26, DATA.train_text[:100_000])
        bpe_block_26 = budget_block_26(cpt_26, P["char_block"])
        train_bpe_26 = torch.tensor(TOK_26.encode(DATA.train_text))
        val_bpe_26 = torch.tensor(TOK_26.encode(DATA.val_text[:P["val_chars"]]))
        model_class_26 = MiniGPT_19 if globals().get("model_ok_19") else wbgpt.ReferenceGPT
        train_26 = train_20 if globals().get("probes_ok_20") else wbgpt.train
        print(f"{fr(cpt_26, 2)} caractères par token : un contexte de {P['char_block']} caractères vaut {bpe_block_26} "
              f"tokens BPE ; par itération, {P['settings']['batch_size']} × {P['char_block']} = "
              f"{P['settings']['batch_size'] * P['char_block']} caractères contre {P['settings']['batch_size']} × "
              f"{bpe_block_26} × {fr(cpt_26, 2)} ≈ {P['settings']['batch_size'] * bpe_block_26 * cpt_26:.0f}.")
        results_26 = {"caractères": [], "BPE 512": []}
        seconds_26 = {"caractères": 0.0, "BPE 512": 0.0}
        sizes_26 = {}
        for seed in P["seeds"]:
            for name, vocab, block, train_ids in (("caractères", len(DATA.itos), P["char_block"], DATA.train_ids),
                                                  ("BPE 512", VOCAB_26, bpe_block_26, train_bpe_26)):
                torch.manual_seed(seed)
                model = model_class_26(**(P["config"] | {"vocab_size": vocab, "block_size": block}))
                sizes_26[name] = n_params(model)
                t0 = time.perf_counter()
                train_26(model, train_ids, DATA.val_ids if name == "caractères" else val_bpe_26, P["settings"],
                         torch.Generator().manual_seed(seed), block_size=block)
                seconds_26[name] += time.perf_counter() - t0
                if name == "caractères":
                    loss = wbgpt.char_loss(model, DATA.val_ids[:P["val_chars"]], block_size=block)
                else:
                    loss = nats_per_char_26(model, val_bpe_26, TOK_26, block)
                results_26[name].append(loss)
        print(f"\n{'modèle':<12} {'contexte':>9} {'paramètres':>11} {'temps':>8}   {'nats/car. (ce notebook)':>24}   "
              f"{'protocole complet (3 graines)':>30}")
        for name, block in (("caractères", P["char_block"]), ("BPE 512", bpe_block_26)):
            mine, full = np.array(results_26[name]), np.array(FULL_26[name])
            print(f"{name:<12} {block:>9} {sizes_26[name]:>11_} {seconds_26[name] / len(P['seeds']):>6.0f} s   "
                  f"{fr(mine.mean(), 3):>16} ± {fr(mine.std(), 3):<5}   {fr(full.mean(), 3):>22} ± {fr(full.std(), 3)}"
                  .replace("_", " "))'''

TODO_26B = r'''better_26 = ...    # b) in the full protocol, at equal budget, which is lower in nats per character: "C" characters,
                   #    "B" BPE, or "E" a tie (the gap is within the dispersion of the seeds)'''

CHECK_26B = r'''wb.check("B3.26b", better_26)'''

# --- 🔬 B3.27 ----------------------------------------------------------------------------------------------------------
GIVEN_27 = r'''from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

LENGTHS_27 = (64, 128, 256, 512, 1024, 2048)
BACKENDS_27 = {"MATH": SDPBackend.MATH, "FLASH": SDPBackend.FLASH_ATTENTION}


def profiled_bytes_27(T, backend, n_head=4, d_head=32):
    """The largest memory allocated by one operator during one causal SDPA (1, n_head, T, d_head) with the kernel
    `backend`, read by the profiler of PyTorch (profile_memory=True), in bytes. tracemalloc would not see it: PyTorch
    allocates its tensors outside of Python's allocator."""
    q, k, v = (torch.randn(1, n_head, T, d_head) for _ in range(3))
    with sdpa_kernel(backend):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return max((event.cpu_memory_usage for event in prof.key_averages()), default=0)'''

TODO_27 = r'''def attention_time_27(T, backend, n_head=4, d_head=32, repeats=5):
    """The best time, in milliseconds, of one causal F.scaled_dot_product_attention on random q, k, v of shape
    (1, n_head, T, d_head) in float32, with the kernel `backend` (contract of Ex B3.27)."""
    raise NotImplementedError("attention_time_27")


def weights_bytes_27(T, n_head=4, bytes_per_value=4):
    """The memory, in bytes, of the attention weights (n_head, T, T) of one sequence (an int)."""
    raise NotImplementedError("weights_bytes_27")'''

SOLUTION_27 = r'''def attention_time_27(T, backend, n_head=4, d_head=32, repeats=5):
    """The best time, in milliseconds, of one causal F.scaled_dot_product_attention on random q, k, v of shape
    (1, n_head, T, d_head) in float32, with the kernel `backend` (contract of Ex B3.27)."""
    q, k, v = (torch.randn(1, n_head, T, d_head) for _ in range(3))
    best = math.inf
    with sdpa_kernel(backend):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)          # a first call, not timed (warm-up)
        for _ in range(repeats):
            start = time.perf_counter()
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            best = min(best, time.perf_counter() - start)               # the best: the least disturbed run
    return 1000 * best


def weights_bytes_27(T, n_head=4, bytes_per_value=4):
    """The memory, in bytes, of the attention weights (n_head, T, T) of one sequence (an int)."""
    return n_head * T * T * bytes_per_value'''

CHECK_27 = r'''timing_ok_27 = False
with checked("B3.27"):
    verdict("B3.27", weights_bytes_27(1000) == 16_000_000 and weights_bytes_27(10, 2, 2) == 400
            and type(weights_bytes_27(1000)) is int,
            "weights_bytes_27 : 4 têtes × 1 000² poids × 4 octets = 16 000 000 octets.",
            f"weights_bytes_27(1000) doit valoir 4 × 1 000 × 1 000 × 4 = 16 000 000 (un int) et weights_bytes_27(10, 2, 2) "
            f"400 ; reçu {weights_bytes_27(1000)!r} et {weights_bytes_27(10, 2, 2)!r}.")
    # the best of several runs: a probe whose first call is 50 ms slower
    sdpa_27, calls_27 = F.scaled_dot_product_attention, []

    def slow_first_27(*args, **kwargs):
        calls_27.append(1)
        if len(calls_27) <= 2:
            time.sleep(0.05)
        return sdpa_27(*args, **kwargs)

    F.scaled_dot_product_attention = slow_first_27
    try:
        probe_27 = attention_time_27(64, SDPBackend.MATH, repeats=5)
    finally:
        F.scaled_dot_product_attention = sdpa_27
    if not isinstance(probe_27, (int, float)) or not probe_27 > 0:
        print(f"❌ Ex B3.27 : attention_time_27 doit rendre un temps en millisecondes, un nombre positif ; reçu {probe_27!r}.")
        raise StopCheck
    if not calls_27:
        print("ℹ️ Ex B3.27 : ta fonction n'appelle pas F.scaled_dot_product_attention par son nom complet : le contrôle du "
              "meilleur de plusieurs essais est sauté.")
        timing_ok_27 = True
    else:
        timing_ok_27 = len(calls_27) >= 3 and probe_27 < 20
        verdict("B3.27", timing_ok_27,
                f"attention_time_27 : le meilleur de plusieurs essais, en millisecondes ({fr(probe_27, 2)} ms à T = 64, "
                "malgré deux premiers appels ralentis de 50 ms).",
                f"attention_time_27 a rendu {fr(probe_27, 1)} pour T = 64 en {len(calls_27)} appel(s), alors que les deux "
                "premiers appels étaient ralentis de 50 ms : rends le meilleur de `repeats` essais, après un premier appel "
                "non mesuré, en millisecondes.")'''

EXPERIMENT_27 = r'''if not globals().get("timing_ok_27"):
    print("⏳ Ex B3.27 : la mesure attend tes deux fonctions (la cellule précédente).")
else:
    with checked("B3.27"):
        q_27 = torch.randn(1, 4, 256, 32)
        print("noyau choisi par défaut sur ce CPU (torch._fused_sdp_choice) :",
              {0: "MATH", 1: "FLASH_ATTENTION", 2: "EFFICIENT_ATTENTION"}.get(torch._fused_sdp_choice(q_27, q_27, q_27, is_causal=True), "?"))
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                F.scaled_dot_product_attention(q_27, q_27, q_27, is_causal=True)
            print("EFFICIENT_ATTENTION : disponible")
        except RuntimeError as exc:
            print("EFFICIENT_ATTENTION :", str(exc).splitlines()[0][:90])
        times_27 = {name: [attention_time_27(T, b) for T in LENGTHS_27] for name, b in BACKENDS_27.items()}
        bytes_27 = {name: [profiled_bytes_27(T, b) for T in LENGTHS_27] for name, b in BACKENDS_27.items()}
        print(f"\n{'T':>6} {'MATH (ms)':>10} {'FLASH (ms)':>11} {'rapport':>8}   {'poids, formule':>15} {'MATH, mesuré':>13} "
              f"{'FLASH, mesuré':>14}")
        for i, T in enumerate(LENGTHS_27):
            print(f"{T:>6} {fr(times_27['MATH'][i], 2):>10} {fr(times_27['FLASH'][i], 2):>11} "
                  f"{fr(times_27['MATH'][i] / times_27['FLASH'][i], 1):>8}   {fr(weights_bytes_27(T) / 2**20, 2) + ' Mio':>15} "
                  f"{fr(bytes_27['MATH'][i] / 2**20, 2) + ' Mio':>13} {fr(bytes_27['FLASH'][i] / 2**20, 2) + ' Mio':>14}")
        slopes_27 = {name: np.polyfit(np.log(LENGTHS_27[3:]), np.log(t[3:]), 1)[0] for name, t in times_27.items()}
        print("\npente du temps en log-log, de 512 à 2 048 tokens :", {n: fr(s, 2) for n, s in slopes_27.items()})
        verdict("B3.27", 1.5 <= slopes_27["MATH"] <= 2.5,
                f"noyau MATH : le temps croît comme T^{fr(slopes_27['MATH'], 1)} au-delà de 512 tokens (quadratique).",
                f"noyau MATH : pente {fr(slopes_27['MATH'], 2)} au-delà de 512 tokens, loin de 2 : la machine est-elle "
                "chargée ? Relance la cellule.")
        verdict("B3.27", abs(bytes_27["MATH"][-1] / weights_bytes_27(LENGTHS_27[-1]) - 1) < 0.1
                and bytes_27["FLASH"][-1] < 0.1 * weights_bytes_27(LENGTHS_27[-1]),
                "mémoire à 2 048 tokens : le noyau MATH alloue la matrice des poids (ta formule), le noyau FLASH beaucoup "
                "moins.", "mémoire à 2 048 tokens : la plus grosse allocation mesurée de MATH devrait être voisine de ta "
                "formule, et celle de FLASH bien plus petite.")
        fig_27, axes_27 = plt.subplots(1, 2, figsize=(12, 4))
        for name in BACKENDS_27:
            axes_27[0].loglog(LENGTHS_27, times_27[name], "o-", label=name)
            axes_27[1].loglog(LENGTHS_27, [b / 2**20 for b in bytes_27[name]], "o-", label=f"{name}, mesuré")
        axes_27[1].loglog(LENGTHS_27, [weights_bytes_27(T) / 2**20 for T in LENGTHS_27], "k--", label="poids, formule")
        axes_27[0].set(xlabel="T (tokens)", ylabel="temps (ms)", title="Temps d'une attention causale (4 têtes, d_h = 32)")
        axes_27[1].set(xlabel="T (tokens)", ylabel="Mio", title="Plus grosse allocation d'un opérateur")
        for ax in axes_27:
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        plt.tight_layout()
        plt.show()'''

# --- 📦 B3.28 ----------------------------------------------------------------------------------------------------------
GIVEN_28 = r'''from wb import datasets as wbdata

N_TRAIN_28, EPOCHS_28 = (12_000, 3) if FAST_MODE else (60_000, «EPOCHS_FULL_28»)
_xtr_28, _ytr_28 = wbdata.load_fashion_mnist("train", n=N_TRAIN_28 if FAST_MODE else None, seed=0)
_xte_28, _yte_28 = wbdata.load_fashion_mnist("test")
MEAN_28, STD_28 = 0.2860, 0.3530                          # the statistics of the training images (ch. 20 and 21)
X_TRAIN_28 = (torch.from_numpy(_xtr_28).float().div(255).sub(MEAN_28).div(STD_28))[:, None]   # (N, 1, 28, 28)
Y_TRAIN_28 = torch.from_numpy(_ytr_28).long()
X_TEST_28 = (torch.from_numpy(_xte_28).float().div(255).sub(MEAN_28).div(STD_28))[:, None]
Y_TEST_28 = torch.from_numpy(_yte_28).long()
TARGET_28 = «TARGET_28» if FAST_MODE else «TARGET_FULL_28»   # accuracy on the 10 000 test images


def train_vit_28(model, X, y, epochs, batch_size=128, lr=2e-3, weight_decay=0.05, seed=28):
    """AdamW, linear warmup then cosine decay (per mini-batch, wbgpt.lr_at), clipping at 1: returns the mean training
    loss of each epoch."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps, step, losses = epochs * math.ceil(len(X) / batch_size), 0, []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            for group in optimizer.param_groups:
                group["lr"] = wbgpt.lr_at(step, steps, lr, steps // 10, lr / 20)
            loss = F.cross_entropy(model(X[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            step += 1
        losses.append(total / len(X))
    return losses


@torch.no_grad()
def accuracy_28(model, X, y, batch_size=1000):
    """The accuracy of `model` (eval mode) on (X, y)."""
    model.eval()
    return sum((model(X[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
               for i in range(0, len(X), batch_size)) / len(X)


print(f"Fashion-MNIST : {len(X_TRAIN_28):_} images d'entraînement, {len(X_TEST_28):_} de test, {EPOCHS_28} epochs."
      .replace("_", " "))'''

TODO_28 = r'''class EncoderBlock_28(nn.Module):
    """A pre-LN encoder block (contract of Ex B3.28): every token sees every token, x (N, T, d) -> (N, T, d)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        raise NotImplementedError("EncoderBlock_28")

    def forward(self, x):
        raise NotImplementedError("EncoderBlock_28.forward")


class ViT_28(nn.Module):
    """A tiny Vision Transformer (contract of Ex B3.28): images (N, 1, 28, 28) -> logits (N, n_classes)."""

    def __init__(self, image_size=28, patch_size=7, n_classes=10, d_model=64, n_layer=4, n_head=4, dropout=0.0):
        super().__init__()
        raise NotImplementedError("ViT_28")

    def forward(self, x):
        raise NotImplementedError("ViT_28.forward")'''

SOLUTION_28 = r'''class EncoderBlock_28(nn.Module):
    """A pre-LN encoder block (contract of Ex B3.28): every token sees every token, x (N, T, d) -> (N, T, d)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"n_head = {n_head} must divide d_model = {d_model}")
        self.n_head, self.dropout = n_head, dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(N, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)   # no mask
        x = x + self.proj(y.transpose(1, 2).reshape(N, T, D))
        return x + self.mlp(self.ln2(x))


class ViT_28(nn.Module):
    """A tiny Vision Transformer (contract of Ex B3.28): images (N, 1, 28, 28) -> logits (N, n_classes)."""

    def __init__(self, image_size=28, patch_size=7, n_classes=10, d_model=64, n_layer=4, n_head=4, dropout=0.0):
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"patch_size = {patch_size} must divide image_size = {image_size}")
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)   # a dense layer per patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([EncoderBlock_28(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)                  # (N, 16, d): one token per patch
        tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1) + self.pos_emb
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.ln_f(tokens[:, 0]))                                # the [CLS] token classifies'''

CHECK_28 = r'''vit_ok_28 = False
with checked("B3.28"):
    need("B3.28", "ViT_28")
    torch.manual_seed(28)
    vit_28 = ViT_28()
    if forbidden_in(vit_28):
        print(f"❌ Ex B3.28 : ton ViT contient {forbidden_in(vit_28)} : le contrat demande ton propre bloc d'encodeur.")
        raise StopCheck
    shapes_28 = {n: tuple(p.shape) for n, p in vit_28.named_parameters()}
    want_28 = {"patch_embed.weight": (64, 1, 7, 7), "patch_embed.bias": (64,), "cls_token": (1, 1, 64),
               "pos_emb": (1, 17, 64), "head.weight": (10, 64), "head.bias": (10,)}
    wrong_28 = {n: shapes_28.get(n) for n, s in want_28.items() if shapes_28.get(n) != s}
    verdict("B3.28", not wrong_28 and len(vit_28.blocks) == 4,
            "les patchs : une convolution 7 × 7 de stride 7 vers d = 64, un token [CLS], 17 positions apprises (16 patchs "
            "+ [CLS]), 4 blocs et une tête de 10 classes.",
            f"paramètres attendus {want_28} et 4 blocs ; différents : {wrong_28}, {len(getattr(vit_28, 'blocks', []))} bloc(s).")
    if wrong_28:
        raise StopCheck
    print(f"{n_params(vit_28):_} paramètres.".replace("_", " "))
    vit_28.eval()
    x_28 = X_TEST_28[:4].clone()
    with torch.no_grad():
        out_28 = vit_28(x_28)
        shape_ok_28 = tuple(out_28.shape) == (4, 10)
        verdict("B3.28", shape_ok_28, "des images (4, 1, 28, 28) donnent des logits (4, 10).",
                f"des images (4, 1, 28, 28) doivent donner des logits (4, 10) ; reçu {tuple(out_28.shape)}.")
        if not shape_ok_28:
            raise StopCheck
        last_28 = x_28.clone()
        last_28[:, :, 21:, 21:] += 1.0                                         # the last patch (bottom right) only
        first_28 = x_28.clone()
        first_28[:, :, :7, :7] += 1.0                                          # the first patch (top left) only
        sees_last_28 = (vit_28(last_28) - out_28).abs().max().item() > 1e-6
        sees_first_28 = (vit_28(first_28) - out_28).abs().max().item() > 1e-6
    verdict("B3.28", sees_last_28 and sees_first_28,
            "la sortie dépend du premier et du dernier patch : le [CLS] voit toute l'image (aucun masque causal).",
            "la sortie ne dépend pas " + ("du dernier patch" if not sees_last_28 else "du premier patch")
            + " : un masque causal resté dans l'encodeur ? Le [CLS] doit voir tous les patchs.")
    with torch.no_grad():
        swapped_28 = x_28.clone()
        swapped_28[:, :, :7, :7], swapped_28[:, :, 21:, 21:] = x_28[:, :, 21:, 21:], x_28[:, :, :7, :7]
        moved_28 = (vit_28(swapped_28) - out_28).abs().max().item()
    verdict("B3.28", moved_28 > 1e-6, "échanger deux patchs change la sortie : les positions apprises sont ajoutées.",
            "échanger le premier et le dernier patch ne change pas la sortie : les positions apprises sont-elles ajoutées "
            "aux tokens ?")
    refuses("B3.28", "patch_size = 5 pour des images de 28 pixels", lambda: ViT_28(patch_size=5))
    vit_ok_28 = sees_last_28 and sees_first_28 and moved_28 > 1e-6'''

EXPERIMENT_28 = r'''if not globals().get("vit_ok_28"):
    print("⏳ Ex B3.28 : l'entraînement attend ton ViT (la cellule précédente).")
else:
    with checked("B3.28"):
        torch.manual_seed(28)
        model_28 = ViT_28()
        t0_28 = time.perf_counter()
        losses_28 = train_vit_28(model_28, X_TRAIN_28, Y_TRAIN_28, EPOCHS_28)
        seconds_28 = time.perf_counter() - t0_28
        acc_28 = accuracy_28(model_28, X_TEST_28, Y_TEST_28)
        print(f"{EPOCHS_28} epochs en {seconds_28:.0f} s ; loss d'entraînement par epoch {[round(l, 3) for l in losses_28]} ; "
              f"accuracy de test {fr(100 * acc_28, 1)} %.")
        verdict("B3.28", acc_28 >= TARGET_28, f"accuracy de test {fr(100 * acc_28, 1)} %, au-dessus de {fr(100 * TARGET_28, 0)} %.",
                f"accuracy de test {fr(100 * acc_28, 1)} % : attendu au moins {fr(100 * TARGET_28, 0)} % ; relis les "
                "contrôles de la cellule précédente.")'''

# --- 🛠️ B3.29 ----------------------------------------------------------------------------------------------------------
GIVEN_29 = r'''# The models under test: your trained mini-GPT (its checkpoint, in the class ReferenceGPT of wb.gpt), or an untrained
# mini-GPT when there is none (causality does not depend on training); and three copies whose attention sees the future
SUBJECT_29 = str(CKPT["path"]) if CKPT is not None else None
_PREAMBLE_29 = """
import torch
import torch.nn.functional as F
from wb import gpt as wbgpt

torch.set_grad_enabled(False)


PATH = {path!r}


def _subject():
    if PATH is None:
        torch.manual_seed(29)
        return wbgpt.ReferenceGPT(**wbgpt.MINI_CONFIG).eval()
    return wbgpt.build_from_checkpoint(wbgpt.ReferenceGPT, torch.load(PATH, map_location="cpu", weights_only=True))


def _with_mask(model, rule):
    # every attention of the model, with the boolean mask rule(T) (True = may attend) instead of the causal one
    def forward(attn, x):
        N, T, D = x.shape
        q, k, v = attn.qkv(x).split(D, dim=2)
        q, k, v = (t.view(N, T, attn.n_head, D // attn.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=rule(T))
        return attn.proj(y.transpose(1, 2).reshape(N, T, D))
    for block in model.blocks:
        block.attn.forward = forward.__get__(block.attn)
    return model


_RULES = {{
    "transposed": lambda T: torch.ones(T, T, dtype=torch.bool).triu(),
    "one_ahead": lambda T: torch.ones(T, T, dtype=torch.bool).tril(diagonal=1),
    "late_leak": lambda T: torch.ones(T, T, dtype=torch.bool).tril() | (torch.eye(T, dtype=torch.bool).roll(1, dims=1)
                                                                       & (torch.arange(T)[:, None] >= 100)),
}}
MODEL = _subject() if {kind!r} == "good" else _with_mask(_subject(), _RULES[{kind!r}])
VOCAB_SIZE, BLOCK_SIZE = MODEL.tok_emb.num_embeddings, MODEL.block_size
"""


def preamble_29(kind):
    """The code that the test file runs before your test: MODEL (eval mode, no gradient), VOCAB_SIZE, BLOCK_SIZE."""
    return _PREAMBLE_29.format(path=SUBJECT_29, kind=kind)


def failures_29(result):
    """(name, reason) of each failed test of a wb.run_pytest result, read in pytest's short summary."""
    out = []
    for line in result.output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            name, _, reason = line.split(" ", 1)[1].partition(" - ")
            out.append((name.split("::")[-1], reason.strip() or "(sans message)"))
    return out


print("modèle testé :", "ton mini-GPT entraîné (B3.20)" if SUBJECT_29 else "un mini-GPT non entraîné (pas de sauvegarde de B3.20)")'''

TODO_29 = r'''def test_causality_29():
    """Changing the token at a position t never changes the logits of the positions before t (MODEL, VOCAB_SIZE,
    BLOCK_SIZE and torch are given by the test file)."""
    raise NotImplementedError("test_causality_29")'''

SOLUTION_29 = r'''def test_causality_29():
    """Changing the token at a position t never changes the logits of the positions before t (MODEL, VOCAB_SIZE,
    BLOCK_SIZE and torch are given by the test file)."""
    generator = torch.Generator().manual_seed(0)
    idx = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE), generator=generator)
    logits = MODEL(idx)
    for t in range(1, BLOCK_SIZE):                         # every position, the first one excepted (nothing before it)
        changed = idx.clone()
        changed[:, t] = (changed[:, t] + 1) % VOCAB_SIZE
        past = (MODEL(changed)[:, :t] - logits[:, :t]).abs().max().item()
        assert past == 0, f"expected the logits before position {t} unchanged, got a change of {past:.2e}"'''

CHECK_29 = r'''import inspect

if "NotImplementedError" in inspect.getsource(test_causality_29):
    print("⏳ Ex B3.29 : écris ton test de causalité.")
else:
    with checked("B3.29"):
        print("Ton test, sur le bon modèle (il doit passer) :")
        good_29 = wb.run_pytest([test_causality_29], preamble=preamble_29("good"))
        verdict("B3.29", good_29.ok, "ton test passe sur le bon modèle.",
                "ton test doit passer sur le bon modèle (la sortie de pytest est au-dessus).")
        if not good_29.ok:
            raise StopCheck
        for kind, label in (("transposed", "le masque causal transposé (triu au lieu de tril)"),
                            ("one_ahead", "un masque qui laisse voir le token suivant"),
                            ("late_leak", "un masque qui laisse voir le token suivant à partir de la position 100 seulement")):
            bad_29 = wb.run_pytest([test_causality_29], preamble=preamble_29(kind), quiet=True)
            caught_29 = not bad_29.ok and bad_29.failed > 0
            verdict("B3.29", caught_29, f"ton test échoue sur un modèle à {label} :",
                    f"ton test laisse passer un modèle à {label}" + ("." if bad_29.errors == 0 else
                    f" (pytest : {bad_29.summary})."))
            for name, reason in failures_29(bad_29)[:1]:
                print(f"   {name}\n   {reason[:300]}")'''

# --- 🏆 B3.30 ----------------------------------------------------------------------------------------------------------
GIVEN_30 = r'''# The challenge: the best loss per character on the validation of Holmes, at most 1 000 000 learned values, 10 min of CPU
MAX_VALUES_30, MAX_BLOCK_30 = 1_000_000, 512
LIMIT_30 = 120.0 if FAST_MODE else 600.0          # seconds: a reduced run in fast mode, THE challenge in full mode
GOAL_30 = «GOAL_FAST_30» if FAST_MODE else «GOAL_FULL_30»   # nats per character to reach (reference + 5 %, worst of the seeds)
REFERENCE_30 = «REF_FAST_30» if FAST_MODE else «REF_FULL_30»   # the reference solution: (best, worst) of the seeds
CONTROL_GAP_30 = «CONTROL_GAP_FAST_30» if FAST_MODE else «CONTROL_GAP_FULL_30»   # control: the gap (validation − held-out
                                                                                # text) of the reference, and its tolerance


def _make_evaluate_30(data):
    """evaluate_30, with the validation and the held-out text kept here only (the global variables that hold them are
    replaced by a decoy while your method runs)."""
    train_text, val_text = data.train_text, data.val_text
    cut = wbtext.split_point(train_text, 8 / 9)
    vocab_size = len(data.itos)
    texts = {"validation": (data.val_ids, data.train_ids), "contrôle": (data.train_ids[cut:], data.train_ids[:cut])}
    decoy_text = " ".join(sorted(set(train_text[:cut].split()))[:3000])

    def marker(text):
        at = next(i for i in range(1000, len(text)) if not set(text[i:i + 60]) & set("\n'\"\\"))
        return text[at:at + 60]

    markers = {"validation": [marker(val_text)], "contrôle": [marker(val_text), marker(train_text[cut:])]}

    def hidden_globals(name):
        g, found = globals(), {}
        for n, v in list(g.items()):
            if n.startswith("_") or n in ("In", "Out"):
                continue
            if isinstance(v, str) and any(m in v for m in markers[name]):
                found[n] = decoy_text
            elif isinstance(v, wbgpt.CharData):
                found[n] = None
            elif isinstance(v, dict) and "state_dict" in v:                    # a checkpoint (CKPT)
                found[n] = None
            elif isinstance(v, torch.Tensor) and v.ndim >= 1 and v.numel() >= 10_000 and not v.dtype.is_floating_point:
                found[n] = v[:0]                                                # an array of ids
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.size >= 10_000 and np.issubdtype(v.dtype, np.integer):
                found[n] = v[:0]
        return found

    def count_values(model):
        seen, total = set(), 0
        for t in list(model.parameters()) + list(model.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                total += t.numel()
        return total

    def run(method, name, seconds):
        """Train the method on the ids that go with `name`, cold (an empty checkpoints folder, no reference model),
        with the global variables that hold a measured text replaced by a decoy and the loaders of wb giving only the
        text it learns on. Returns (model, seconds) or None."""
        measured, learned = texts[name]
        learned_text = data.decode(learned)
        g = globals()
        hidden = hidden_globals(name)
        before = [m for m in g.values() if isinstance(m, nn.Module)]
        pointers = {p.data_ptr() for m in before for p in m.parameters()}
        saved = {"load_text": wbtext.load_text, "load_holmes": wbtext.datasets.load_holmes, "char_data": wbgpt.char_data,
                 "checkpoints_dir": wbgpt.checkpoints_dir, "reference_model": wbgpt.reference_model,
                 "texts": dict(wbtext._TEXTS), "data": dict(wbgpt._DATA)}
        empty = Path(tempfile.mkdtemp(prefix="wb_b3_30_"))

        def refused(*args, **kwargs):
            raise RuntimeError("pendant le défi, ta méthode ne reçoit que train_ids, vocab_size et seconds")

        originals = {n: g[n] for n in hidden}
        try:
            g.update(hidden)
            wbtext.load_text = lambda text_name, _load=saved["load_text"]: learned_text if text_name == "holmes" else _load(text_name)
            wbtext.datasets.load_holmes = lambda *args, **kwargs: learned_text
            wbtext._TEXTS.pop("holmes", None)
            wbgpt._DATA.clear()
            wbgpt.char_data, wbgpt.reference_model = refused, refused
            wbgpt.checkpoints_dir = lambda: empty
            torch.manual_seed(3030)
            start = time.perf_counter()
            model = method(learned.clone(), vocab_size, seconds)
            elapsed = time.perf_counter() - start
        finally:
            g.update(originals)
            wbtext.load_text, wbtext.datasets.load_holmes = saved["load_text"], saved["load_holmes"]
            wbgpt.char_data, wbgpt.reference_model = saved["char_data"], saved["reference_model"]
            wbgpt.checkpoints_dir = saved["checkpoints_dir"]
            wbtext._TEXTS.clear()
            wbtext._TEXTS.update(saved["texts"])
            wbgpt._DATA.clear()
            wbgpt._DATA.update(saved["data"])
        if not isinstance(model, nn.Module):
            print(f"❌ Ex B3.30 : method_30 doit rendre un nn.Module ; reçu {type(model).__name__}.")
            return None
        old = [p for m in before for p in m.parameters()]
        copied = any(p.data_ptr() in pointers or any(q.shape == p.shape and torch.equal(q.detach(), p.detach()) for q in old)
                     for p in model.parameters() if p.numel() > 1000)
        if any(model is m for m in before) or copied:
            print("❌ Ex B3.30 : method_30 doit construire et entraîner un modèle neuf pendant la course : le modèle rendu "
                  "existait déjà, ou il a les poids d'un modèle du notebook.")
            return None
        block = getattr(model, "block_size", None)
        if not isinstance(block, int) or not 1 <= block <= MAX_BLOCK_30:
            print(f"❌ Ex B3.30 : le modèle doit avoir un attribut block_size, un entier de 1 à {MAX_BLOCK_30} ; reçu {block!r}.")
            return None
        return model, elapsed

    def score(model, ids):
        """Nats per character on ids, every target from index MAX_BLOCK_30 on (the same targets for every context)."""
        block = model.block_size
        return wbgpt.char_loss(model, ids[MAX_BLOCK_30 - block:], block_size=block, stride=max(1, min(32, block // 2)))

    def evaluate_30(method):
        """method(train_ids, vocab_size, seconds) -> model: the rules of the statement, then the loss per character."""
        out = run(method, "validation", LIMIT_30)
        if out is None:
            return None
        model, seconds = out
        values = count_values(model)
        loss = score(model, data.val_ids)
        print(f"{values:_} valeurs apprises, contexte de {model.block_size} caractères ; entraînement en {fr(seconds, 1)} s ; "
              f"validation : {fr(loss, 4)} nats par caractère (perplexité {fr(math.exp(loss), 2)}).".replace("_", " "))
        verdict("B3.30", values <= MAX_VALUES_30, f"{values:_} valeurs apprises, au plus {MAX_VALUES_30:_}.".replace("_", " "),
                f"{values:_} valeurs apprises (paramètres et buffers) : le défi en permet {MAX_VALUES_30:_}."
                .replace("_", " "))
        verdict("B3.30", seconds <= LIMIT_30, f"entraînement en {fr(seconds, 1)} s, sous les {LIMIT_30:.0f} s.",
                f"entraînement en {fr(seconds, 1)} s : le défi en permet {LIMIT_30:.0f} (ta méthode doit respecter seconds).")
        # the control: the same method, on the first 8/9 of the training text, with a quarter of the time
        honest, control = False, run(method, "contrôle", LIMIT_30 / 4)
        c_val = c_held = c_seconds = None
        if control is not None:
            c_model, c_seconds = control
            c_val, c_held = score(c_model, data.val_ids), score(c_model, texts["contrôle"][0])
            gap, (ref_gap, tol) = c_val - c_held, CONTROL_GAP_30
            honest = abs(gap - ref_gap) <= tol and c_seconds <= LIMIT_30 / 4
            verdict("B3.30", c_seconds <= LIMIT_30 / 4,
                    f"contrôle : entraînement en {fr(c_seconds, 1)} s, sous les {LIMIT_30 / 4:.0f} s demandées.",
                    f"contrôle : entraînement en {fr(c_seconds, 1)} s pour {LIMIT_30 / 4:.0f} s demandées : ta méthode doit "
                    "respecter son argument seconds.")
            verdict("B3.30", abs(gap - ref_gap) <= tol,
                    f"contrôle : appris sur les 8/9 de l'entraînement, ton modèle fait {fr(c_val, 3)} sur la validation et "
                    f"{fr(c_held, 3)} sur le dernier neuvième, tenu à l'écart : l'écart de la référence ({fr(ref_gap, 2)} "
                    f"± {fr(tol, 2)}).",
                    f"contrôle : appris sur les 8/9 de l'entraînement, ton modèle fait {fr(c_val, 3)} sur la validation mais "
                    f"{fr(c_held, 3)} sur le dernier neuvième, tenu à l'écart (écart de la référence {fr(ref_gap, 2)} ± "
                    f"{fr(tol, 2)}) : apprend-il autre chose que les ids qu'il reçoit ?")
        ok = values <= MAX_VALUES_30 and seconds <= LIMIT_30 and honest
        verdict("B3.30", ok and loss <= GOAL_30,
                f"défi réussi : {fr(loss, 3)} nats par caractère (objectif {fr(GOAL_30, 2)} ; référence de "
                f"{fr(REFERENCE_30[0], 3)} à {fr(REFERENCE_30[1], 3)}).",
                f"pas encore : {fr(loss, 3)} nats par caractère, il en faut au plus {fr(GOAL_30, 2)} en respectant les "
                "règles.")
        return {"loss": loss, "seconds": seconds, "values": values, "honest": honest, "ok": ok and loss <= GOAL_30,
                "control_val": c_val, "control_held": c_held, "control_seconds": c_seconds}

    return evaluate_30


import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from wb import text as wbtext  # noqa: E402

evaluate_30 = _make_evaluate_30(DATA)
print(f"Course de {LIMIT_30:.0f} s ; objectif : au plus {fr(GOAL_30, 2)} nats par caractère sur la validation.")'''

TODO_30 = r'''def method_30(train_ids, vocab_size, seconds):
    """Your method: build and train, in at most `seconds` seconds, a NEW model of at most 1 000 000 learned values on
    train_ids ONLY (1-D int64, the characters of Holmes); return it: model(idx (N, T)) -> logits (N, T, vocab_size)
    for T <= model.block_size."""
    raise NotImplementedError("method_30")'''

SOLUTION_30 = r'''SEED_30 = 3030          # the seed of the reference (history_chB3.py calibrates the goal on several seeds)


def method_30(train_ids, vocab_size, seconds):
    """Your method: build and train, in at most `seconds` seconds, a NEW model of at most 1 000 000 learned values on
    train_ids ONLY (1-D int64, the characters of Holmes); return it: model(idx (N, T)) -> logits (N, T, vocab_size)
    for T <= model.block_size."""
    start = time.perf_counter()
    torch.manual_seed(SEED_30)
    generator = torch.Generator().manual_seed(SEED_30)
    model = MiniGPT_19(vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128)    # 820 992 parameters
    optimizer = configure_optimizer_20(model, 5e-3, 0.1, (0.9, 0.99))
    model.train()
    warmup, total, it = 50, None, 0
    while total is None or it < total:
        elapsed = time.perf_counter() - start
        if it == 20:                                        # the speed of this machine, measured: the number of iterations
            total = max(21, int(0.92 * seconds / (elapsed / 20)))
        if elapsed > 0.95 * seconds:                        # a slower machine than measured: stop in time
            break
        lr = wbgpt.lr_at(it, total or 10**9, 5e-3, warmup, 5e-4)   # warmup, then a cosine down to the last iteration
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = wbgpt.get_batch(train_ids, 32, 128, generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        it += 1
    return model.eval()'''

CHECK_30 = r'''with checked("B3.30"):
    need("B3.30", "method_30")
    result_30 = evaluate_30(method_30)'''
