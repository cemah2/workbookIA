#!/usr/bin/env python
"""Calibrate the predictions and precompute the generated replies of chapter B4, first half (used by Claude).

    python tools/chapters/history_chB4.py                    # everything (about 35 min on a 2-core CPU)
    python tools/chapters/history_chB4.py --only p13 p19      # some entries (names below)

Writes tools/chapters/history_chB4.json (statistics only), which chB4_parts.py reads:
- "c12": 📦 B4.12, the three counts of SmolLM2-135M-Instruct and the values of their classic mistakes;
- "p13": 🔮 B4.13, the prompt of the notebook drawn with 10 series of 10 seeds at 0.7 and 1.5 (seeds 0 to 99), and
  three other prompts with 5 series each: both answers (at least 3 of the 10 replies at 0.7 identical to the greedy
  one; at least 7 of them naming Watson) must be the same in every series of every prompt;
- "p19": 🔮 B4.19, the ranks of the relevant chunk of the 10 paraphrased questions for BM25, the dense search
  (MiniLM) and the hybrid search, then the same counts after 20 small perturbations of the embeddings (Gaussian
  noise of standard deviation 1e-3, then norm 1): the three answers must hold for every perturbation;
- "e5_holmes": 📦 B4.20, the solution's evaluate_20 with multilingual-e5-small on Holmes (about 1 min of embeddings);
- "p16_full": 🔬 B4.16, the full protocol (Qwen2.5-0.5B-Instruct, the 8 passages, the same prompts and seeds);
- "r21": 📦 B4.21, the solution's RAG on the 30 questions with Qwen2.5 (for the second half of the chapter).
"generations" (always run with the others, or alone) executes the cells of the solutions notebook (chB4_code.py) in one
namespace, fast mode then full mode, with a cache of source "solution" in a temporary folder, and writes the replies
into tools/chapters/history_chB4_generations.json (wb.llm.SOLUTION_CACHE): the frozen cache that the solutions
notebook reads. The replies depend on the machine only through rounding; the key of each reply holds the versions of
transformers and torch.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB4.json"


def namespace(fast: bool = True, folder: str | None = None, blocks=()) -> dict:
    """The namespace of the solutions notebook: wb, the reference library, the tools of part A with a cache of source
    "solution" in `folder` (a temporary one by default), then the cells `blocks` of chB4_code.py, executed in order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB4_code as C

    plt.show = lambda *a, **k: plt.close("all")
    folder = folder or tempfile.mkdtemp(prefix="wb_b4_")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=fast), "FAST_MODE": fast, "__name__": "history_chB4",
          "mylearn": wb.load_mylearn("ref")}
    tools = C.TOOLS_A.replace(C.LEARNER_CACHE, f'CACHE = wbllm.GenerationCache("solution", folder={folder!r})')
    run(tools, ns)
    for block in blocks:
        run(block if block.startswith(("#", "\n")) or "\n" in block else getattr(C, block), ns)
    return ns


def run(code: str, ns: dict, quiet: bool = True) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out if quiet else sys.stdout):
        exec(code, ns)
    return out.getvalue()


def cells(*names) -> list:
    import chB4_code as C
    return [getattr(C, n) for n in names]


# ---------------------------------------------------------------------------------------------------------------------
def c12(_) -> dict:
    import torch.nn as nn

    import chB4_code as C
    from wb import llm as wbllm

    ns = {}
    exec(C.GIVEN_12, ns)
    llm = wbllm.load_llm("smollm2")
    tok, model = llm.tokenizer, llm.model
    layer = model.model.layers[0]
    attention = sum(p.numel() for p in layer.self_attn.parameters())
    mlp = sum(p.numel() for p in layer.mlp.parameters())
    out = {"n_prompt": len(tok(ns["PROMPT_12"])["input_ids"]),
           "n_linear": sum(isinstance(m, nn.Linear) for m in model.modules()),
           "n_block": sum(p.numel() for p in layer.parameters()),
           "words": len(ns["PROMPT_12"].split()), "chars": len(ns["PROMPT_12"]),
           "chat_prompt": len(llm.prompt_ids(ns["PROMPT_12"])), "attention": attention, "mlp": mlp,
           "norms": sum(p.numel() for p in layer.input_layernorm.parameters())
           + sum(p.numel() for p in layer.post_attention_layernorm.parameters())}
    values = [out[k] for k in ("n_prompt", "words", "chars", "chat_prompt")]
    if len(set(values)) != len(values):
        raise SystemExit(f"❌ c12: a mistake gives the answer of a): {values}")
    print(f"  c12 {out}", flush=True)
    return out


P13_OTHERS = ("Who writes down the adventures of Sherlock Holmes? Answer with a name.",
              "Who is Sherlock Holmes's friend and assistant? Answer with a name.",
              "Sherlock Holmes has a loyal companion. What is his name? Answer with a name.")


def p13(_) -> dict:
    import chB4_code as C
    from wb import llm as wbllm

    ns = {}
    exec(C.GIVEN_13.split("\n\n\ndef")[0], ns)
    llm = wbllm.load_llm("smollm2")
    cache = wbllm.GenerationCache("calibration")
    rows, answers = [], set()
    for prompt, n_series in ((ns["PROMPT_13"], 10), *((p, 5) for p in P13_OTHERS)):
        greedy = llm.generate(prompt, max_new_tokens=24, cache=cache)
        for series in range(n_series):
            seeds = range(10 * series, 10 * series + 10)
            row = {"prompt": prompt, "seeds": [seeds.start, seeds.stop - 1]}
            for t in (0.7, 1.5):
                replies = [llm.generate(prompt, max_new_tokens=24, do_sample=True, temperature=t, top_k=0, top_p=1.0,
                                        seed=s, cache=cache) for s in seeds]
                row[str(t)] = {"identical": sum(r == greedy for r in replies), "distinct": len(set(replies)),
                               "watson": sum("watson" in r.lower() for r in replies)}
            rows.append(row)
            answers.add((row["0.7"]["identical"] >= 3, row["0.7"]["watson"] >= 7))
            print(f"  p13 {prompt[:40]!r} seeds {row['seeds']}: {row['0.7']} | {row['1.5']}", flush=True)
    if len(answers) != 1:
        raise SystemExit(f"❌ p13: the answers depend on the seeds or the prompt: {answers}")
    identical, half = answers.pop()
    return {"rows": rows, "answers": {"identical": bool(identical), "watson": bool(half)},
            "max_watson_07": max(r["0.7"]["watson"] for r in rows),
            "max_identical_07": max(r["0.7"]["identical"] for r in rows)}


def p19(_) -> dict:
    import mylearn_ref.retrieval as R
    from wb import rag as wbrag
    from wb import text as wbtext

    chunks, _ = wbrag.reference_chunks("holmes")
    sm = wbtext.load_sentence_model("minilm")
    emb = wbtext.passage_embeddings(sm, chunks, "b4-holmes-120-20", verbose=False)
    bm = R.BM25().fit([wbrag.tokenize(c) for c in chunks])
    para = [q for q in wbrag.load_qa("holmes") if q["paraphrased"]]
    queries = sm.embed([q["question"] for q in para], kind="query")

    def counts(E, Q):
        found = {"bm25": 0, "dense": 0, "hybrid": 0}
        ranks = []
        for item, q in zip(para, Q):
            rel = wbrag.relevant_ids(chunks, item)
            b = np.argsort(-bm.get_scores(wbrag.tokenize(item["question"])), kind="stable")[:20]
            d, _ = R.cosine_top_k(q, E, k=20)
            h = R.reciprocal_rank_fusion([b, d], k=60)[:20]
            row = {}
            for name, ranking in (("bm25", b), ("dense", d), ("hybrid", h)):
                rank = next((r for r, i in enumerate(np.asarray(ranking).tolist(), 1) if i in rel), None)
                row[name] = rank
                found[name] += rank is not None and rank <= 5
            ranks.append({"id": item["id"], **row})
        return found, ranks

    def answers(found):
        return (found["bm25"] >= 5, found["dense"] >= 5, found["hybrid"] >= max(found["bm25"], found["dense"]))

    found, ranks = counts(emb, queries)
    print(f"  p19 {found}", flush=True)
    for r in ranks:
        print(f"    {r}", flush=True)
    base = answers(found)
    rng = np.random.default_rng(19)
    perturbed = []
    for _ in range(20):
        E = emb + rng.normal(scale=1e-3, size=emb.shape).astype(np.float32)
        Q = queries + rng.normal(scale=1e-3, size=queries.shape).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        f, _ = counts(E, Q)
        perturbed.append(f)
        if answers(f) != base:
            raise SystemExit(f"❌ p19: a small perturbation of the embeddings changes the answers: {found} → {f}")
    print(f"  p19 perturbed counts: {sorted({tuple(f.values()) for f in perturbed})}", flush=True)
    return {"found": found, "ranks": ranks, "perturbed": sorted({tuple(f.values()) for f in perturbed}),
            "answers": {"bm25_half": base[0], "dense_half": base[1], "hybrid_best": base[2]}}


def e5_holmes(_) -> dict:
    ns = namespace(True, blocks=cells("TOOLS_B", "SOLUTION_20"))
    run("DEPTH_20 = 20", ns)
    from wb import text as wbtext
    start = time.perf_counter()
    index = ns["build_index_20"](ns["REF_CHUNKS"], wbtext.load_sentence_model("e5"), "b4-holmes-120-20")
    seconds = time.perf_counter() - start
    groups = {"30 questions": ns["QA"], "20 directes": [q for q in ns["QA"] if not q["paraphrased"]],
              "10 paraphrasées": [q for q in ns["QA"] if q["paraphrased"]]}
    rows = [(m, g, ns["evaluate_20"](index, items, m)) for m in ("bm25", "dense", "hybrid") for g, items in groups.items()]
    for r in rows:
        print(f"  e5 {r}", flush=True)
    return {"rows": rows, "embedding_seconds": round(seconds, 1)}


# ---------------------------------------------------------------------------------------------------------------------
FAST_CELLS = ("GIVEN_12", "SOLUTION_12", "CHECK_12", "GIVEN_13", "identical_13 = watson_13 = False", "EXPERIMENT_13",
              "GIVEN_14", "SOLUTION_14", "CHECK_14", "GIVEN_15", "SOLUTION_15", "EXPERIMENT_15", "GIVEN_16",
              "FULL_16 = {}", "SOLUTION_16", "EXPERIMENT_16", "TOOLS_B", "GIVEN_19", "GIVEN_20",
              "E5_HOLMES_20 = []", "SOLUTION_20", "CHECK_20", "GIVEN_21", "SOLUTION_21", "EXPERIMENT_21")


def _exec_cells(fast: bool, folder: str, names) -> dict:
    import chB4_code as C
    ns = namespace(fast, folder)
    for name in names:
        code = getattr(C, name) if hasattr(C, name) else name
        start = time.perf_counter()
        text = run(code, ns)
        bad = [line for line in text.splitlines() if line.startswith("❌")]
        print(f"  {'fast' if fast else 'full'} {name}: {time.perf_counter() - start:.0f} s"
              + (f" ❌ {bad[:3]}" if bad else ""), flush=True)
    return ns


def generations(_) -> dict:
    from wb import llm as wbllm
    folder = tempfile.mkdtemp(prefix="wb_b4_gen_")
    try:
        ns = _exec_cells(True, folder, FAST_CELLS)
        r21_fast = ns.get("rows_21")
        _exec_cells(False, folder, ("GIVEN_15", "SOLUTION_15", "GIVEN_16", "FULL_16 = {}", "SOLUTION_16",
                                    "EXPERIMENT_16"))
        full = _exec_cells(False, folder, ("TOOLS_B", "GIVEN_19", "GIVEN_20", "E5_HOLMES_20 = []", "SOLUTION_20",
                                           "CHECK_20", "GIVEN_21", "SOLUTION_21", "EXPERIMENT_21"))
        entries = json.loads((Path(folder) / "generations-solution.json").read_text(encoding="utf-8"))["entries"]
        target = ROOT / wbllm.SOLUTION_CACHE
        wbllm.write_json_atomic(target, {"version": wbllm.CACHE_VERSION, "source": "solution", "entries": entries})
        print(f"  generations: {len(entries)} replies → {target.relative_to(ROOT)}", flush=True)
        return {"n": len(entries), "r21_fast": r21_fast, "r21_full": full.get("rows_21")}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def p16_full(_) -> dict:
    """The full protocol of B4.16, read from the frozen cache written by `generations` (no new generation)."""
    from wb import llm as wbllm
    ns = namespace(False, blocks=cells("GIVEN_15", "SOLUTION_15", "GIVEN_16", "SOLUTION_16"))
    ns["CACHE"] = wbllm.GenerationCache("solution", folder=tempfile.mkdtemp(), frozen=ROOT / wbllm.SOLUTION_CACHE)
    grid_f = [[ns["accuracy_16"](f, ns["EXAMPLES_15"], s) for s in ns["SEEDS_16"]] for f in ns["FORMULATIONS_16"]]
    grid_o = [[ns["accuracy_16"](ns["SYSTEM_15"], [ns["EXAMPLES_15"][i] for i in o], s) for s in ns["SEEDS_16"]]
              for o in ns["ORDERS_16"]]
    if ns["CACHE"].misses:
        raise SystemExit("❌ p16_full: run `generations` first (the frozen cache misses replies)")
    out = {"model": "Qwen2.5-0.5B-Instruct", "snippets": len(ns["SNIPPETS_16"]), "grid_f": grid_f, "grid_o": grid_o}
    print(f"  p16_full {out}", flush=True)
    return out


def r21(_) -> dict:
    """The reference RAG on the 30 questions (Qwen2.5, hybrid search with MiniLM, k = 3), from the frozen cache."""
    from wb import llm as wbllm
    from wb import rag as wbrag
    from wb import text as wbtext
    chunks, _ = wbrag.reference_chunks("holmes")
    rag = wbrag.ReferenceRAG(wbllm.load_llm("qwen2.5"),
                             wbrag.ReferenceIndex(chunks, wbtext.load_sentence_model("minilm"), "b4-holmes-120-20"),
                             cache=wbllm.GenerationCache("solution", folder=tempfile.mkdtemp(),
                                                         frozen=ROOT / wbllm.SOLUTION_CACHE))
    rows = []
    for item in wbrag.load_qa("holmes"):
        out = rag.answer(item["question"])
        rows.append({"id": item["id"], "paraphrased": item["paraphrased"],
                     "found": bool(wbrag.relevant_ids(chunks, item) & set(out["ids"])),
                     "correct": wbrag.answer_in(out["answer"], item), "verified": out["verified"], "idk": out["idk"],
                     "reply": out["reply"]})
    summary = {k: sum(r[k] for r in rows) for k in ("found", "correct", "verified", "idk")}
    summary["correct_and_verified"] = sum(r["correct"] and r["verified"] for r in rows)
    print(f"  r21 {summary} (cache misses: {rag.cache.misses})", flush=True)
    return {"rows": rows, "summary": summary}


ENTRIES = {"c12": c12, "p13": p13, "p19": p19, "e5_holmes": e5_holmes, "generations": generations,
           "p16_full": p16_full, "r21": r21}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(ENTRIES), nargs="+")
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, fn in ENTRIES.items():
        if args.only and name not in args.only:
            continue
        start = time.perf_counter()
        print(f"{name}…", flush=True)
        data[name] = fn(None)
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
