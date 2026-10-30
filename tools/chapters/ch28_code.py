"""Code cells of the chapter 28 notebook (parts A to D), shared by build_ch28.py and history_ch28.py (used by Claude).

Each constant is the source of one cell: TOOLS (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell).
"""

TOOLS = r'''# Tools for the notebook exercises (parts A to D)
import json
import math
import re
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wb.nbtools import StopCheck, checked, filled, fr, returned, same, verdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md).")

WORK_DIR = Path(tempfile.mkdtemp(prefix="wb_ch28_"))   # a temporary folder for the files the exercises write
# The small network of the whole notebook (both modes): SqueezeNet 1.1, 4.7 MB of weights, its convolutional part only,
# frozen, in eval() mode, ReLUs without inplace (wb.datasets.load_pretrained). NET[i] is its i-th module.
NET, MEAN, STD = wb.datasets.load_pretrained("squeezenet1_1")
LAYERS = {"conv1": 1, "fire2": 3, "fire3": 4, "fire4": 6, "fire5": 7, "fire6": 9, "fire7": 10, "fire8": 11, "fire9": 12}
SIZE = 160 if FAST_MODE else 256        # the longer side of the images of parts B and C
# The full mode (🚀) of 28.16, 28.22, 28.23 and 28.24 can use VGG16 (528 MiB, downloaded once) instead:
VGG_LAYERS = {"block1_conv1": 1, "block1_conv2": 3, "block2_conv1": 6, "block2_conv2": 8, "block3_conv1": 11,
              "block3_conv2": 13, "block3_conv3": 15, "block4_conv1": 18, "block4_conv2": 20, "block4_conv3": 22,
              "block5_conv1": 25, "block5_conv2": 27, "block5_conv3": 29}


def pct(x, digits=1):
    """A share written the French way: 0.734 -> '73,4 %'."""
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def unpack(text):
    """Data packed by the builder (zlib + base64 of a JSON text)."""
    import base64
    import zlib
    return json.loads(zlib.decompress(base64.b64decode(text)).decode("utf-8"))


def to_batch(img):
    """A (3, H, W) array or tensor in [0, 1] -> a float32 tensor (1, 3, H, W) (a copy, without graph)."""
    return torch.as_tensor(np.asarray(img, dtype=np.float32)).unsqueeze(0).clone()


def to_image(x):
    """A tensor (1, 3, H, W) or (3, H, W) -> a numpy array (3, H, W) in [0, 1] (detached, clipped)."""
    x = x.detach()
    return np.clip((x[0] if x.dim() == 4 else x).numpy(), 0, 1).astype(np.float32)


def normalize(x):
    """ImageNet normalisation of an image batch in [0, 1], as the network expects (differentiable)."""
    return (x - MEAN) / STD


def show_images(images, titles=None, size=2.6, cols=None):
    """Show (3, H, W) arrays in [0, 1] side by side (cols per row)."""
    n = len(images)
    cols = cols or n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, img in enumerate(images):
        ax = axes.ravel()[k]
        ax.imshow(np.clip(np.asarray(img).transpose(1, 2, 0), 0, 1))
        if titles:
            ax.set_title(titles[k], fontsize=9)
    plt.tight_layout()
    plt.show()


def weights_signature(net):
    """A number that changes as soon as one weight of `net` changes."""
    with torch.no_grad():
        return float(sum((p.double() * (k + 1)).sum() for k, p in enumerate(net.parameters())))


def high_freq_share(img):
    """The share of the variance of an image (3, H, W) that sits in its highest spatial frequencies (above half the
    Nyquist frequency): about 0.75 for white noise, a few % for a photo."""
    a = np.asarray(img, dtype=np.float64).mean(axis=0)
    spec = np.abs(np.fft.fft2(a - a.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing="ij")
    high = np.hypot(fx, fy) > 0.25
    return float(spec[high].sum() / spec.sum())


def dominant_period(img):
    """The period in pixels of the strongest spatial frequency of an image (3, H, W) (its mean over the channels)."""
    a = np.asarray(img, dtype=np.float64).mean(axis=0)
    spec = np.abs(np.fft.fft2(a - a.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing="ij")
    spec[0, 0] = 0
    k = np.unravel_index(np.argmax(spec), spec.shape)
    return float(1.0 / np.hypot(fx[k], fy[k]))


def need(ex_id, *names):
    """Stop a check cell with ⏳ while a function of an earlier exercise is not written yet."""
    for name in names:
        fn = globals().get(name)
        if fn is None:
            print(f"⏳ Ex {ex_id} : définis d'abord {name}.")
            raise StopCheck
        try:
            fn.__wb_probe__ = True
        except AttributeError:
            pass


torch.manual_seed(0)
print(f"Réseau : SqueezeNet 1.1, {sum(p.numel() for p in NET.parameters()):_} paramètres dans ses convolutions, ".replace("_", " ")
      + f"tous gelés ; images de {SIZE} pixels de côté au plus.")'''

# --- Part A ---------------------------------------------------------------------------------------------------------
TODO_11 = r'''def extract_11(net, x, layers):
    """The activations of several layers in ONE forward pass, recorded by hooks.

    net: an nn.Sequential (NET); x: a batch (N, 3, H, W) already normalised; layers: a dict {name: index}. Returns a
    dict {name: output of net[index]}, the tensors as they come out (with their graph: no detach). One forward hook per
    layer, registered before the pass and removed after it, even if the pass fails (try/finally)."""
    raise NotImplementedError("extract_11")'''

SOLUTION_11 = r'''def extract_11(net, x, layers):
    """The activations of several layers in ONE forward pass, recorded by hooks.

    net: an nn.Sequential (NET); x: a batch (N, 3, H, W) already normalised; layers: a dict {name: index}. Returns a
    dict {name: output of net[index]}, the tensors as they come out (with their graph: no detach). One forward hook per
    layer, registered before the pass and removed after it, even if the pass fails (try/finally)."""
    acts, handles = {}, []
    for name, index in layers.items():
        handles.append(net[index].register_forward_hook(lambda module, inputs, output, name=name: acts.__setitem__(name, output)))
    try:
        net(x)
    finally:
        for handle in handles:
            handle.remove()
    return acts'''

CHECK_11 = r'''with checked("28.11"):
    x_11 = normalize(to_batch(wb.datasets.load_sample_image("china", 128)))
    acts_11 = extract_11(NET, x_11, LAYERS)
    if not returned("28.11", "extract_11", acts_11):
        raise StopCheck
    verdict("28.11", isinstance(acts_11, dict) and set(acts_11) == set(LAYERS), "un dictionnaire, une entrée par couche.",
            f"attendu un dict aux clés {sorted(LAYERS)}, reçu {type(acts_11).__name__} {sorted(acts_11) if isinstance(acts_11, dict) else ''}.")
    with torch.no_grad():
        want_11 = NET[:LAYERS["fire5"] + 1](x_11)
    same("28.11", "acts['fire5'] (comparé à NET[:8](x))", acts_11["fire5"].detach(), want_11, rtol=1e-5, atol=1e-6)
    left_11 = sum(len(m._forward_hooks) for m in NET.modules())
    verdict("28.11", left_11 == 0, "aucun hook ne reste accroché après l'appel.",
            f"{left_11} hook(s) encore accroché(s) : retire-les (handle.remove()) dans un finally.")
    x_grad_11 = x_11.clone().requires_grad_(True)
    out_11 = extract_11(NET, x_grad_11, {"fire3": LAYERS["fire3"]})["fire3"]
    verdict("28.11", out_11.requires_grad, "les activations gardent leur graphe : on pourra dériver par rapport à l'image.",
            "l'activation n'a plus de graphe : pas de detach() ni de torch.no_grad() dans extract_11.")
    wb.check("28.11a", int(acts_11["fire4"].numel()), computed=True)
    wb.check("28.11b", int(extract_11(NET, normalize(torch.zeros(1, 3, 192, 192)), {"f9": LAYERS["fire9"]})["f9"].shape[-1]), computed=True)'''

RECORD_11 = r'''wb.record("28.11a", 38400, mistakes={"la photo n'est pas carrée : elle mesure 128 × 85 pixels": 256 * 15 * 15,
                                      "compte les valeurs de tous les canaux, pas seulement une carte": 150})
wb.record("28.11b", 11, mistakes={"à 192 pixels, pas à 128 : relance avec une image 192 × 192": 7})'''

# 28.12 and 28.13: the visualisation of one filter
TODO_13 = r'''def activation_13(net, x, layer_index, channel):
    """a(x): the mean of the squared activation map of filter `channel` of net[layer_index], for an image batch x in
    [0, 1] (normalised inside), a 0-d tensor with its graph."""
    raise NotImplementedError("activation_13")


def visualize_filter_13(net, layer_index, channel, size=128, steps=100, lr=0.05, seed=0):
    """The image (3, size, size) in [0, 1], a float32 numpy array, that excites filter `channel` of net[layer_index]:
    start from torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4 (a grey noise),
    then `steps` steps of Adam([x], lr) on -activation_13(...), the pixels clamped to [0, 1] after every step. The
    weights of the network are never changed."""
    raise NotImplementedError("visualize_filter_13")'''

SOLUTION_13 = r'''def activation_13(net, x, layer_index, channel):
    """a(x): the mean of the squared activation map of filter `channel` of net[layer_index], for an image batch x in
    [0, 1] (normalised inside), a 0-d tensor with its graph."""
    out = net[:layer_index + 1](normalize(x))
    return (out[:, channel] ** 2).mean()


def visualize_filter_13(net, layer_index, channel, size=128, steps=100, lr=0.05, seed=0):
    """The image (3, size, size) in [0, 1], a float32 numpy array, that excites filter `channel` of net[layer_index]:
    start from torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4 (a grey noise),
    then `steps` steps of Adam([x], lr) on -activation_13(...), the pixels clamped to [0, 1] after every step. The
    weights of the network are never changed."""
    x = (torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = -activation_13(net, x, layer_index, channel)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return to_image(x)'''

CHECK_13 = r'''with checked("28.13"):
    noise_13 = to_batch(np.random.default_rng(13).uniform(0.4, 0.6, (3, 64, 64)))
    a_13 = activation_13(NET, noise_13, LAYERS["fire3"], 5)
    with torch.no_grad():
        want_a_13 = float((NET[:LAYERS["fire3"] + 1](normalize(noise_13))[:, 5] ** 2).mean())
    same("28.13", "activation_13(NET, bruit, fire3, 5)", float(a_13), want_a_13, rtol=1e-5)
    signature_13 = weights_signature(NET)
    start_13 = to_image(torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(1)) * 0.2 + 0.4)
    img_13 = visualize_filter_13(NET, LAYERS["fire3"], 5, size=64, steps=30, seed=1)
    if not returned("28.13", "visualize_filter_13", img_13):
        raise StopCheck
    img_13 = np.asarray(img_13)
    verdict("28.13", img_13.shape == (3, 64, 64) and img_13.dtype == np.float32,
            "une image float32 (3, 64, 64).", f"attendu un tableau float32 (3, 64, 64), reçu {img_13.dtype} {img_13.shape}.")
    verdict("28.13", img_13.min() >= 0 and img_13.max() <= 1, "les pixels restent dans [0, 1].",
            f"des pixels hors de [0, 1] ({fr(img_13.min())} à {fr(img_13.max())}) : clamp_ après chaque pas.")
    verdict("28.13", weights_signature(NET) == signature_13, "les poids du réseau n'ont pas bougé.",
            "les poids du réseau ont changé : l'optimiseur ne doit recevoir que l'image.")
    with torch.no_grad():
        before_13 = float(activation_13(NET, to_batch(start_13), LAYERS["fire3"], 5))
        after_13 = float(activation_13(NET, to_batch(img_13), LAYERS["fire3"], 5))
    verdict("28.13", after_13 > 5 * before_13, f"l’activation du filtre passe de {fr(before_13, 4)} à {fr(after_13, 1)}.",
            f"l'activation ne monte pas assez ({fr(before_13)} → {fr(after_13)}) : une montée (loss = −a), le bon départ ?")'''

EXPERIMENT_13 = r'''with checked("28.13"):
    picks_13 = [("conv1", 7), ("conv1", 30), ("fire3", 5), ("fire5", 40), ("fire7", 100), ("fire9", 200)]
    t0_13 = time.perf_counter()
    images_13 = [visualize_filter_13(NET, LAYERS[name], ch, size=128, steps=100 if FAST_MODE else 300) for name, ch in picks_13]
    print(f"6 visualisations en {fr(time.perf_counter() - t0_13, 1)} s")
    show_images(images_13, [f"{name}, filtre {ch}" for name, ch in picks_13], cols=6, size=2.2)'''

# 28.14: one image per layer
TODO_14 = r'''def visualize_layer_14(net, layer_index, size=128, steps=100, lr=0.05, seed=0):
    """Same as visualize_filter_13, but the quantity that goes up is the mean of the squared activations of ALL the
    filters of net[layer_index] (the whole layer)."""
    raise NotImplementedError("visualize_layer_14")'''

SOLUTION_14 = r'''def visualize_layer_14(net, layer_index, size=128, steps=100, lr=0.05, seed=0):
    """Same as visualize_filter_13, but the quantity that goes up is the mean of the squared activations of ALL the
    filters of net[layer_index] (the whole layer)."""
    x = (torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = -(net[:layer_index + 1](normalize(x)) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return to_image(x)'''

CHECK_14 = r'''with checked("28.14"):
    t0_14 = time.perf_counter()
    gallery_14 = {name: visualize_layer_14(NET, index, size=128, steps=80 if FAST_MODE else 300) for name, index in LAYERS.items()}
    if not returned("28.14", "visualize_layer_14", gallery_14["conv1"]):
        raise StopCheck
    print(f"9 couches en {fr(time.perf_counter() - t0_14, 1)} s")
    gains_14 = []
    for name, img in gallery_14.items():
        with torch.no_grad():
            start = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(0)) * 0.2 + 0.4
            a0 = float((NET[:LAYERS[name] + 1](normalize(start)) ** 2).mean())
            a1 = float((NET[:LAYERS[name] + 1](normalize(to_batch(img))) ** 2).mean())
        gains_14.append(a1 / a0)
    verdict("28.14", min(gains_14) > 2, f"chaque couche s'active plus fort (gain minimal × {fr(min(gains_14), 1)}).",
            f"une couche ne s'active pas davantage (gain × {fr(min(gains_14), 2)}) : la loss porte-t-elle sur toute la couche ?")
    show_images(list(gallery_14.values()), [f"{n} (×{fr(g, 0)})" for n, g in zip(gallery_14, gains_14)], cols=9, size=1.8)
    hf_14 = {name: high_freq_share(img) for name, img in gallery_14.items()}
    print("part de hautes fréquences : " + ", ".join(f"{n} {pct(v, 0)}" for n, v in hf_14.items()))'''

# 28.15: regularisation
TODO_15 = r'''def visualize_reg_15(net, layer_index, channel, size=128, steps=100, lr=0.05, seed=0, jitter=0, blur_every=0,
                     octaves=1, octave_scale=1.4):
    """visualize_filter_13 with three regularisations:
    - jitter: before each step, the image is rolled by (dy, dx) drawn uniformly in [-jitter, jitter] (torch.roll on the
      two last axes, a torch.Generator seeded with seed + 1), the step is made on the rolled image, then rolled back;
    - blur_every: every blur_every steps (if > 0), a 3 x 3 box blur of the image (F.avg_pool2d(x, 3, 1, 1,
      count_include_pad=False)), under torch.no_grad();
    - octaves: optimise first an image of size / octave_scale**(octaves - 1) pixels, then enlarge it
      (F.interpolate, bilinear, align_corners=False) by octave_scale up to `size`, `steps` steps at each scale.
    Return the final image (3, size, size), float32, in [0, 1]."""
    raise NotImplementedError("visualize_reg_15")'''

SOLUTION_15 = r'''def visualize_reg_15(net, layer_index, channel, size=128, steps=100, lr=0.05, seed=0, jitter=0, blur_every=0,
                     octaves=1, octave_scale=1.4):
    """visualize_filter_13 with three regularisations:
    - jitter: before each step, the image is rolled by (dy, dx) drawn uniformly in [-jitter, jitter] (torch.roll on the
      two last axes, a torch.Generator seeded with seed + 1), the step is made on the rolled image, then rolled back;
    - blur_every: every blur_every steps (if > 0), a 3 x 3 box blur of the image (F.avg_pool2d(x, 3, 1, 1,
      count_include_pad=False)), under torch.no_grad();
    - octaves: optimise first an image of size / octave_scale**(octaves - 1) pixels, then enlarge it
      (F.interpolate, bilinear, align_corners=False) by octave_scale up to `size`, `steps` steps at each scale.
    Return the final image (3, size, size), float32, in [0, 1]."""
    shifts = torch.Generator().manual_seed(seed + 1)
    sizes = [max(16, round(size / octave_scale ** k)) for k in range(octaves - 1, -1, -1)]
    x = torch.rand(1, 3, sizes[0], sizes[0], generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4
    for k, s in enumerate(sizes):
        if k > 0:
            x = F.interpolate(x.detach(), size=(s, s), mode="bilinear", align_corners=False).clamp(0, 1)
        x = x.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        for step in range(steps):
            dy, dx = (torch.randint(-jitter, jitter + 1, (2,), generator=shifts).tolist() if jitter else (0, 0))
            optimizer.zero_grad()
            rolled = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            loss = -activation_13(net, rolled, layer_index, channel)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if blur_every and (step + 1) % blur_every == 0:
                    x.copy_(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False))
                x.clamp_(0, 1)
    return to_image(x)'''

CHECK_15 = r'''with checked("28.15"):
    need("28.15", "activation_13")
    plain_15 = visualize_reg_15(NET, LAYERS["fire7"], 100, size=64, steps=20)
    if not returned("28.15", "visualize_reg_15", plain_15):
        raise StopCheck
    ref_15 = visualize_filter_13(NET, LAYERS["fire7"], 100, size=64, steps=20)
    same("28.15", "sans régularisation (jitter=0, blur_every=0, octaves=1), même image que visualize_filter_13",
         np.asarray(plain_15), ref_15, rtol=1e-4, atol=1e-4)
    reg_15 = np.asarray(visualize_reg_15(NET, LAYERS["fire7"], 100, size=64, steps=20, jitter=4, blur_every=5, octaves=2))
    verdict("28.15", reg_15.shape == (3, 64, 64) and 0 <= reg_15.min() and reg_15.max() <= 1,
            "avec les trois régularisations : une image (3, 64, 64) dans [0, 1].",
            f"attendu (3, 64, 64) dans [0, 1], reçu {reg_15.shape}, de {fr(reg_15.min())} à {fr(reg_15.max())}.")'''

EXPERIMENT_15 = r'''with checked("28.15"):
    settings_15 = {"brut": {}, "décalages": {"jitter": 6}, "flou": {"blur_every": 8},
                   "octaves": {"octaves": 3}, "les trois": {"jitter": 6, "blur_every": 8, "octaves": 3}}
    steps_15 = 60 if FAST_MODE else 200
    results_15 = {}
    for name, kw in settings_15.items():
        img = visualize_reg_15(NET, LAYERS["fire7"], 100, size=128, steps=steps_15, **kw)
        with torch.no_grad():
            act = float(activation_13(NET, to_batch(img), LAYERS["fire7"], 100))
        results_15[name] = (img, act, high_freq_share(img))
    for name, (img, act, hf) in results_15.items():
        print(f"{name:>10s} : activation {fr(act, 1):>7s}, hautes fréquences {pct(hf, 0)}")
    show_images([r[0] for r in results_15.values()], list(results_15), cols=5, size=2.4)'''

# --- Part B -----------------------------------------------------------------------------------------------------------
GIVEN_B = r'''# The photo of part B: a free photo shipped with scikit-learn (CC BY 2.0)
PHOTO_B = wb.datasets.load_sample_image("china", SIZE)
print(wb.datasets.image_credit("china"))
DREAM_LAYERS = {"fire4": 1.0, "fire6": 1.5} if FAST_MODE else {"fire4": 1.0, "fire6": 1.5, "fire8": 1.0}'''

TODO_16 = r'''def dream_loss_16(net, x, layer_weights):
    """L_dream(x) = sum over the layers of weight * mean(A**2), A the activation of net[LAYERS[name]] for the batch x in
    [0, 1] (normalised inside); layer_weights: {name: weight}. A 0-d tensor with its graph."""
    raise NotImplementedError("dream_loss_16")


def deep_dream_16(net, photo, layer_weights, steps=40, lr=0.01, octaves=3, octave_scale=1.4, mask=None):
    """Deep dream of `photo` (3, H, W) in [0, 1]: gradient ASCENT on dream_loss_16, at `octaves` scales (the smallest
    first: H / octave_scale**(octaves - 1)). At each scale: the dreamed image so far, enlarged (bilinear) to this
    scale, plus the details lost by shrinking the photo (the photo at this scale minus the photo shrunk once more and
    re-enlarged), clamped to [0, 1]; then `steps` steps x <- x + lr * g / mean(|g|) (g the gradient, normalised by its
    mean absolute value + 1e-8), multiplied by `mask` (a tensor (H', W') of 0 and 1 resized to the scale, nearest)
    if given, the pixels clamped after each step. Returns (3, H, W), float32, in [0, 1]."""
    raise NotImplementedError("deep_dream_16")'''

SOLUTION_16 = r'''def dream_loss_16(net, x, layer_weights):
    """L_dream(x) = sum over the layers of weight * mean(A**2), A the activation of net[LAYERS[name]] for the batch x in
    [0, 1] (normalised inside); layer_weights: {name: weight}. A 0-d tensor with its graph."""
    acts = extract_11(net, normalize(x), {name: LAYERS[name] for name in layer_weights})
    return sum(w * (acts[name] ** 2).mean() for name, w in layer_weights.items())


def deep_dream_16(net, photo, layer_weights, steps=40, lr=0.01, octaves=3, octave_scale=1.4, mask=None):
    """Deep dream of `photo` (3, H, W) in [0, 1]: gradient ASCENT on dream_loss_16, at `octaves` scales (the smallest
    first: H / octave_scale**(octaves - 1)). At each scale: the dreamed image so far, enlarged (bilinear) to this
    scale, plus the details lost by shrinking the photo (the photo at this scale minus the photo shrunk once more and
    re-enlarged), clamped to [0, 1]; then `steps` steps x <- x + lr * g / mean(|g|) (g the gradient, normalised by its
    mean absolute value + 1e-8), multiplied by `mask` (a tensor (H', W') of 0 and 1 resized to the scale, nearest)
    if given, the pixels clamped after each step. Returns (3, H, W), float32, in [0, 1]."""
    base = to_batch(photo)
    H, W = base.shape[-2:]
    shapes = [(max(16, round(H / octave_scale ** k)), max(16, round(W / octave_scale ** k))) for k in range(octaves - 1, -1, -1)]
    resize = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
    dream = None
    for k, shape in enumerate(shapes):
        photo_k = resize(base, shape)
        if dream is None:
            x = photo_k.clone()
        else:
            detail = photo_k - resize(resize(base, shapes[k - 1]), shape)
            x = (resize(dream, shape) + detail).clamp(0, 1)
        m = None
        if mask is not None:
            m = F.interpolate(torch.as_tensor(mask, dtype=torch.float32)[None, None], size=shape, mode="nearest")
        x = x.detach().requires_grad_(True)
        for _ in range(steps):
            loss = dream_loss_16(net, x, layer_weights)
            (g,) = torch.autograd.grad(loss, x)
            g = g / (g.abs().mean() + 1e-8)
            if m is not None:
                g = g * m
            with torch.no_grad():
                x += lr * g
                x.clamp_(0, 1)
        dream = x.detach()
    return to_image(dream)'''

CHECK_16 = r'''with checked("28.16"):
    need("28.16", "extract_11")
    small_16 = wb.datasets.load_sample_image("flower", 64)
    loss0_16 = dream_loss_16(NET, to_batch(small_16), {"fire3": 1.0, "fire5": 2.0})
    with torch.no_grad():
        xn_16 = normalize(to_batch(small_16))
        want0_16 = float((NET[:5](xn_16) ** 2).mean() + 2 * (NET[:8](xn_16) ** 2).mean())
    same("28.16", "dream_loss_16(fleur, {fire3: 1, fire5: 2})", float(loss0_16), want0_16, rtol=1e-5)
    signature_16 = weights_signature(NET)
    dream_16 = deep_dream_16(NET, small_16, {"fire3": 1.0, "fire5": 2.0}, steps=8, octaves=2)
    if not returned("28.16", "deep_dream_16", dream_16):
        raise StopCheck
    dream_16 = np.asarray(dream_16)
    verdict("28.16", dream_16.shape == small_16.shape and 0 <= dream_16.min() and dream_16.max() <= 1,
            "une image de la taille de la photo, dans [0, 1].", f"attendu {small_16.shape} dans [0, 1], reçu {dream_16.shape}.")
    with torch.no_grad():
        gain_16 = float(dream_loss_16(NET, to_batch(dream_16), {"fire3": 1.0, "fire5": 2.0})) / float(loss0_16)
    verdict("28.16", gain_16 > 1.3, f"le rêve fait monter la loss de rêve (× {fr(gain_16, 2)}).",
            f"la loss de rêve ne monte pas (× {fr(gain_16, 2)}) : une montée, sur l'image, avec le gradient normalisé ?")
    verdict("28.16", weights_signature(NET) == signature_16, "les poids du réseau n'ont pas bougé.", "les poids ont changé.")
    mask_16 = torch.zeros(dream_16.shape[1:])
    mask_16[:, : dream_16.shape[2] // 2] = 1
    half_16 = np.asarray(deep_dream_16(NET, small_16, {"fire3": 1.0}, steps=6, octaves=1, mask=mask_16))
    right_16 = float(np.abs(half_16 - small_16)[:, :, dream_16.shape[2] // 2:].max())
    verdict("28.16", right_16 < 1e-6, "avec un masque, la moitié masquée ne bouge pas.",
            f"la moitié masquée a changé (jusqu'à {fr(right_16)}) : multiplie le gradient par le masque (octaves=1 ici).")'''

EXPERIMENT_16 = r'''with checked("28.16"):
    t0_16 = time.perf_counter()
    dreams_16 = [deep_dream_16(NET, PHOTO_B, w, steps=30 if FAST_MODE else 80, lr=0.012)
                 for w in ({"conv1": 1.0, "fire2": 1.0}, DREAM_LAYERS, {"fire8": 1.0, "fire9": 1.0})]
    print(f"3 rêves en {fr(time.perf_counter() - t0_16, 1)} s")
    show_images([PHOTO_B] + dreams_16, ["la photo", "couches précoces", "DREAM_LAYERS", "couches profondes"], cols=4, size=3)'''

TODO_17 = r'''layers_17 = ...    # your choice of {layer name: weight}, at least two layers
mask_17 = ...      # a tensor (H, W) of 0 and 1, the size of PHOTO_B: 1 where the dream may change the photo'''

SOLUTION_17 = r'''layers_17 = {"fire5": 1.0, "fire7": 2.0}
mask_17 = torch.zeros(PHOTO_B.shape[1:])
mask_17[: PHOTO_B.shape[1] * 2 // 5] = 1        # the sky only (the top of the photo)'''

CHECK_17 = r'''with checked("28.17"):
    need("28.17", "deep_dream_16")
    if not filled(layers_17, mask_17):
        print("⏳ Ex 28.17 : choisis tes couches et ton masque dans la cellule précédente.")
        raise StopCheck
    m_17 = torch.as_tensor(mask_17, dtype=torch.float32)
    verdict("28.17", tuple(m_17.shape) == tuple(PHOTO_B.shape[1:]) and bool(((m_17 == 0) | (m_17 == 1)).all()),
            "un masque de 0 et de 1 à la taille de la photo.", f"attendu un masque {tuple(PHOTO_B.shape[1:])} de 0 et de 1, reçu {tuple(m_17.shape)}.")
    dream_17 = np.asarray(deep_dream_16(NET, PHOTO_B, layers_17, steps=30 if FAST_MODE else 80, lr=0.012, octaves=1, mask=m_17))
    change_17 = np.abs(dream_17 - PHOTO_B).mean(axis=0)
    inside_17 = float(change_17[m_17.numpy() == 1].mean()) if m_17.sum() > 0 else 0.0
    outside_17 = float(change_17[m_17.numpy() == 0].max()) if (m_17 == 0).any() else 0.0
    print(f"changement moyen dans la zone : {fr(inside_17, 3)} ; changement maximal hors de la zone : {fr(outside_17, 6)}")
    verdict("28.17", outside_17 < 1e-6, "hors de la zone, la photo n'a pas bougé.", "des pixels ont changé hors de la zone.")
    show_images([PHOTO_B, dream_17, np.stack([change_17 / max(change_17.max(), 1e-8)] * 3)],
                ["la photo", "le rêve dans la zone", "où l'image a changé"], size=3.2)'''

# --- Part C -----------------------------------------------------------------------------------------------------------
GIVEN_C = r'''# The images and the settings of part C: a free photo (content) and a public-domain painting (style)
CONTENT_C = wb.datasets.load_sample_image("china", SIZE)
STYLE_C = wb.datasets.load_style_image("starry_night", SIZE)
CONTENT_LAYERS = ["fire6"]                                # one deep layer for the content
STYLE_LAYERS = ["conv1", "fire2", "fire3", "fire5"]       # four layers for the style, equal weights
ALPHA, BETA, TV = 1.0, 10.0, 0.05                         # tuned for these conventions (see 28.18 and 28.23)
STEPS_C = 150 if FAST_MODE else 400
print(wb.datasets.image_credit("starry_night"))
show_images([CONTENT_C, STYLE_C], ["contenu", "style"], size=3)'''

TODO_18 = r'''def gram_matrix_18(A):
    """The normalised Gram matrix of activations A: a tensor (C, H, W) -> (C, C), or (N, C, H, W) -> (N, C, C);
    G = F F^T / (C * H * W) with F the maps flattened into rows. Differentiable (no numpy inside)."""
    raise NotImplementedError("gram_matrix_18")


def tv_loss_18(x):
    """The total variation of an image batch x (N, 3, H, W): the sum of the squared differences between neighbouring
    pixels, vertically and horizontally, over every channel and image. A 0-d tensor."""
    raise NotImplementedError("tv_loss_18")'''

SOLUTION_18 = r'''def gram_matrix_18(A):
    """The normalised Gram matrix of activations A: a tensor (C, H, W) -> (C, C), or (N, C, H, W) -> (N, C, C);
    G = F F^T / (C * H * W) with F the maps flattened into rows. Differentiable (no numpy inside)."""
    C, H, W = A.shape[-3:]
    flat = A.reshape(*A.shape[:-2], H * W)
    return flat @ flat.transpose(-1, -2) / (C * H * W)


def tv_loss_18(x):
    """The total variation of an image batch x (N, 3, H, W): the sum of the squared differences between neighbouring
    pixels, vertically and horizontally, over every channel and image. A 0-d tensor."""
    return ((x[..., 1:, :] - x[..., :-1, :]) ** 2).sum() + ((x[..., :, 1:] - x[..., :, :-1]) ** 2).sum()'''

CHECK_18 = r'''with checked("28.18"):
    rng_18 = np.random.default_rng(18)
    A_18 = rng_18.random((5, 7, 6))
    G_18 = gram_matrix_18(torch.tensor(A_18))
    if not returned("28.18", "gram_matrix_18", G_18):
        raise StopCheck
    same("28.18", "gram_matrix_18 sur (5, 7, 6), contre np.einsum", G_18, np.einsum("chw,dhw->cd", A_18, A_18) / A_18.size)
    B_18 = rng_18.random((2, 4, 3, 5))
    same("28.18", "gram_matrix_18 sur un batch (2, 4, 3, 5)", gram_matrix_18(torch.tensor(B_18)),
         np.einsum("nchw,ndhw->ncd", B_18, B_18) / (4 * 3 * 5))
    A_grad_18 = torch.tensor(A_18, requires_grad=True)
    gram_matrix_18(A_grad_18).sum().backward()
    verdict("28.18", A_grad_18.grad is not None, "le gradient traverse gram_matrix_18.", "pas de gradient : du NumPy dans la fonction ?")
    X_18 = rng_18.random((2, 3, 4, 5))
    same("28.18", "tv_loss_18 sur (2, 3, 4, 5)", float(tv_loss_18(torch.tensor(X_18))),
         float((np.diff(X_18, axis=2) ** 2).sum() + (np.diff(X_18, axis=3) ** 2).sum()))'''

# The style transfer of the reference (28.23), reused by the history script for 28.19, 28.20 and 28.22
TODO_23 = r'''def style_transfer_23(net, content, style, content_layers, style_layers, alpha, beta, tv, steps=150, lr=0.05,
                      init="content", seed=0, every=1):
    """Gatys's style transfer with Adam on the pixels.

    content, style: (3, H, W) arrays in [0, 1] (of any sizes); the targets are computed once, without graph:
    the activations of content at content_layers, the Gram matrices (gram_matrix_18) of style at style_layers.
    Start: a copy of content (init="content") or torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed))
    (init="noise"). At each step: ONE pass of the image through extract_11 for all the needed layers,
      L_c = sum over content_layers of mean((A - A_c)**2)          (F.mse_loss, a MEAN)
      L_s = sum over style_layers of sum((G - G_s)**2)              (a SUM over the C x C entries)
      L_tv = tv_loss_18(x),
    loss = alpha * L_c + beta * L_s + tv * L_tv, a step of Adam([x], lr), the pixels clamped to [0, 1].
    Returns (image (3, H, W) float32 in [0, 1], history): history = {"content": [...], "style": [...], "tv": [...]},
    the three WEIGHTED terms (alpha * L_c...) as floats, recorded before each step whose index is a multiple of
    `every`, plus once more after the last step."""
    raise NotImplementedError("style_transfer_23")'''

SOLUTION_23 = r'''def style_transfer_23(net, content, style, content_layers, style_layers, alpha, beta, tv, steps=150, lr=0.05,
                      init="content", seed=0, every=1):
    """Gatys's style transfer with Adam on the pixels.

    content, style: (3, H, W) arrays in [0, 1] (of any sizes); the targets are computed once, without graph:
    the activations of content at content_layers, the Gram matrices (gram_matrix_18) of style at style_layers.
    Start: a copy of content (init="content") or torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed))
    (init="noise"). At each step: ONE pass of the image through extract_11 for all the needed layers,
      L_c = sum over content_layers of mean((A - A_c)**2)          (F.mse_loss, a MEAN)
      L_s = sum over style_layers of sum((G - G_s)**2)              (a SUM over the C x C entries)
      L_tv = tv_loss_18(x),
    loss = alpha * L_c + beta * L_s + tv * L_tv, a step of Adam([x], lr), the pixels clamped to [0, 1].
    Returns (image (3, H, W) float32 in [0, 1], history): history = {"content": [...], "style": [...], "tv": [...]},
    the three WEIGHTED terms (alpha * L_c...) as floats, recorded before each step whose index is a multiple of
    `every`, plus once more after the last step."""
    layers = {name: LAYERS[name] for name in set(content_layers) | set(style_layers)}
    c, s = to_batch(content), to_batch(style)
    with torch.no_grad():
        acts_c = extract_11(net, normalize(c), {n: LAYERS[n] for n in content_layers})
        acts_s = extract_11(net, normalize(s), {n: LAYERS[n] for n in style_layers})
        targets_c = {n: acts_c[n] for n in content_layers}
        targets_s = {n: gram_matrix_18(acts_s[n]) for n in style_layers}
    if init == "content":
        x = c.clone()
    else:
        x = torch.rand(c.shape, generator=torch.Generator().manual_seed(seed))
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    history = {"content": [], "style": [], "tv": []}

    def terms(x):
        acts = extract_11(net, normalize(x), layers)
        l_c = sum(F.mse_loss(acts[n], targets_c[n]) for n in content_layers)
        l_s = sum(((gram_matrix_18(acts[n]) - targets_s[n]) ** 2).sum() for n in style_layers)
        return alpha * l_c, beta * l_s, tv * tv_loss_18(x)

    for step in range(steps + 1):
        t_c, t_s, t_tv = terms(x)
        if step % every == 0 or step == steps:
            for key, value in zip(("content", "style", "tv"), (t_c, t_s, t_tv)):
                history[key].append(float(value.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        (t_c + t_s + t_tv).backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return to_image(x), history'''

CHECK_23 = r'''with checked("28.23"):
    need("28.23", "extract_11", "gram_matrix_18", "tv_loss_18")
    small_c_23, small_s_23 = wb.datasets.load_sample_image("china", 48), wb.datasets.load_style_image("great_wave", 40)
    signature_23 = weights_signature(NET)
    out_23 = style_transfer_23(NET, small_c_23, small_s_23, ["fire5"], ["conv1", "fire3"], 1.0, 10.0, 0.05, steps=12)
    if not returned("28.23", "style_transfer_23", out_23):
        raise StopCheck
    img_23, hist_23 = out_23
    img_23 = np.asarray(img_23)
    verdict("28.23", img_23.shape == small_c_23.shape and img_23.dtype == np.float32 and 0 <= img_23.min() and img_23.max() <= 1,
            "une image float32 de la taille du contenu, dans [0, 1].", f"attendu float32 {small_c_23.shape} dans [0, 1], reçu {img_23.dtype} {img_23.shape}.")
    verdict("28.23", set(hist_23) == {"content", "style", "tv"} and all(len(v) == 13 for v in hist_23.values()),
            "un historique de 13 valeurs (12 pas, plus la fin) pour chacun des trois termes.",
            f"attendu les clés content, style, tv avec 13 valeurs, reçu { {k: len(v) for k, v in hist_23.items()} }.")
    verdict("28.23", hist_23["content"][0] == 0.0, "partir du contenu donne une loss de contenu nulle au départ.",
            f"la loss de contenu de départ vaut {fr(hist_23['content'][0])} : part d'une copie du contenu (init=\"content\").")
    total_23 = [sum(v) for v in zip(*hist_23.values())]
    verdict("28.23", total_23[-1] < 0.5 * total_23[0], f"la loss totale baisse ({fr(total_23[0], 1)} → {fr(total_23[-1], 1)}).",
            f"la loss totale baisse trop peu ({fr(total_23[0], 1)} → {fr(total_23[-1], 1)}) : une descente, sur les pixels ?")
    verdict("28.23", weights_signature(NET) == signature_23, "les poids du réseau n'ont pas bougé.", "les poids du réseau ont changé.")
    with torch.no_grad():
        acts_s_23 = extract_11(NET, normalize(to_batch(small_s_23)), {"conv1": 1})
        acts_c_23 = extract_11(NET, normalize(to_batch(small_c_23)), {"conv1": 1})
        start_style_23 = 10.0 * float(((gram_matrix_18(acts_c_23["conv1"]) - gram_matrix_18(acts_s_23["conv1"])) ** 2).sum())
    verdict("28.23", hist_23["style"][0] > start_style_23,
            "la loss de style de départ additionne bien plusieurs couches, pondérées par beta.",
            "la loss de style de départ est trop petite : additionne les couches de style, multipliées par beta.")
    noise_23, _ = style_transfer_23(NET, small_c_23, small_s_23, ["fire5"], ["conv1"], 1.0, 10.0, 0.0, steps=0, init="noise", seed=3)
    same("28.23", "départ init=\"noise\", seed=3, sans aucun pas", np.asarray(noise_23),
         to_image(torch.rand((1,) + small_c_23.shape, generator=torch.Generator().manual_seed(3))), rtol=1e-6, atol=1e-7)'''

EXPERIMENT_23 = r'''with checked("28.23"):
    t0_23 = time.perf_counter()
    styled_23 = {}
    for name in ("starry_night", "great_wave", "composition_vii"):
        styled_23[name], _ = style_transfer_23(NET, CONTENT_C, wb.datasets.load_style_image(name, SIZE), CONTENT_LAYERS,
                                               STYLE_LAYERS, ALPHA, BETA, TV, steps=STEPS_C)
    print(f"3 transferts de style en {fr(time.perf_counter() - t0_23, 1)} s")
    show_images([CONTENT_C] + [wb.datasets.load_style_image(n, SIZE) for n in styled_23] + [CONTENT_C] + list(styled_23.values()),
                ["contenu", "Van Gogh", "Hokusai", "Kandinsky", "contenu", "→ Van Gogh", "→ Hokusai", "→ Kandinsky"], cols=4, size=2.8)'''

TODO_22 = r'''def reconstruct_22(net, target, layer_names, kind="content", steps=150, lr=0.05, seed=0):
    """An image rebuilt from noise so that its activations (kind="content": sum over layer_names of
    mean((A - A_target)**2)) or its Gram matrices (kind="style": sum of sum((G - G_target)**2), gram_matrix_18) match
    those of `target` (3, H, W). Start: torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed)); Adam on
    the pixels, clamped to [0, 1]. Returns (3, H, W), float32."""
    raise NotImplementedError("reconstruct_22")'''

SOLUTION_22 = r'''def reconstruct_22(net, target, layer_names, kind="content", steps=150, lr=0.05, seed=0):
    """An image rebuilt from noise so that its activations (kind="content": sum over layer_names of
    mean((A - A_target)**2)) or its Gram matrices (kind="style": sum of sum((G - G_target)**2), gram_matrix_18) match
    those of `target` (3, H, W). Start: torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed)); Adam on
    the pixels, clamped to [0, 1]. Returns (3, H, W), float32."""
    if kind not in ("content", "style"):
        raise ValueError("kind must be 'content' or 'style'")
    layers = {n: LAYERS[n] for n in layer_names}
    t = to_batch(target)
    with torch.no_grad():
        acts_t = extract_11(net, normalize(t), layers)
        goals = {n: (acts_t[n] if kind == "content" else gram_matrix_18(acts_t[n])) for n in layer_names}
    x = torch.rand(t.shape, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        acts = extract_11(net, normalize(x), layers)
        if kind == "content":
            loss = sum(F.mse_loss(acts[n], goals[n]) for n in layer_names)
        else:
            loss = sum(((gram_matrix_18(acts[n]) - goals[n]) ** 2).sum() for n in layer_names)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return to_image(x)'''

CHECK_22 = r'''with checked("28.22"):
    need("28.22", "extract_11", "gram_matrix_18")
    tiny_22 = wb.datasets.load_sample_image("flower", 48)
    rec_22 = reconstruct_22(NET, tiny_22, ["conv1"], "content", steps=60)
    if not returned("28.22", "reconstruct_22", rec_22):
        raise StopCheck
    rec_22 = np.asarray(rec_22)
    start_22 = to_image(torch.rand((1,) + tiny_22.shape, generator=torch.Generator().manual_seed(0)))
    err0_22, err1_22 = float(((start_22 - tiny_22) ** 2).mean()), float(((rec_22 - tiny_22) ** 2).mean())
    verdict("28.22", rec_22.shape == tiny_22.shape and err1_22 < 0.3 * err0_22,
            f"depuis le bruit, la couche conv1 rapproche l'image de la photo (erreur pixel {fr(err0_22, 3)} → {fr(err1_22, 3)}).",
            f"l'image ne se rapproche pas de la photo ({fr(err0_22, 3)} → {fr(err1_22, 3)}) : départ, loss de contenu, descente ?")
    sty_22 = np.asarray(reconstruct_22(NET, tiny_22, ["conv1"], "style", steps=30))
    with torch.no_grad():
        g_t = gram_matrix_18(extract_11(NET, normalize(to_batch(tiny_22)), {"c": 1})["c"])
        d0_22 = float(((gram_matrix_18(extract_11(NET, normalize(to_batch(start_22)), {"c": 1})["c"]) - g_t) ** 2).sum())
        d1_22 = float(((gram_matrix_18(extract_11(NET, normalize(to_batch(sty_22)), {"c": 1})["c"]) - g_t) ** 2).sum())
    verdict("28.22", d1_22 < 0.5 * d0_22, f"kind=\"style\" rapproche les matrices de Gram ({fr(d0_22, 2)} → {fr(d1_22, 2)}).",
            f"kind=\"style\" ne rapproche pas les matrices de Gram ({fr(d0_22, 2)} → {fr(d1_22, 2)}).")'''

EXPERIMENT_22 = r'''with checked("28.22"):
    names_22 = ["conv1", "fire3", "fire5", "fire7", "fire9"]
    steps_22 = 120 if FAST_MODE else 400
    t0_22 = time.perf_counter()
    content_rows_22 = [reconstruct_22(NET, CONTENT_C, [n], "content", steps_22) for n in names_22]
    style_rows_22 = [reconstruct_22(NET, STYLE_C, [n], "style", steps_22) for n in names_22]
    cumul_rows_22 = [reconstruct_22(NET, STYLE_C, names_22[:k + 1], "style", steps_22) for k in range(len(names_22))]
    print(f"15 reconstructions en {fr(time.perf_counter() - t0_22, 1)} s")
    show_images(content_rows_22 + style_rows_22 + cumul_rows_22,
                [f"contenu {n}" for n in names_22] + [f"style {n}" for n in names_22] + [f"style jusqu'à {n}" for n in names_22],
                cols=5, size=2.4)'''

GIVEN_19 = r'''# Reconstructions from noise (the solution of 28.22, tools/chapters/history_ch28.py), for three seeds: the photo
# CONTENT_C rebuilt from the activations of conv1 or of fire7 alone, and the painting STYLE_C rebuilt from the Gram
# matrices of all of STYLE_LAYERS. Measured: the mean squared error to the photo, pixel by pixel (images in [0, 1]),
# and the correlation, pixel by pixel, between the style reconstruction and the photo (both in grey levels).
RECON_19 = unpack("{RECON19}")   # packed, so that the numbers stay hidden until the experiment cell'''

TODO_19 = r'''ratio_19 = ...     # a) the pixel error of the fire7 reconstruction divided by that of the conv1 reconstruction:
                   #    "A" below 1.5, "B" from 1.5 to 5, "C" above 5
corr_19 = ...      # b) True or False: the correlation between the style reconstruction and the photo stays below 0.1 in
                   #    absolute value, for the three seeds'''

SOLUTION_19 = r'''ratio_19, corr_19 = "{RATIO}", {CORR}'''

EXPERIMENT_19 = r'''print(f"{'graine':>6s} {'erreur conv1':>13s} {'erreur fire7':>13s} {'rapport':>8s} {'corrélation (style)':>20s}")
for run_19 in RECON_19["runs"]:
    print(f"{run_19['seed']:6d} {fr(run_19['mse_conv1'], 4):>13s} {fr(run_19['mse_fire7'], 4):>13s} "
          f"{fr(run_19['mse_fire7'] / run_19['mse_conv1'], 1):>8s} {fr(run_19['corr_style'], 3):>20s}")
wb.check("28.19a", ratio_19)
wb.check("28.19b", corr_19)'''

GIVEN_20 = r'''# Three real style transfers (the solution of 28.23, CONTENT_C, STYLE_C, 150 steps), under shuffled letters: the
# three WEIGHTED terms (alpha * L_c, beta * L_s, tv * L_tv) every 5 steps. One run has the settings of part C; the
# others change ONE thing each.
CURVES_20 = unpack("{CURVES20}")
fig_20, axes_20 = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax_20, (letter_20, run_20) in zip(axes_20, sorted(CURVES_20["runs"].items())):
    steps_20 = [5 * k for k in range(len(run_20["content"]))]
    for key_20, label_20 in (("content", "contenu (α·L_c)"), ("style", "style (β·L_s)"), ("tv", "variation totale (λ·L_TV)")):
        ax_20.plot(steps_20, run_20[key_20], label=label_20)
    ax_20.set_yscale("log")
    ax_20.set_title(f"essai {letter_20}")
    ax_20.set_xlabel("pas")
axes_20[0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''

TODO_20 = r'''noise_20 = ...      # a) the letter of the run that starts from a noise (not from the photo)
heavy_20 = ...      # b) the letter of the run whose weighted style term, at the end, is still more than 10 times its
                    #    weighted content term
first_20 = ...      # c) run A: the weighted style term at step 0, divided by its value at the end (1 decimal)'''

SOLUTION_20 = r'''noise_20, heavy_20, first_20 = "{NOISE}", "{HEAVY}", {FIRST}'''

CHECK_20 = r'''wb.check("28.20a", noise_20)
wb.check("28.20b", heavy_20)
wb.check("28.20c", first_20)'''

GIVEN_21 = r'''# The script to debug (do not change this cell: copy the function into the next one to fix it)
def style_transfer_buggy_21(net, content, style, steps=60, lr=0.05):
    def features(x, index):                                                       # A
        return net[:index + 1](x)                                                 # B
    def gram(A):                                                                  # C
        flat = A.reshape(A.shape[1], -1)                                          # D
        return flat @ flat.T                                                      # E
    c, s = to_batch(content), to_batch(style)                                     # F
    with torch.no_grad():                                                         # G
        target_c = features(c, LAYERS["fire6"])                                   # H
        targets_s = {n: gram(features(s, LAYERS[n])) for n in STYLE_LAYERS}      # I
    x = torch.full_like(c, 0.5).requires_grad_(True)                              # J
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)                         # K
    first = None
    for step in range(steps):                                                     # L
        l_c = F.mse_loss(features(x, LAYERS["fire6"]), target_c)                  # M
        l_s = sum(((gram(features(x, LAYERS[n])) - targets_s[n]) ** 2).sum() for n in STYLE_LAYERS)   # N
        l_tv = ((x[..., 1:, :] - x[..., :-1, :]) ** 2).sum() + ((x[..., 1:] - x[..., :-1]) ** 2).sum()  # O
        if first is None:
            first = (ALPHA * l_c.item(), BETA * l_s.item())
        optimizer.zero_grad()                                                     # P
        (ALPHA * l_c + BETA * l_s + TV * l_tv).backward()                         # Q
        optimizer.step()                                                          # R
        with torch.no_grad():
            x.clamp_(-3, 3)                                                       # S
    return x.detach()[0].numpy(), first


def diagnose_21(fn, steps=25):
    """Four diagnoses in a row, on small images; each one stops the chain when it fails (one bug can hide another)."""
    c, s = wb.datasets.load_sample_image("china", 64), wb.datasets.load_style_image("starry_night", 64)
    seen = []
    handle = NET[0].register_forward_pre_hook(lambda m, inputs: seen.append(float(inputs[0].detach().mean())))
    try:
        img, first = fn(NET, c, s, steps=steps)
    finally:
        handle.remove()
    img = np.asarray(img)
    moved = float(np.abs(img - 0.5).max())
    print(f"1. les pixels bougent-ils ? écart maximal à l'image grise de départ : {fr(moved, 4)}")
    if moved < 1e-6:
        print("   ❌ l'image n'a pas changé : qu'est-ce que l'optimiseur met à jour ?")
        return False
    print(f"2. ce que reçoit la première couche : moyenne des pixels {fr(min(seen), 3)} à {fr(max(seen), 3)}")
    if min(seen) > 0.2:
        print("   ❌ des pixels autour de 0,45 : le réseau attend des images normalisées comme celles d'ImageNet.")
        return False
    ratio = first[1] / max(first[0], 1e-12)
    print(f"3. au premier pas, terme de style / terme de contenu = {ratio:.3g}")
    if ratio > 1e6:
        print("   ❌ le style écrase tout : compare l'échelle des matrices de Gram à celle de la fiche.")
        return False
    print(f"4. plage des pixels à la fin : {fr(img.min(), 3)} à {fr(img.max(), 3)}")
    if img.min() < 0 or img.max() > 1:
        print("   ❌ des pixels hors de [0, 1] : ils ne s'affichent pas, et le réseau n'en a jamais vu de tels.")
        return False
    print("✅ les quatre diagnostics passent.")
    return True


_ = diagnose_21(style_transfer_buggy_21)'''

TODO_21 = r'''faulty_21 = ...   # the letters of the faulty lines, e.g. "AB"


def style_transfer_fixed_21(net, content, style, steps=60, lr=0.05):
    """The script of style_transfer_buggy_21, fixed (same signature, same returns, same gray start)."""
    raise NotImplementedError("style_transfer_fixed_21")'''

SOLUTION_21 = r'''faulty_21 = "BEKS"


def style_transfer_fixed_21(net, content, style, steps=60, lr=0.05):
    """The script of style_transfer_buggy_21, fixed (same signature, same returns, same gray start)."""
    def features(x, index):                                                       # A
        return net[:index + 1](normalize(x))                                      # B: the ImageNet normalisation
    def gram(A):                                                                  # C
        flat = A.reshape(A.shape[1], -1)                                          # D
        return flat @ flat.T / flat.numel()                                       # E: divided by C * H * W
    c, s = to_batch(content), to_batch(style)                                     # F
    with torch.no_grad():                                                         # G
        target_c = features(c, LAYERS["fire6"])                                   # H
        targets_s = {n: gram(features(s, LAYERS[n])) for n in STYLE_LAYERS}      # I
    x = torch.full_like(c, 0.5).requires_grad_(True)                              # J
    optimizer = torch.optim.Adam([x], lr=lr)                                      # K: the pixels, not the weights
    first = None
    for step in range(steps):                                                     # L
        l_c = F.mse_loss(features(x, LAYERS["fire6"]), target_c)                  # M
        l_s = sum(((gram(features(x, LAYERS[n])) - targets_s[n]) ** 2).sum() for n in STYLE_LAYERS)   # N
        l_tv = ((x[..., 1:, :] - x[..., :-1, :]) ** 2).sum() + ((x[..., 1:] - x[..., :-1]) ** 2).sum()  # O
        if first is None:
            first = (ALPHA * l_c.item(), BETA * l_s.item())
        optimizer.zero_grad()                                                     # P
        (ALPHA * l_c + BETA * l_s + TV * l_tv).backward()                         # Q
        optimizer.step()                                                          # R
        with torch.no_grad():
            x.clamp_(0, 1)                                                        # S: the range of the pixels
    return x.detach()[0].numpy(), first'''

CHECK_21 = r'''wb.check("28.21", faulty_21)
with checked("28.21"):
    ok_21 = diagnose_21(style_transfer_fixed_21)
    verdict("28.21", ok_21, "ta version corrigée passe les quatre diagnostics.", "ta version corrigée échoue encore à un diagnostic (ci-dessus).")
    if ok_21:
        img_21, _ = style_transfer_fixed_21(NET, CONTENT_C, STYLE_C, steps=STEPS_C)
        show_images([CONTENT_C, STYLE_C, np.clip(img_21, 0, 1)], ["contenu", "style", "corrigé, départ gris"], size=3)'''

GIVEN_24 = r'''def measure_24(img, content=None, style=None):
    """Two fixed yardsticks for comparing settings (not the losses that were optimised): the content distance (mean
    squared difference of the fire6 activations) to `content` and the style distance (sum over STYLE_LAYERS of the
    squared differences of the normalised Gram matrices) to `style`, as Python floats."""
    content = CONTENT_C if content is None else content
    style = STYLE_C if style is None else style
    with torch.no_grad():
        def acts(image):
            out, h = {}, normalize(to_batch(image))
            for i, module in enumerate(NET[:LAYERS["fire6"] + 1]):
                h = module(h)
                out[i] = h
            return out
        def gram(A):
            flat = A.reshape(A.shape[1], -1)
            return flat @ flat.T / flat.numel()
        a, c, s = acts(img), acts(content), acts(style)
        d_c = float(((a[LAYERS["fire6"]] - c[LAYERS["fire6"]]) ** 2).mean())
        d_s = float(sum(((gram(a[LAYERS[n]]) - gram(s[LAYERS[n]])) ** 2).sum() for n in STYLE_LAYERS))
    return d_c, d_s'''

TODO_24 = r'''settings_24 = ...   # a list of at least four dicts of keyword arguments for style_transfer_23, e.g.
                    # {"beta": 1.0, "init": "content"}: the missing ones take the values of part C (ALPHA, BETA, TV)'''

SOLUTION_24 = r'''settings_24 = [{"beta": 1.0}, {"beta": 10.0}, {"beta": 100.0}, {"beta": 10.0, "tv": 0.0},
               {"beta": 10.0, "init": "noise"}, {"beta": 100.0, "init": "noise"}]'''

CHECK_24 = r'''with checked("28.24"):
    need("28.24", "style_transfer_23")
    if not filled(settings_24):
        print("⏳ Ex 28.24 : écris tes réglages dans la cellule précédente.")
        raise StopCheck
    verdict("28.24", isinstance(settings_24, (list, tuple)) and len(settings_24) >= 4, f"{len(settings_24)} réglages à comparer.",
            "au moins quatre réglages, dans une liste de dictionnaires.")
    rows_24, images_24 = [], []
    t0_24 = time.perf_counter()
    for setting in settings_24:
        kw = {"alpha": ALPHA, "beta": BETA, "tv": TV, "steps": STEPS_C, "init": "content"}
        kw.update(setting)
        img, hist = style_transfer_23(NET, CONTENT_C, STYLE_C, CONTENT_LAYERS, STYLE_LAYERS, **kw)
        d_c, d_s = measure_24(img)
        rows_24.append((setting, d_c, d_s, high_freq_share(img)))
        images_24.append(img)
    print(f"{len(settings_24)} transferts en {fr(time.perf_counter() - t0_24, 1)} s")
    d_s0_24 = measure_24(CONTENT_C)[1]
    print(f"repère : la photo elle-même a une distance de style de {fr(d_s0_24, 1)}")
    for setting, d_c, d_s, hf in rows_24:
        print(f"{str(setting):45s} contenu {fr(d_c, 1):>8s}   style {fr(d_s, 1):>9s}   hautes fréquences {pct(hf, 0)}")
    show_images(images_24, [str(s).replace("'", "")[:28] for s in settings_24], cols=min(4, len(images_24)), size=2.8)'''

TODO_25 = r'''def channel_stats_25(A):
    """(mean, std) of each channel of activations A (N, C, H, W) over the H x W positions: two tensors (N, C), with
    std = sqrt(var + 1e-5), var the population variance (unbiased=False)."""
    raise NotImplementedError("channel_stats_25")


def stats_loss_25(A, target_mean, target_std):
    """The AdaIN-style statistics loss: sum of the squared differences between the channel means of A and target_mean,
    plus those between the channel standard deviations and target_std (sums over the N x C entries)."""
    raise NotImplementedError("stats_loss_25")'''

SOLUTION_25 = r'''def channel_stats_25(A):
    """(mean, std) of each channel of activations A (N, C, H, W) over the H x W positions: two tensors (N, C), with
    std = sqrt(var + 1e-5), var the population variance (unbiased=False)."""
    flat = A.reshape(A.shape[0], A.shape[1], -1)
    return flat.mean(dim=2), torch.sqrt(flat.var(dim=2, unbiased=False) + 1e-5)


def stats_loss_25(A, target_mean, target_std):
    """The AdaIN-style statistics loss: sum of the squared differences between the channel means of A and target_mean,
    plus those between the channel standard deviations and target_std (sums over the N x C entries)."""
    mean, std = channel_stats_25(A)
    return ((mean - target_mean) ** 2).sum() + ((std - target_std) ** 2).sum()'''

GIVEN_25 = r'''def transfer_25(style_term, beta, steps=STEPS_C, lr=0.05):
    """The style transfer of 28.23 (from the photo, your extract_11 and tv_loss_18), with a style term of your choice:
    style_term(acts_x, acts_s) -> a 0-d tensor, acts_x and acts_s the dicts of activations at STYLE_LAYERS."""
    layers = {n: LAYERS[n] for n in set(CONTENT_LAYERS) | set(STYLE_LAYERS)}
    c = to_batch(CONTENT_C)
    with torch.no_grad():
        acts_c = extract_11(NET, normalize(c), layers)
        acts_s = extract_11(NET, normalize(to_batch(STYLE_C)), {n: LAYERS[n] for n in STYLE_LAYERS})
    x = c.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        acts = extract_11(NET, normalize(x), layers)
        loss = (ALPHA * sum(F.mse_loss(acts[n], acts_c[n]) for n in CONTENT_LAYERS)
                + beta * style_term({n: acts[n] for n in STYLE_LAYERS}, acts_s) + TV * tv_loss_18(x))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return to_image(x)'''

CHECK_25 = r'''with checked("28.25"):
    A_25 = torch.tensor(np.random.default_rng(25).random((2, 3, 4, 5)))
    stats_25 = channel_stats_25(A_25)
    if not returned("28.25", "channel_stats_25", stats_25):
        raise StopCheck
    flat_25 = A_25.numpy().reshape(2, 3, -1)
    same("28.25", "channel_stats_25 : moyennes", stats_25[0], flat_25.mean(axis=2))
    same("28.25", "channel_stats_25 : écarts-types", stats_25[1], np.sqrt(flat_25.var(axis=2) + 1e-5))
    same("28.25", "stats_loss_25 contre la cible (0, 1)", float(stats_loss_25(A_25, torch.zeros(2, 3), torch.ones(2, 3))),
         float((flat_25.mean(axis=2) ** 2).sum() + ((np.sqrt(flat_25.var(axis=2) + 1e-5) - 1) ** 2).sum()))'''

EXPERIMENT_25 = r'''with checked("28.25"):
    need("28.25", "extract_11", "gram_matrix_18", "tv_loss_18")
    def gram_term_25(acts_x, acts_s):
        return sum(((gram_matrix_18(acts_x[n]) - gram_matrix_18(acts_s[n])) ** 2).sum() for n in acts_x)
    def stats_term_25(acts_x, acts_s):
        return sum(stats_loss_25(acts_x[n], *channel_stats_25(acts_s[n])) for n in acts_x)
    results_25 = {}
    for name, term, beta in (("Gram", gram_term_25, BETA), ("moyennes et écarts-types", stats_term_25, 1.0)):
        t0 = time.perf_counter()
        img = transfer_25(term, beta)
        results_25[name] = (img, time.perf_counter() - t0) + measure_24(img)
    print(f"repère : la photo a une distance de style (Gram) de {fr(measure_24(CONTENT_C)[1], 1)}")
    for name, (img, seconds, d_c, d_s) in results_25.items():
        print(f"{name:>25s} : {fr(seconds, 1)} s, distance de contenu {fr(d_c, 1)}, distance de style (Gram) {fr(d_s, 1)}")
    show_images([STYLE_C] + [r[0] for r in results_25.values()], ["style"] + list(results_25), size=3)'''

TODO_26 = r'''def write_gallery_26(folder, entries):
    """Write a reproducible gallery into `folder` (a Path): one PNG per entry and a README.md.

    entries: a list of dicts {"file": "name.png", "image": (3, H, W) in [0, 1], "content": name of a photo,
    "style": name of a painting, "params": dict of the settings}. The README gives, for each image: its file name (as
    a Markdown image), the credit lines of its photo and of its painting (wb.datasets.image_credit), and every setting
    of its "params" as "key: value"; then, once, the network and its weights ("SqueezeNet 1.1, IMAGENET1K_V1"), the
    versions of torch and torchvision, the date, and a sentence that says the images are generated."""
    raise NotImplementedError("write_gallery_26")'''

SOLUTION_26 = r'''def write_gallery_26(folder, entries):
    """Write a reproducible gallery into `folder` (a Path): one PNG per entry and a README.md.

    entries: a list of dicts {"file": "name.png", "image": (3, H, W) in [0, 1], "content": name of a photo,
    "style": name of a painting, "params": dict of the settings}. The README gives, for each image: its file name (as
    a Markdown image), the credit lines of its photo and of its painting (wb.datasets.image_credit), and every setting
    of its "params" as "key: value"; then, once, the network and its weights ("SqueezeNet 1.1, IMAGENET1K_V1"), the
    versions of torch and torchvision, the date, and a sentence that says the images are generated."""
    import datetime

    import torchvision
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["# Galerie de transferts de style", "",
             "Images **générées** par transfert de style neuronal (L. Gatys et coll., 2015) : ce ne sont ni des photos ni "
             "des œuvres des artistes cités.", ""]
    for entry in entries:
        plt.imsave(folder / entry["file"], np.clip(np.asarray(entry["image"]).transpose(1, 2, 0), 0, 1))
        lines += [f"## {entry['file']}", "", f"![{entry['file']}]({entry['file']})", "",
                  f"- Contenu : {wb.datasets.image_credit(entry['content'])}",
                  f"- Style : {wb.datasets.image_credit(entry['style'])}",
                  "- Réglages : " + ", ".join(f"{k}: {v}" for k, v in entry["params"].items()), ""]
    lines += ["## Reproduire", "", "- Réseau : SqueezeNet 1.1, poids IMAGENET1K_V1 (torchvision), gelé",
              f"- Versions : torch {torch.__version__}, torchvision {torchvision.__version__}",
              f"- Date : {datetime.date.today().isoformat()}", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")'''

GIVEN_26 = r'''def validate_gallery_26(folder, entries):
    """Check a gallery written by write_gallery_26: files, credits, settings, reproducibility, no protected work."""
    folder = Path(folder)
    readme = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").exists() else ""
    problems = []
    if not readme:
        problems.append("pas de README.md")
    for entry in entries:
        if not (folder / entry["file"]).exists():
            problems.append(f"{entry['file']} absent du dossier")
        if entry["file"] not in readme:
            problems.append(f"{entry['file']} n'est pas cité dans le README")
        for name in (entry["content"], entry["style"]):
            if wb.datasets.image_credit(name) not in readme:
                problems.append(f"le crédit de {name!r} manque (wb.datasets.image_credit)")
        for key, value in entry["params"].items():
            if f"{key}: {value}" not in readme:
                problems.append(f"le réglage « {key}: {value} » manque")
    for word, why in (("SqueezeNet", "le réseau"), ("IMAGENET1K_V1", "les poids"), ("torchvision", "les versions"),
                      ("généré", "la mention « images générées »")):
        if word.lower() not in readme.lower():
            problems.append(f"{why} : « {word} » n'apparaît pas")
    if not re.search(r"20\d\d-\d\d-\d\d", readme):
        problems.append("pas de date au format AAAA-MM-JJ")
    for artist in ("Picasso", "Hopper"):
        if artist in readme:
            problems.append(f"{artist} : une œuvre encore protégée dans l'Union européenne")
    return problems'''

CHECK_26 = r'''with checked("28.26"):
    need("28.26", "style_transfer_23")
    entries_26 = []
    for style_name in ("starry_night", "great_wave"):
        params = {"content_layers": CONTENT_LAYERS, "style_layers": STYLE_LAYERS, "alpha": ALPHA, "beta": BETA, "tv": TV,
                  "steps": 60, "lr": 0.05, "init": "content", "size": 96}
        img, _ = style_transfer_23(NET, wb.datasets.load_sample_image("flower", 96), wb.datasets.load_style_image(style_name, 96),
                                   CONTENT_LAYERS, STYLE_LAYERS, ALPHA, BETA, TV, steps=60)
        entries_26.append({"file": f"fleur_{style_name}.png", "image": img, "content": "flower", "style": style_name,
                           "params": params})
    folder_26 = WORK_DIR / "galerie"
    out_26 = write_gallery_26(folder_26, entries_26)
    problems_26 = validate_gallery_26(folder_26, entries_26)
    verdict("28.26", not problems_26, f"la galerie est complète ({folder_26}).", " ; ".join(problems_26[:4]))
    if not problems_26:
        print((folder_26 / "README.md").read_text(encoding="utf-8")[:1500])'''

GIVEN_29 = r'''# The challenge: a photo and a painting you have not tuned on, a fixed yardstick, a time limit
CONTENT_29 = wb.datasets.load_sample_image("flower", SIZE)
STYLE_29 = wb.datasets.load_style_image("great_wave", SIZE)
LIMIT_SECONDS_29 = 120
CONTENT_GOAL_29 = {GOAL}            # the content distance (measure_24) must stay below this (a reference + 15 %)


def evaluate_29(make):
    """Run make(content, style) -> image (3, H, W) in [0, 1] once, timed, and print the three criteria."""
    start = time.perf_counter()
    img = np.asarray(make(CONTENT_29, STYLE_29), dtype=np.float32)
    seconds = time.perf_counter() - start
    d_c, d_s = measure_24(img, CONTENT_29, STYLE_29)
    d_s0 = measure_24(CONTENT_29, CONTENT_29, STYLE_29)[1]
    ok = (img.shape == CONTENT_29.shape and 0 <= img.min() and img.max() <= 1, seconds <= LIMIT_SECONDS_29,
          d_s <= d_s0 / 10, d_c <= CONTENT_GOAL_29)
    print(f"temps {fr(seconds, 1)} s (≤ {LIMIT_SECONDS_29}) ; distance de style {fr(d_s, 1)} pour {fr(d_s0, 1)} au départ "
          f"(÷ {fr(d_s0 / max(d_s, 1e-9), 1)}, il faut ÷ 10) ; distance de contenu {fr(d_c, 1)} (≤ {fr(CONTENT_GOAL_29, 1)})")
    show_images([CONTENT_29, STYLE_29, img], ["contenu", "style", "ton image"], size=3)
    return all(ok)'''

TODO_29 = r'''def make_29(content, style):
    """Your style transfer for the challenge: (3, H, W) arrays in [0, 1] -> an image (3, H, W) in [0, 1]."""
    raise NotImplementedError("make_29")'''

SOLUTION_29 = r'''def make_29(content, style):
    """Your style transfer for the challenge: (3, H, W) arrays in [0, 1] -> an image (3, H, W) in [0, 1]."""
    # a small style weight (the style only has to be divided by 10), small steps, more of them: the content stays close
    img, _ = style_transfer_23(NET, content, style, ["fire6"], STYLE_LAYERS, 1.0, 0.05, 0.05, steps=300, lr=0.02)
    return img'''

CHECK_29 = r'''with checked("28.29"):
    need("28.29", "make_29", "measure_24")
    passed_29 = evaluate_29(make_29)
    verdict("28.29", passed_29, "défi réussi : moins de 2 minutes, style divisé par 10 ou plus, contenu reconnaissable.",
            "pas encore : regarde le critère qui manque ci-dessus.")'''

# --- Part D -----------------------------------------------------------------------------------------------------------
GIVEN_D = r'''VERNE = wb.datasets.load_verne().replace("_", "").replace("--", "—")   # without the italics marks, real dashes
EPOCHS_27 = 12 if FAST_MODE else 40
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def build_vocab(*args, **kwargs):           # your chapter 22 library (mylearn.sequences), called through these names
    return mylearn.sequences.build_vocab(*args, **kwargs)


def encode(*args, **kwargs):
    return mylearn.sequences.encode(*args, **kwargs)


def decode_words(ids, itos):
    """Ids -> text: tokens joined by spaces, no space before , . ; : ! ? ) or after an apostrophe or ( ."""
    text = " ".join(itos[i] for i in ids)
    return re.sub(r"\s+([,.;:!?)»])|([’'(«])\s+", lambda m: m.group(1) or m.group(2), text)


def make_sequence_pairs(*args, **kwargs):
    return mylearn.sequences.make_sequence_pairs(*args, **kwargs)


def sample_from_logits(*args, **kwargs):
    return mylearn.sequences.sample_from_logits(*args, **kwargs)


def make_loader(X, Y, batch_size, seed):
    """Shuffled mini-batches of (X, Y), the same order for the same seed."""
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed))


class WordLM2(nn.Module):
    """The book's model (§28.5): an embedding, TWO layers of LSTM of 128 cells, one shared Linear at every step;
    ids (N, T) -> (logits (N, T, V), state)."""

    def __init__(self, vocab_size, emb_dim=64, hidden=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, ids, state=None):
        output, state = self.lstm(self.emb(ids), state)
        return self.head(output), state


print(f"Verne : {len(VERNE)} caractères")'''

TODO_27 = r'''# Copy here your functions of 24.30 (renamed _27), then adapt the tokenizer to French.
def tokenize_words_27(text):
    """The text in lower case, cut into tokens: words = runs of LETTERS of any alphabet (accented letters included:
    the regex class [^\\W\\d_]), an apostrophe ' or ’ that follows letters staying glued to them ("l'",
    "qu'", "aujourd'"), and one token for every other character that is not a space (punctuation, digits)."""
    raise NotImplementedError("tokenize_words_27")


def make_vocab_27(tokens, max_size):
    """(stoi, itos): build_vocab with the unknown token "<unk>" (id 0), at most max_size entries."""
    raise NotImplementedError("make_vocab_27")


def split_pairs_27(ids, seq_len, val_share=0.1):
    """(X_tr, Y_tr, X_val, Y_val), int64 tensors: the last int(len(ids) * val_share) ids for the validation (the END
    of the novel), the others for the training, each part cut by make_sequence_pairs(part, seq_len)."""
    raise NotImplementedError("split_pairs_27")


def train_lm_27(model, X, Y, epochs, batch_size=64, lr=3e-3, seed=0, on_epoch=None):
    """Train `model` on (X, Y) as train_lm_30: Adam(lr), make_loader(X, Y, batch_size, seed), cross-entropy over every
    position, gradients clipped to a norm of 1; after each epoch, call on_epoch(epoch, model) if given (epoch from 1).
    Return the mean training loss per token of each epoch (a list of floats)."""
    raise NotImplementedError("train_lm_27")


def mean_loss_27(model, X, Y, batch_size=256):
    """The mean cross-entropy per token of `model` on (X, Y), in nats (eval mode, no graph), a Python float."""
    raise NotImplementedError("mean_loss_27")'''

SOLUTION_27 = r'''# Copy here your functions of 24.30 (renamed _27), then adapt the tokenizer to French.
WORD_RE_27 = re.compile(r"[^\W\d_]+['’]?|\S")


def tokenize_words_27(text):
    """The text in lower case, cut into tokens: words = runs of LETTERS of any alphabet (accented letters included:
    the regex class [^\\W\\d_]), an apostrophe ' or ’ that follows letters staying glued to them ("l'",
    "qu'", "aujourd'"), and one token for every other character that is not a space (punctuation, digits)."""
    return WORD_RE_27.findall(text.lower())


def make_vocab_27(tokens, max_size):
    """(stoi, itos): build_vocab with the unknown token "<unk>" (id 0), at most max_size entries."""
    return build_vocab(tokens, max_size=max_size, unk_token="<unk>")


def split_pairs_27(ids, seq_len, val_share=0.1):
    """(X_tr, Y_tr, X_val, Y_val), int64 tensors: the last int(len(ids) * val_share) ids for the validation (the END
    of the novel), the others for the training, each part cut by make_sequence_pairs(part, seq_len)."""
    ids = np.asarray(ids)
    n_val = int(len(ids) * val_share)
    out = []
    for part in (ids[:len(ids) - n_val], ids[len(ids) - n_val:]):
        X, Y = make_sequence_pairs(part, seq_len)
        out += [torch.as_tensor(X, dtype=torch.int64), torch.as_tensor(Y, dtype=torch.int64)]
    return tuple(out)


def train_lm_27(model, X, Y, epochs, batch_size=64, lr=3e-3, seed=0, on_epoch=None):
    """Train `model` on (X, Y) as train_lm_30: Adam(lr), make_loader(X, Y, batch_size, seed), cross-entropy over every
    position, gradients clipped to a norm of 1; after each epoch, call on_epoch(epoch, model) if given (epoch from 1).
    Return the mean training loss per token of each epoch (a list of floats)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(X, Y, batch_size, seed)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * yb.numel()
            count += yb.numel()
        losses.append(total / count)
        if on_epoch is not None:
            on_epoch(epoch, model)
    return losses


def mean_loss_27(model, X, Y, batch_size=256):
    """The mean cross-entropy per token of `model` on (X, Y), in nats (eval mode, no graph), a Python float."""
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            logits, _ = model(X[i:i + batch_size])
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y[i:i + batch_size].reshape(-1),
                                     reduction="sum").item()
    return total / Y.numel()'''

CHECK_27 = r'''with checked("28.27"):
    toks_27 = tokenize_words_27("L'été, Phileas Fogg — aujourd'hui à 8 h 45 — dit : « Partons ! »")
    if not returned("28.27", "tokenize_words_27", toks_27):
        raise StopCheck
    want_27 = ["l'", "été", ",", "phileas", "fogg", "—", "aujourd'", "hui", "à", "8", "h", "4", "5", "—", "dit", ":", "«",
               "partons", "!", "»"]
    verdict("28.27", list(toks_27) == want_27, "le tokenizer garde les lettres accentuées et colle l'apostrophe au mot.",
            f"attendu {want_27}, reçu {list(toks_27)}.")
    stoi_27, itos_27 = make_vocab_27(tokenize_words_27(VERNE), 5001)
    verdict("28.27", len(itos_27) == 5001 and itos_27[0] == "<unk>", "5 001 entrées, « <unk> » à l'id 0.",
            f"attendu 5 001 entrées et « <unk> » à l'id 0, reçu {len(itos_27)} entrées, la première {itos_27[0]!r}.")
    ids_27 = encode(tokenize_words_27(VERNE), stoi_27, unk_token="<unk>")
    X_tr_27, Y_tr_27, X_val_27, Y_val_27 = split_pairs_27(ids_27, 32)
    n_val_27 = int(len(ids_27) * 0.1)
    verdict("28.27", X_val_27.dtype == torch.int64 and bool((Y_tr_27[:, :-1] == X_tr_27[:, 1:]).all())
            and len(X_val_27) == (n_val_27 - 1) // 32
            and bool((X_val_27[0] == torch.as_tensor(np.asarray(ids_27)[len(ids_27) - n_val_27:][:32])).all()),
            "paires décalées d'un pas, validation prise à la fin du roman.", "les paires ou la séparation ne suivent pas la docstring.")
    wb.check("28.27a", len(tokenize_words_27(VERNE)), computed=True)
    wb.check("28.27b", float(np.mean(np.asarray(ids_27) == 0)), computed=True)'''

RECORD_27 = r'''ascii_27 = len(re.findall(r"[a-z]+(?:['’][a-z]+)*|[^\sa-z]", VERNE.lower()))   # the tokenizer of 24.30
wb.record("28.27a", len(tokenize_words_27(VERNE)),
          mistakes={"ton tokenizer coupe-t-il les mots accentués ? Les lettres de tout alphabet : [^\\W\\d_]": ascii_27})
wb.record("28.27b", float(np.mean(np.asarray(ids_27) == 0)), decimals=4)'''

EXPERIMENT_27 = r'''with checked("28.27"):
    if "itos_27" not in globals() or "X_val_27" not in globals():
        print("⏳ Ex 28.27 : fais d'abord passer la cellule de vérification précédente (tokenizer, vocabulaire, paires).")
        raise StopCheck
    torch.manual_seed(27)
    MODEL_27 = WordLM2(len(itos_27))
    prompt_27 = encode(tokenize_words_27("Phileas Fogg"), stoi_27, unk_token="<unk>")
    samples_27, val_27 = {}, []

    def on_epoch_27(epoch, model):
        val_27.append(mean_loss_27(model, X_val_27, Y_val_27))
        if epoch in (1, EPOCHS_27 // 2, EPOCHS_27):
            rng = np.random.default_rng(0)
            model.eval()
            with torch.inference_mode():
                logits, state = model(torch.as_tensor([prompt_27]))
                out = []
                for _ in range(40):
                    nxt = int(sample_from_logits(logits[0, -1].numpy(), temperature=0.8, rng=rng))
                    out.append(nxt)
                    logits, state = model(torch.tensor([[nxt]]), state)
            samples_27[epoch] = decode_words(list(prompt_27) + out, itos_27)

    t0_27 = time.perf_counter()
    train_27 = train_lm_27(MODEL_27, X_tr_27, Y_tr_27, EPOCHS_27, lr=1e-2, on_epoch=on_epoch_27)
    print(f"{EPOCHS_27} epochs en {fr(time.perf_counter() - t0_27, 1)} s")
    for epoch, (tr, va) in enumerate(zip(train_27, val_27), start=1):
        print(f"epoch {epoch:2d} : loss {fr(tr, 3)}, perplexité de validation {fr(math.exp(va), 1)}")
    print(f"repère : un modèle uniforme sur 5 001 mots a une perplexité de 5 001")
    for epoch, text in samples_27.items():
        print(f"\n— après {epoch} epoch(s), T = 0,8 :\n{text}")
    verdict("28.27", math.exp(val_27[-1]) < 500, f"la perplexité de validation tombe à {fr(math.exp(val_27[-1]), 1)}.",
            f"la perplexité de validation reste à {fr(math.exp(val_27[-1]), 1)} : l'entraînement a-t-il lieu ?")'''

TODO_28 = r'''def pick_28(logits, strategy, rng, temperature=1.0, k=None, p=None):
    """One token id (a Python int) from the logits (V,) of the next token:
    - "greedy": the argmax (first one on ties), no randomness;
    - "temperature": a draw from softmax(logits / temperature);
    - "top_k": a draw among the k largest logits only (softmax of logits / temperature on them, renormalised);
    - "top_p": a draw among the smallest set of the most probable tokens (softmax(logits / temperature)) whose total
      probability reaches p, renormalised.
    Each draw uses rng.choice(ids, p=probs) once (ids the kept ids in decreasing probability)."""
    raise NotImplementedError("pick_28")


def repeated_share_28(ids, n=3):
    """The share of the n-grams of `ids` that already appeared earlier in `ids` (0 when there are none)."""
    raise NotImplementedError("repeated_share_28")'''

SOLUTION_28 = r'''def pick_28(logits, strategy, rng, temperature=1.0, k=None, p=None):
    """One token id (a Python int) from the logits (V,) of the next token:
    - "greedy": the argmax (first one on ties), no randomness;
    - "temperature": a draw from softmax(logits / temperature);
    - "top_k": a draw among the k largest logits only (softmax of logits / temperature on them, renormalised);
    - "top_p": a draw among the smallest set of the most probable tokens (softmax(logits / temperature)) whose total
      probability reaches p, renormalised.
    Each draw uses rng.choice(ids, p=probs) once (ids the kept ids in decreasing probability)."""
    z = np.asarray(logits, dtype=np.float64)
    if strategy == "greedy":
        return int(np.argmax(z))
    probs = np.exp((z - z.max()) / temperature)
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")
    if strategy == "temperature":
        keep = order
    elif strategy == "top_k":
        keep = order[:k]
    elif strategy == "top_p":
        keep = order[: int(np.searchsorted(np.cumsum(probs[order]), p)) + 1]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    kept = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=kept))


def repeated_share_28(ids, n=3):
    """The share of the n-grams of `ids` that already appeared earlier in `ids` (0 when there are none)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    seen, repeated = set(), 0
    for g in grams:
        repeated += g in seen
        seen.add(g)
    return repeated / len(grams) if grams else 0.0'''

CHECK_28 = r'''with checked("28.28"):
    z_28 = np.array([2.0, 1.0, 0.5, 3.0, -1.0])
    first_28 = pick_28(z_28, "greedy", np.random.default_rng(0))
    if not returned("28.28", "pick_28", first_28):
        raise StopCheck
    verdict("28.28", first_28 == 3 and pick_28(z_28, "top_k", np.random.default_rng(1), k=1) == 3
            and pick_28(z_28, "top_p", np.random.default_rng(2), p=0.01) == 3,
            "glouton, top-k avec k = 1 et top-p avec un petit p donnent le même mot.",
            "le glouton, le top-k avec k = 1 ou le top-p avec p = 0,01 ne rendent pas l'argmax.")
    def freq_28(**kw):
        rng = np.random.default_rng(28)
        draws = [pick_28(z_28, rng=rng, **kw) for _ in range(6000)]
        return np.bincount(draws, minlength=5) / len(draws)
    probs_28 = np.exp(z_28 - z_28.max()); probs_28 /= probs_28.sum()
    f_t = freq_28(strategy="temperature")
    verdict("28.28", np.abs(f_t - probs_28).max() < 0.03, "« temperature » à T = 1 suit la softmax.",
            f"fréquences {np.round(f_t, 3)}, attendu environ {np.round(probs_28, 3)}.")
    f_k = freq_28(strategy="top_k", k=2)
    want_k = np.zeros(5); want_k[[3, 0]] = probs_28[[3, 0]] / probs_28[[3, 0]].sum()
    verdict("28.28", np.abs(f_k - want_k).max() < 0.03, "« top_k » avec k = 2 ne tire que les deux meilleurs mots.",
            f"fréquences {np.round(f_k, 3)}, attendu environ {np.round(want_k, 3)}.")
    f_p = freq_28(strategy="top_p", p=0.9)
    keep_p = np.argsort(-probs_28)[:int(np.searchsorted(np.cumsum(np.sort(probs_28)[::-1]), 0.9)) + 1]
    want_p = np.zeros(5); want_p[keep_p] = probs_28[keep_p] / probs_28[keep_p].sum()
    verdict("28.28", np.abs(f_p - want_p).max() < 0.03, f"« top_p » avec p = 0,9 garde {len(keep_p)} mots.",
            f"fréquences {np.round(f_p, 3)}, attendu environ {np.round(want_p, 3)}.")
    same("28.28", "repeated_share_28([1, 2, 3, 1, 2, 3, 1], 3)", repeated_share_28([1, 2, 3, 1, 2, 3, 1], 3), 0.4)'''

EXPERIMENT_28 = r'''with checked("28.28"):
    if globals().get("MODEL_27") is None:
        print("⏳ Ex 28.28 : entraîne d'abord le modèle de 28.27 (MODEL_27).")
        raise StopCheck
    strategies_28 = {"glouton": dict(strategy="greedy"), "T = 0,7": dict(strategy="temperature", temperature=0.7),
                     "T = 1,2": dict(strategy="temperature", temperature=1.2), "top-k 20": dict(strategy="top_k", k=20),
                     "top-p 0,9": dict(strategy="top_p", p=0.9)}
    MODEL_27.eval()
    report_28 = {}
    for name, kw in strategies_28.items():
        reps, distinct, texts = [], [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            with torch.inference_mode():
                logits, state = MODEL_27(torch.as_tensor([prompt_27]))
                out = []
                for _ in range(120):
                    nxt = pick_28(logits[0, -1].numpy(), rng=rng, **kw)
                    out.append(nxt)
                    logits, state = MODEL_27(torch.tensor([[nxt]]), state)
            reps.append(repeated_share_28(out, 3))
            distinct.append(len(set(out)) / len(out))
            texts.append(decode_words(out[:30], itos_27))
        report_28[name] = (np.mean(reps), np.mean(distinct), texts[0])
    for name, (rep, dist, text) in report_28.items():
        print(f"{name:>10s} : trigrammes répétés {pct(rep, 0):>5s}, mots distincts {pct(dist, 0):>5s}   « {text} »")'''
