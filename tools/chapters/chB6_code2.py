"""Code cells of the chapter B6 notebook, second half (end of part B: B6.21 and B6.22; part C, fairness: B6.23 to
B6.29), shared by build_chB6.py (through chB6_parts2.py) and history_chB6.py (used by Claude).

Same conventions as chB6_code.py: TOOLS_* and GIVEN_* (both notebooks), TODO_* (exercise notebook), SOLUTION_*
(solutions notebook), CHECK_* and EXPERIMENT_* (both), RECORD_* (solutions). Part B's cells reuse RESNET,
RESNET_INFO, SIZE_20, POOL_20, RIGHT_20, WRONG_20, X_20 (LOAD_20) and CNN, X_CIFAR, Y_CIFAR, show_rows (TOOLS_B);
the code given here goes through the reference extractors of wbx, never through the functions of B6.19 and B6.20.
Placeholders «NAME» are filled by chB6_parts2.py from history_chB6.json.
"""

# =====================================================================================================================
# Part B (end) · networks
# =====================================================================================================================
# --- 🐛 B6.21 ---------------------------------------------------------------------------------------------------------
GIVEN_21 = r'''# Six images of B6.20 (three right, three errors), their true classes, another class for each, and the tools of the
# exercise: the extraction by hooks (correct, as in B6.20), the chain of five diagnoses, and the display
PICK_21 = [0, 1, 2, 8, 9, 10]
X_21 = X_20[PICK_21]                                                     # (6, 3, SIZE_20, SIZE_20), prepared
TRUE_21 = POOL_20[1][np.concatenate([RIGHT_20, WRONG_20])][PICK_21]     # their true classes
PRED_21 = PRED_20[np.concatenate([RIGHT_20, WRONG_20])][PICK_21]        # the classes RESNET predicts
OTHER_21 = np.array([next(k for k in range(10) if k not in (t, p)) for t, p in zip(TRUE_21, PRED_21)])
H_21 = SIZE_20 // 32                                                     # the side of the output of layer4


def extract_21(model, layer, images, target, score_fn):
    """(A, G): float64 tensors (N, C, H, W), the output of `layer` for the prepared images and the gradient of
    score_fn(logits, target) with respect to it (two hooks, removed afterwards; the input requires grad)."""
    store = {}

    def save(module, inputs, output):
        store["A"] = output
        output.register_hook(lambda grad: store.__setitem__("G", grad))

    x = torch.as_tensor(images, dtype=torch.float32).clone().requires_grad_(True)
    handle = layer.register_forward_hook(save)
    try:
        score_fn(model(x), torch.as_tensor(np.asarray(target)).long()).backward()
    finally:
        handle.remove()
    if "A" not in store:
        raise ValueError("la couche n'est pas utilisée par la passe avant de RESNET")
    if "G" not in store:
        raise ValueError("le score ne dépend pas de la sortie de la couche (aucun gradient ne lui parvient)")
    return store["A"].detach().double(), store["G"].detach().double()


def pipeline_21(layer, score_fn, combine_fn, upsample_fn, images, target):
    """The whole Grad-CAM: extraction at `layer`, combination, resizing to SIZE_20 x SIZE_20."""
    A, G = extract_21(RESNET, layer, images, target, score_fn)
    return upsample_fn(combine_fn(A, G), SIZE_20)


def diagnose_21(layer, score_fn, combine_fn, upsample_fn):
    """Five diagnoses of the Grad-CAM, in a row: the first one that fails stops the chain (one bug can hide another).
    Returns True when the five pass."""
    total = lambda logits, t: logits.sum()               # a score that does not depend on score_fn

    def point1():
        A, _ = extract_21(RESNET, layer, X_21[:2], TRUE_21[:2], total)
        h, w = A.shape[-2:]
        if (h, w) != (H_21, H_21):
            return False, (f"les activations de ta couche font {h} × {w} sur une image de {SIZE_20} pixels (une case "
                           f"pour {SIZE_20 // h} × {SIZE_20 // w} pixels) : la carte sera très fine, et ses canaux "
                           "répondent à des bords et à des textures plutôt qu'à des objets"), \
                "quelle couche Grad-CAM vise-t-il, et pourquoi (206.6, 🧠 Q8) ?"
        A4, _ = extract_21(RESNET, RESNET.layer4, X_21[:2], TRUE_21[:2], total)
        if A.shape != A4.shape or not torch.allclose(A, A4):
            return False, (f"ta couche est en {h} × {w}, mais sa sortie n'est pas celle du dernier bloc entier de "
                           "RESNET (après sa connexion résiduelle et sa ReLU)"), "relis la liste des modules de RESNET"
        return True, f"les activations font {h} × {w} sur une image de {SIZE_20} pixels", ""

    def point2():
        _, G1 = extract_21(RESNET, layer, X_21, TRUE_21, score_fn)
        _, G2 = extract_21(RESNET, layer, X_21, OTHER_21, score_fn)
        if torch.allclose(G1, G2):
            return False, ("les gradients sont les mêmes quand on demande la vraie classe ou une autre classe, sur les "
                           f"{len(X_21)} images : la carte ne dépend pas de la classe étudiée"), \
                "de quelle classe ton score prend-il le logit ?"
        _, G_ref, _ = wbx.activations_and_gradients(RESNET, layer, X_21, TRUE_21)
        G1 = G1.numpy()
        scale = float((G1 * G_ref).sum() / (G_ref * G_ref).sum())
        if not (scale > 0 and np.allclose(G1, scale * G_ref, rtol=1e-4, atol=1e-7 * np.abs(G_ref).max())):
            return False, ("les gradients changent avec la classe demandée, mais ne sont pas ceux du logit de target[n] "
                           "de chaque image"), "que dérive exactement ton score, image par image ?"
        return True, "les gradients sont ceux du logit de la classe demandée, image par image", ""

    def point3():
        A, G = extract_21(RESNET, layer, X_21, TRUE_21, score_fn)
        cams = torch.as_tensor(combine_fn(A.clone(), G.clone())).double()
        if tuple(cams.shape) != (len(X_21), H_21, H_21):
            return False, f"combine_21 rend la forme {tuple(cams.shape)}", f"attendu ({len(X_21)}, {H_21}, {H_21})"
        low = float(cams.min())
        if low < -1e-9:
            i = int(cams.amin(dim=(1, 2)).argmin())
            return False, (f"les cartes ont des valeurs négatives (jusqu'à {fr(low, 2)} sur l'image {i}) : les régions "
                           "dont l'activation fait baisser le score de la classe y restent"), \
                "relis la formule de Grad-CAM (206.6) : que garde-t-elle de la somme pondérée ?"
        raw = np.einsum("nc,nchw->nhw", G.numpy().mean(axis=(2, 3)), A.numpy())
        want = np.maximum(raw, 0)
        peak = np.abs(want).max(axis=(1, 2), keepdims=True)
        want = np.divide(want, peak, out=np.zeros_like(want), where=peak > 0)
        if not np.allclose(cams.numpy(), want, rtol=1e-6, atol=1e-9):
            return False, (f"les cartes sont positives, mais s'écartent de la formule de 206.6 de "
                           f"{fr(float(np.abs(cams.numpy() - want).max()), 3)}"), "relis combine_21 ligne à ligne"
        return True, "les cartes sont la ReLU de la somme pondérée, chacune divisée par son maximum", ""

    def point4():
        ramp = torch.arange(H_21, dtype=torch.float64).repeat(1, H_21, 1)       # 0, 1, 2 from left to right
        big = torch.as_tensor(upsample_fn(ramp.clone(), SIZE_20)).double()
        if tuple(big.shape) != (1, SIZE_20, SIZE_20):
            return False, f"upsample_21 rend la forme {tuple(big.shape)}", f"attendu (1, {SIZE_20}, {SIZE_20})"
        want = F.interpolate(ramp[:, None], size=(SIZE_20, SIZE_20), mode="bilinear", align_corners=False)[:, 0]
        if torch.allclose(big, want, atol=1e-6):
            return True, ("agrandie, une carte de cases 0, 1, 2 vaut 0, 1 et 2 aux centres de ses cases, comme les "
                          "pixels des images"), ""
        cell = SIZE_20 // H_21
        first = float(big[0, 0, cell // 2 - 1: cell // 2 + 1].mean())          # the centre of the first column of cells
        if abs(first) > 0.01:
            return False, (f"une carte de {H_21} × {H_21} cases qui valent 0, 1, 2 de gauche à droite, agrandie, vaut "
                           f"{fr(first, 2)} au centre de la première case de l'image (pixels {cell // 2 - 1} et "
                           f"{cell // 2}), au lieu de 0 : la carte est décalée de {fr(first * cell, 0)} pixels sur "
                           "l'image à cet endroit"), \
                "à quelle position de l'image chaque case correspond-elle, et comment les images ont-elles été agrandies (wbx.model_input, B6.20) ?"
        return False, ("agrandie, une carte de cases 0, 1, 2 ne passe pas progressivement d'une case à la suivante "
                       "comme les pixels des images"), "comment les images ont-elles été agrandies (wbx.model_input, B6.20) ?"

    def point5():
        mine = torch.as_tensor(pipeline_21(layer, score_fn, combine_fn, upsample_fn, X_21, TRUE_21)).double().numpy()
        want = wbx.grad_cam(RESNET, RESNET.layer4, X_21, TRUE_21, size=SIZE_20)
        if mine.shape == want.shape and np.allclose(mine, want, rtol=1e-4, atol=1e-5):
            return True, f"les {len(X_21)} cartes sont celles de la référence (wbx.grad_cam)", ""
        gap = float(np.abs(mine - want).max()) if mine.shape == want.shape else float("nan")
        return False, f"tes cartes s'écartent de la référence de {fr(gap, 3)}", "un autre changement que les quatre défauts ?"

    points = [("la couche", point1), ("la classe étudiée", point2), ("le signe des cartes", point3),
              ("l'alignement sur l'image", point4), ("les cartes finales", point5)]
    for k, (name, run) in enumerate(points, start=1):
        try:
            ok, measured, why = run()
        except Exception as error:  # noqa: BLE001 - the learner's code: one ❌ line, then the chain stops
            ok, measured, why = False, f"une erreur {type(error).__name__} : {str(error)[:200]}", "relis la fonction en cause"
        finally:
            for module in RESNET.modules():                  # a hook left behind by an earlier run
                module._forward_hooks.clear()
        if ok:
            print(f"✅ Ex B6.21 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B6.21 : point {k}, {name} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B6.21 : point {later} : corrige d'abord le point {k}.")
            return False
    print("✅ Ex B6.21 : les cinq diagnostics passent.")
    return True


def overlay_21(image, cam):
    """An image (H, W, 3) in [0, 1] under a map (H, W) in [0, 1] (colour map jet)."""
    return np.clip(0.55 * image + 0.45 * plt.cm.jet(np.clip(cam, 0, 1))[..., :3], 0, 1)


PICS_21 = wbx.model_input(POOL_20[0][np.concatenate([RIGHT_20, WRONG_20])][PICK_21],
                          {"size": SIZE_20, "mean": (0, 0, 0), "std": (1, 1, 1)}).permute(0, 2, 3, 1).clamp(0, 1).numpy()'''

TODO_21 = r'''# The Grad-CAM of a colleague, to repair IN PLACE (same names, same signatures). They wanted the Grad-CAM of 206.6 for
# the class target[n] of each image, on RESNET, the map resized like the images; four bugs slipped in.
LAYER_21 = RESNET.layer2       # the layer whose activations Grad-CAM weighs


def score_21(logits, target):
    """The scalar to differentiate: the sum over the images of the logit of the class target[n] ((N,) int64)."""
    return logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()


def combine_21(A, G):
    """Grad-CAM maps (N, H, W) from the activations A and the gradients G, tensors (N, C, H, W): channel weights =
    spatial mean of G, ReLU of the weighted sum, each map divided by its largest absolute value (a zero map stays 0)."""
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1)
    peak = cam.abs().amax(dim=(1, 2), keepdim=True)
    return torch.where(peak > 0, cam / torch.where(peak > 0, peak, 1.0), 0.0)


def upsample_21(cam, size):
    """The maps (N, H, W) resized to (N, size, size), the way the images were resized (wbx.model_input)."""
    return F.interpolate(cam[:, None], size=(size, size), mode="bilinear", align_corners=True)[:, 0]'''

SOLUTION_21 = r'''LAYER_21 = RESNET.layer4       # 1. the last convolutional block (3 x 3 at 96 pixels), not layer2 (12 x 12)


def score_21(logits, target):
    """The scalar to differentiate: the sum over the images of the logit of the class target[n] ((N,) int64)."""
    return logits.gather(1, target.reshape(-1, 1)).sum()            # 2. the class studied, not the predicted one


def combine_21(A, G):
    """Grad-CAM maps (N, H, W) from the activations A and the gradients G, tensors (N, C, H, W): channel weights =
    spatial mean of G, ReLU of the weighted sum, each map divided by its largest absolute value (a zero map stays 0)."""
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * A).sum(dim=1))                      # 3. the ReLU, once, on the weighted sum
    peak = cam.abs().amax(dim=(1, 2), keepdim=True)
    return torch.where(peak > 0, cam / torch.where(peak > 0, peak, 1.0), 0.0)


def upsample_21(cam, size):
    """The maps (N, H, W) resized to (N, size, size), the way the images were resized (wbx.model_input)."""
    return F.interpolate(cam[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]   # 4.'''

CHECK_21 = r'''repaired_21 = False
with checked("B6.21"):
    repaired_21 = diagnose_21(LAYER_21, score_21, combine_21, upsample_21)'''

EXPERIMENT_21 = r'''# Your Grad-CAM, as it is now (it shows the symptoms before the repair), next to the reference of wbx (layer4)
try:
    mine_21 = torch.as_tensor(pipeline_21(LAYER_21, score_21, combine_21, upsample_21, X_21, TRUE_21)).double().numpy()
except Exception as error_21:  # noqa: BLE001 - the learner's code
    mine_21 = None
    print(f"⏳ Ex B6.21 : ton Grad-CAM lève {type(error_21).__name__} ({str(error_21)[:160]}) : rien à montrer.")
finally:
    for module_21 in RESNET.modules():
        module_21._forward_hooks.clear()
if mine_21 is not None and mine_21.shape == (len(X_21), SIZE_20, SIZE_20):
    ref_21 = wbx.grad_cam(RESNET, RESNET.layer4, X_21, TRUE_21, size=SIZE_20)
    rows_21 = [[PICS_21[i], overlay_21(PICS_21[i], (mine_21[i] - min(0, mine_21[i].min())) / max(np.ptp(mine_21[i]), 1e-12)),
                overlay_21(PICS_21[i], ref_21[i])] for i in range(len(X_21))]
    show_rows(rows_21, [f"{CLASSES[TRUE_21[i]]} (prédit {CLASSES[PRED_21[i]]})" for i in range(len(X_21))],
              ["image", "ton Grad-CAM " + ("réparé" if repaired_21 else "tel quel"), "référence"], [None, None, None],
              size=1.6, suptitle="Grad-CAM de la vraie classe")
    print("valeurs de tes cartes : de", fr(float(mine_21.min()), 2), "à", fr(float(mine_21.max()), 2),
          "; écart maximal à la référence :", fr(float(np.abs(mine_21 - ref_21).max()), 3))
elif mine_21 is not None:
    print(f"⏳ Ex B6.21 : ton Grad-CAM rend la forme {mine_21.shape}, attendu ({len(X_21)}, {SIZE_20}, {SIZE_20}).")'''

# --- 🔬 B6.22 ---------------------------------------------------------------------------------------------------------
GIVEN_22 = r'''# 16 test images that the small CNN classifies right, their classes, and the parts of the CNN that hold weights, from
# the output down (the order of a cascading randomisation)
RIGHT_CNN_22 = np.flatnonzero(wbx.predict(CNN, X_CIFAR).argmax(axis=1) == Y_CIFAR)[:16]
X_22 = wbx.pixels(X_CIFAR[RIGHT_CNN_22])            # (16, 3, 32, 32) in [0, 1]
T_22 = Y_CIFAR[RIGHT_CNN_22]                         # their true (and predicted) classes
NAMES_22 = wbx.cascade_layers(CNN)                  # ['fc', 'features.17', 'features.14', ..., 'features.0']
print("parties du CNN, de la sortie vers l'entrée :", NAMES_22)'''

TODO_22 = r'''def randomize_22(model, names, seed=0):
    """A frozen copy (eval() mode, requires_grad=False everywhere) of model, in which every Conv2d and Linear module
    inside the parts `names` (model.get_submodule(name)) gets new weights from its own reset_parameters(), drawn after
    torch.manual_seed(seed) inside torch.random.fork_rng(); model itself and the global random generator do not change."""
    raise NotImplementedError("randomize_22")


def sanity_22(model, explain, names, mode="cascade", seed=0, absolute=False):
    """np.ndarray (len(names), N): row k is the Spearman rank correlation, image by image, between the maps
    explain(model) and explain(copy), copy = randomize_22(model, names[:k + 1], seed) for mode="cascade" (the k + 1
    first parts), randomize_22(model, [names[k]], seed) for mode="independent" (part k alone). explain(m) returns the
    maps (N, ...) of the N images for the model m; absolute=True compares their absolute values; a constant map: nan."""
    raise NotImplementedError("sanity_22")'''

SOLUTION_22 = r'''import copy

from scipy.stats import spearmanr


def randomize_22(model, names, seed=0):
    """A frozen copy (eval() mode, requires_grad=False everywhere) of model, in which every Conv2d and Linear module
    inside the parts `names` (model.get_submodule(name)) gets new weights from its own reset_parameters(), drawn after
    torch.manual_seed(seed) inside torch.random.fork_rng(); model itself and the global random generator do not change."""
    out = copy.deepcopy(model)
    with torch.random.fork_rng():                     # the draws of the following cells do not move
        torch.manual_seed(seed)
        for name in names:
            for module in out.get_submodule(name).modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.reset_parameters()
    return out.eval().requires_grad_(False)


def sanity_22(model, explain, names, mode="cascade", seed=0, absolute=False):
    """np.ndarray (len(names), N): row k is the Spearman rank correlation, image by image, between the maps
    explain(model) and explain(copy), copy = randomize_22(model, names[:k + 1], seed) for mode="cascade" (the k + 1
    first parts), randomize_22(model, [names[k]], seed) for mode="independent" (part k alone). explain(m) returns the
    maps (N, ...) of the N images for the model m; absolute=True compares their absolute values; a constant map: nan."""
    if mode not in ("cascade", "independent"):
        raise ValueError(f"mode must be 'cascade' or 'independent', got {mode!r}")
    flat = lambda maps: np.abs(np.asarray(maps, dtype=float)) if absolute else np.asarray(maps, dtype=float)
    before = flat(explain(model))
    rows = []
    for k in range(len(names)):
        parts = names[:k + 1] if mode == "cascade" else [names[k]]
        after = flat(explain(randomize_22(model, parts, seed)))
        row = []
        for u, v in zip(before.reshape(len(before), -1), after.reshape(len(after), -1)):
            row.append(np.nan if np.ptp(u) == 0 or np.ptp(v) == 0 else float(spearmanr(u, v).statistic))
        rows.append(row)
    return np.array(rows)'''

CHECK_22 = r'''ok_22 = {}
with checked("B6.22"):
    before_22 = {k: v.clone() for k, v in CNN.state_dict().items()}
    rng_22 = torch.get_rng_state()
    copy_22 = randomize_22(CNN, ["fc", "features.17"], seed=3)
    changed_22 = [k for k, v in CNN.state_dict().items() if not torch.equal(v, before_22[k])]
    if changed_22:
        CNN.load_state_dict(before_22)                 # the CNN of the following cells stays the trained one
        CNN.eval().requires_grad_(False)
    if not returned("B6.22", "randomize_22", copy_22):
        raise StopCheck
    if copy_22 is CNN:
        print("❌ Ex B6.22 : randomize_22 rend le modèle reçu lui-même : le contrat veut une copie (copy.deepcopy).")
        raise StopCheck
    if changed_22:
        print(f"❌ Ex B6.22 : randomize_22 a modifié le modèle reçu ({changed_22[0]}…) : travaille sur une copie.")
        raise StopCheck
    verdict("B6.22", torch.equal(torch.get_rng_state(), rng_22), "le générateur aléatoire global n'a pas bougé.",
            "randomize_22 a déplacé le générateur aléatoire global : tire les poids dans torch.random.fork_rng().")
    verdict("B6.22", not copy_22.training and not any(p.requires_grad for p in copy_22.parameters()),
            "la copie est gelée, en mode eval().", "la copie doit être en mode eval() et gelée (requires_grad_(False)).")
    new_22, old_22 = copy_22.state_dict(), CNN.state_dict()
    inside_22 = [k for k in old_22 if (k.startswith("fc.") or k.startswith("features.17.")) and k.endswith(("weight", "bias"))]
    outside_22 = [k for k in old_22 if k not in inside_22]
    verdict("B6.22", all(not torch.equal(new_22[k], old_22[k]) for k in inside_22),
            "les poids et biais de fc et de features.17 ont été retirés.",
            "un poids ou un biais de fc ou de features.17 n'a pas changé : chaque Conv2d et chaque Linear des parties "
            "nommées doit reprendre son reset_parameters().")
    verdict("B6.22", all(torch.equal(new_22[k], old_22[k]) for k in outside_22),
            "les autres parties (et les batchnorms) sont intactes.",
            "une autre partie que fc et features.17 a changé (" + next((k for k in outside_22 if not torch.equal(new_22[k], old_22[k])), "") + ").")
    block_22 = randomize_22(RESNET, ["layer4.1"], seed=3).state_dict()     # a block: two convolutions, two batchnorms
    old_r_22 = RESNET.state_dict()
    bn_22 = [k for k in old_r_22 if k.startswith("layer4.1.bn")]
    verdict("B6.22", all(torch.equal(block_22[k], old_r_22[k]) for k in bn_22)
            and not torch.equal(block_22["layer4.1.conv1.weight"], old_r_22["layer4.1.conv1.weight"]),
            "dans un bloc du ResNet (layer4.1), les convolutions changent, les batchnorms restent.",
            "dans le bloc layer4.1 du ResNet, seules les Conv2d (et les Linear) doivent reprendre leur reset_parameters() : "
            "les batchnorms gardent leurs poids et leurs statistiques.")
    again_22 = randomize_22(CNN, ["fc", "features.17"], seed=3).state_dict()
    other_22 = randomize_22(CNN, ["fc", "features.17"], seed=4).state_dict()
    verdict("B6.22", all(torch.equal(again_22[k], new_22[k]) for k in new_22)
            and not all(torch.equal(other_22[k], new_22[k]) for k in inside_22),
            "la même graine redonne les mêmes poids, une autre graine d'autres poids.",
            "randomize_22 doit être reproductible (même graine, mêmes poids) et changer avec la graine.")
    ok_22["randomize"] = True

with checked("B6.22"):
    if not ok_22:
        raise StopCheck
    probe_22 = lambda m: wbx.input_gradients(m, X_22[:4], T_22[:4])     # a cheap map: the gradient of the logit
    names_22 = ["fc", "features.17", "features.14"]
    for mode_22 in ("cascade", "independent"):
        for absolute_22 in (False, True):
            got_22 = sanity_22(CNN, probe_22, names_22, mode=mode_22, seed=5, absolute=absolute_22)
            if not returned("B6.22", "sanity_22", got_22):
                raise StopCheck
            ref_maps_22 = probe_22(CNN)
            want_22 = np.array([wbx.spearman_per_image(ref_maps_22, probe_22(randomize_22(
                CNN, names_22[:k + 1] if mode_22 == "cascade" else [names_22[k]], seed=5)), absolute=absolute_22)
                for k in range(len(names_22))])
            arr_22 = np.asarray(got_22, dtype=float)
            if arr_22.shape != want_22.shape:
                print(f"❌ Ex B6.22 : sanity_22 doit rendre un tableau (3, 4) (une ligne par partie, une colonne par image), "
                      f"reçu la forme {arr_22.shape}.")
                raise StopCheck
            if not same("B6.22", f"sanity_22, mode {mode_22}, absolute={absolute_22}", arr_22, want_22, rtol=1e-9, atol=1e-9):
                raise StopCheck
    ok_22["sanity"] = True'''

EXPERIMENT_22 = r'''# The sanity checks of Adebayo et al. on the small CNN (16 images, the true class): four maps, cascading and independent
# randomisation, three seeds of the random weights; then the Grad-CAM of the ResNet of B6.20 in cascade
SOBEL_22 = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])


def edges_22(model):
    """The control: the edges of the images (Sobel), a map that does not look at the model."""
    gray = X_22.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, SOBEL_22.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, SOBEL_22.T.reshape(1, 1, 3, 3), padding=1)
    return (gx ** 2 + gy ** 2).sqrt()[:, 0].numpy()


METHODS_22 = {   # name: (explain(model) -> maps, absolute)
    "Grad-CAM (features.19, 8 × 8)": (lambda m: wbx.grad_cam(m, m.features[19], X_22, T_22), False),
    "gradients intégrés (20 intervalles)": (lambda m: wbx.integrated_gradients(m, X_22, T_22, n_steps=20).sum(axis=1), True),
    "gradient (saillance)": (lambda m: np.abs(wbx.input_gradients(m, X_22, T_22)).max(axis=1), False),
    "contrôle : bords de l'image": (edges_22, False),
}
SEEDS_22 = (0, 1, 2)
if len(ok_22) == 2:
    start_22 = time.perf_counter()
    RESULTS_22 = {}
    for method_22, (explain_22, absolute_22) in METHODS_22.items():
        for mode_22 in ("cascade", "independent"):
            RESULTS_22[method_22, mode_22] = np.stack([sanity_22(CNN, explain_22, NAMES_22, mode=mode_22, seed=s,
                                                                 absolute=absolute_22) for s in SEEDS_22])
    names_r_22 = wbx.cascade_layers(RESNET)
    resnet_cam_22 = lambda m: wbx.grad_cam(m, m.layer4, X_20, POOL_20[1][np.concatenate([RIGHT_20, WRONG_20])])
    RESNET_22 = np.stack([sanity_22(RESNET, resnet_cam_22, names_r_22, mode="cascade", seed=s) for s in SEEDS_22])

    fig_22, axes_22 = plt.subplots(1, 3, figsize=(15, 4))
    for ax_22, mode_22 in zip(axes_22[:2], ("cascade", "independent")):
        for method_22 in METHODS_22:
            rho_22 = RESULTS_22[method_22, mode_22].reshape(-1, len(NAMES_22), 16).transpose(1, 0, 2).reshape(len(NAMES_22), -1)
            med_22 = np.nanmedian(rho_22, axis=1)
            q1_22, q3_22 = np.nanpercentile(rho_22, 25, axis=1), np.nanpercentile(rho_22, 75, axis=1)
            line_22, = ax_22.plot(range(len(NAMES_22)), med_22, marker="o", label=method_22)
            ax_22.fill_between(range(len(NAMES_22)), q1_22, q3_22, color=line_22.get_color(), alpha=0.15)
        ax_22.set_xticks(range(len(NAMES_22)), NAMES_22, rotation=45, ha="right", fontsize=8)
        ax_22.axhline(0, color="gray", lw=0.5)
        ax_22.set_ylim(-0.6, 1.05)
        ax_22.set_title(f"petit CNN, randomisation {'en cascade (jusqu’à la partie)' if mode_22 == 'cascade' else 'indépendante (cette partie seule)'}", fontsize=9)
        ax_22.set_ylabel("corrélation de Spearman (médiane, quartiles)")
    rho_r_22 = RESNET_22.transpose(1, 0, 2).reshape(len(names_r_22), -1)
    axes_22[2].plot(range(len(names_r_22)), np.nanmedian(rho_r_22, axis=1), marker="o", color="C0", label="Grad-CAM (layer4)")
    axes_22[2].fill_between(range(len(names_r_22)), np.nanpercentile(rho_r_22, 25, axis=1),
                            np.nanpercentile(rho_r_22, 75, axis=1), color="C0", alpha=0.15)
    axes_22[2].set_xticks(range(len(names_r_22)), names_r_22, rotation=45, ha="right", fontsize=8)
    axes_22[2].axhline(0, color="gray", lw=0.5)
    axes_22[2].set_ylim(-0.6, 1.05)
    axes_22[2].set_title("ResNet de B6.20, randomisation en cascade (16 images)", fontsize=9)
    axes_22[0].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    print("médiane [1er quartile ; 3e quartile] sur 16 images × 3 graines ; entre parenthèses, l'étendue des médianes "
          "des 3 graines ; nan : cartes constantes (comptées à part)")
    for method_22 in METHODS_22:
        for mode_22 in ("cascade", "independent"):
            r_22 = RESULTS_22[method_22, mode_22]                      # (seeds, parts, images)
            cells_22 = []
            for k_22 in (0, 1, 3, len(NAMES_22) - 1):
                v_22 = r_22[:, k_22, :]
                per_seed_22 = np.nanmedian(v_22, axis=1)
                cells_22.append(f"{NAMES_22[k_22]} {fr(np.nanmedian(v_22), 2)} [{fr(np.nanpercentile(v_22, 25), 2)} ; "
                                f"{fr(np.nanpercentile(v_22, 75), 2)}] ({fr(per_seed_22.min(), 2)} à {fr(per_seed_22.max(), 2)})"
                                + (f", {int(np.isnan(v_22).sum())} nan" if np.isnan(v_22).any() else ""))
            print(f"  {method_22}, {mode_22} : " + " · ".join(cells_22))
    for k_22 in (0, 1, 2, len(names_r_22) - 1):
        v_22 = RESNET_22[:, k_22, :]
        print(f"  ResNet, Grad-CAM, cascade jusqu'à {names_r_22[k_22]} : {fr(np.nanmedian(v_22), 2)} "
              f"[{fr(np.nanpercentile(v_22, 25), 2)} ; {fr(np.nanpercentile(v_22, 75), 2)}]"
              + (f", {int(np.isnan(v_22).sum())} nan" if np.isnan(v_22).any() else ""))
    print(f"ℹ️ {time.perf_counter() - start_22:.0f} s")
else:
    print("⏳ Ex B6.22 : l'expérience attend tes deux fonctions (cellule précédente).")'''

# =====================================================================================================================
# Part C · fairness on Adult
# =====================================================================================================================
TOOLS_C = r'''# Tools of part C: Adult (frozen split; its test part, the one of the MP3, stays aside until the challenge), the
# pipeline of the MP3, the age bands of the MP3, and the reference model of the part (with its validation scores)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

ADULT = wbx.adult()                                   # 29 304 / 9 769 / 9 769 people (the test part: the one of the MP3)
XA_TRAIN, YA_TRAIN = ADULT.X_train, ADULT.y_train.to_numpy()
XA_VAL, YA_VAL = ADULT.X_val, ADULT.y_val.to_numpy()
NUMERIC_C = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_C = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
AGE_BINS_C, AGE_LABELS_C = [25, 35, 45, 55, 65], ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


def adult_pipeline(model=None, drop=()):
    """The pipeline of the MP3 (income.build_pipeline(model, encoding="ordinal")), not fitted: the categorical columns
    coded 0, 1, 2… (OrdinalEncoder, categories seen fewer than 20 times grouped, missing and unknown values as NaN),
    then the numeric columns unchanged; fnlwgt and education left out, and the columns of `drop` too. model: by
    default HistGradientBoostingClassifier(random_state=0), told which columns are categorical."""
    categorical = [c for c in CATEGORICAL_C if c not in drop]
    numeric = [c for c in NUMERIC_C if c not in drop]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=20)
    prep = ColumnTransformer([("categorical", encoder, categorical), ("numeric", "passthrough", numeric)],
                             remainder="drop", verbose_feature_names_out=False, sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(random_state=0) if model is None else model
    if isinstance(model, HistGradientBoostingClassifier):
        model.set_params(categorical_features=list(range(len(categorical))) or None)
    return Pipeline([("prep", prep), ("model", model)])


def age_band(age):
    """The age band of each person (the bands of the MP3): "<25", "25-34", …, "65+" (25 is in "25-34")."""
    return np.asarray(AGE_LABELS_C, dtype=object)[np.digitize(np.asarray(age, dtype=float), AGE_BINS_C)]


def sk_audit(y, p, g):
    """({group: {"n", "positives", "selection_rate", "tpr", "fpr", "accuracy"}}, {"dp", "tpr", "fpr", "eo"}): the rates
    of each group and the gaps (largest minus smallest), from scikit-learn's confusion matrix: the reference of the
    checks and of the experiments given in part C (they never depend on your code)."""
    y, p, g = np.asarray(y, dtype=int), np.asarray(p, dtype=int), np.asarray(g)
    rates = {}
    for key in np.unique(g):
        m = g == key
        tn, fp, fn, tp = sk_confusion_matrix(y[m], p[m], labels=[0, 1]).ravel()
        n = int(m.sum())
        rates[key] = {"n": n, "positives": int(tp + fn), "selection_rate": (tp + fp) / n,
                      "tpr": tp / (tp + fn) if tp + fn else float("nan"),
                      "fpr": fp / (fp + tn) if fp + tn else float("nan"), "accuracy": (tp + tn) / n}
    spread = lambda k: float(max(r[k] for r in rates.values()) - min(r[k] for r in rates.values()))
    gaps = {"dp": spread("selection_rate"), "tpr": spread("tpr"), "fpr": spread("fpr")}
    gaps["eo"] = max(gaps["tpr"], gaps["fpr"])
    return rates, gaps


HGB_C = adult_pipeline().fit(XA_TRAIN, YA_TRAIN)      # the reference model of part C, fitted on the training part
P_VAL = HGB_C.predict_proba(XA_VAL)[:, 1]             # its scores on the validation part
PRED_VAL = (P_VAL >= 0.5).astype(int)                 # its decisions, threshold 0.5
SEX_VAL, AGE_VAL = XA_VAL["sex"].to_numpy(), age_band(XA_VAL["age"])
print(f"Adult : {len(XA_TRAIN)} personnes d'entraînement, {len(XA_VAL)} de validation (le test du MP3, "
      f"{len(ADULT.X_test)} personnes, reste à part jusqu'au défi) ; taux de base {fr(YA_TRAIN.mean(), 3)}. Gradient "
      f"boosting de référence (pipeline du MP3), sur la validation : accuracy {fr(np.mean(PRED_VAL == YA_VAL), 4)} au "
      f"seuil 0,5, ROC-AUC {fr(roc_auc_score(YA_VAL, P_VAL), 4)}.")'''

# --- 🔨 B6.23 ---------------------------------------------------------------------------------------------------------
CHECK_23 = r'''with checked("B6.23"):
    E = mylearn.explain                        # inside the block: a library not created yet gives ⏳
    y_23, p_23, g_23 = [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0], ["a"] * 4 + ["b"] * 4     # the docstring
    r_23 = E.group_rates(list(y_23), list(p_23), list(g_23))
    if returned("B6.23", "group_rates", r_23):
        if not isinstance(r_23, dict) or list(r_23) != ["a", "b"]:
            print(f"❌ Ex B6.23 : group_rates doit rendre un dictionnaire dont les clés sont les groupes, dans l'ordre de "
                  f"np.unique (['a', 'b']), reçu {type(r_23).__name__} {list(r_23)[:4] if isinstance(r_23, dict) else ''}.")
            raise StopCheck
        keys_23 = ["accuracy", "base_rate", "fpr", "n", "precision", "selection_rate", "tpr"]
        verdict("B6.23", sorted(r_23["a"]) == keys_23, "les sept clés de chaque groupe sont là.",
                f"chaque groupe doit avoir les sept clés {keys_23}, reçu {sorted(r_23['a'])}.")
        got_23 = (r_23["a"]["n"], r_23["a"]["tpr"], r_23["a"]["fpr"], r_23["b"]["n"], r_23["b"]["tpr"], r_23["b"]["fpr"])
        verdict("B6.23", got_23 == (4, 0.5, 0.5, 4, 1.0, 0.0) and isinstance(r_23["a"]["n"], (int, np.integer)),
                "l'exemple de la docstring : (4, 0,5, 0,5) et (4, 1,0, 0,0), 'n' entier.",
                f"l'exemple de la docstring donne {got_23} au lieu de (4, 0.5, 0.5, 4, 1.0, 0.0) (n, TPR, FPR des groupes a et b).")

with checked("B6.23"):     # three groups of different sizes, labels drawn at random: every rate against scikit-learn
    E = mylearn.explain                        # inside the block: a library not created yet gives ⏳
    rng_23 = np.random.default_rng(23)
    g3_23 = np.repeat([30, 10, 20], [40, 25, 35])                    # integer groups, not sorted
    y3_23, p3_23 = rng_23.integers(0, 2, 100), rng_23.integers(0, 2, 100)
    r3_23 = E.group_rates(y3_23.copy(), p3_23.copy(), g3_23.copy())
    if returned("B6.23", "group_rates", r3_23):
        if list(r3_23) != [10, 20, 30]:
            print(f"❌ Ex B6.23 : avec les groupes 30, 10 et 20, les clés doivent être 10, 20, 30 (l'ordre de np.unique), "
                  f"reçu {list(r3_23)}.")
            raise StopCheck
        bad_23 = []
        for key_23 in (10, 20, 30):
            m_23 = g3_23 == key_23
            tn, fp, fn, tp = sk_confusion_matrix(y3_23[m_23], p3_23[m_23], labels=[0, 1]).ravel()
            want_23 = {"n": tn + fp + fn + tp, "base_rate": (tp + fn) / m_23.sum(), "selection_rate": (tp + fp) / m_23.sum(),
                       "tpr": tp / (tp + fn), "fpr": fp / (fp + tn), "precision": tp / (tp + fp), "accuracy": (tp + tn) / m_23.sum()}
            bad_23 += [f"{key}[{key_23}] = {fr(float(r3_23[key_23][key]), 4)} au lieu de {fr(float(value), 4)}"
                       for key, value in want_23.items() if not np.isclose(float(r3_23[key_23][key]), float(value))]
        if bad_23:
            swapped_23 = any("tpr" in b for b in bad_23) and any("fpr" in b for b in bad_23)
            print("❌ Ex B6.23 : contre scikit-learn, " + " ; ".join(bad_23[:3])
                  + (" : C.ravel() rend TN, FP, FN, TP (lignes = vérité, comme scikit-learn)" if swapped_23 else "") + ".")
        else:
            print("✅ Ex B6.23 : trois groupes entiers non triés : les sept taux de chaque groupe sont ceux de scikit-learn.")

with checked("B6.23"):     # a group without positives: nan for its TPR, and no crash
    E = mylearn.explain                        # inside the block: a library not created yet gives ⏳
    r4_23 = E.group_rates([0, 0, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0], ["p", "p", "p", "q", "q", "q"])
    if returned("B6.23", "group_rates", r4_23):
        tpr_p_23 = r4_23["p"]["tpr"]
        verdict("B6.23", isinstance(tpr_p_23, float) and math.isnan(tpr_p_23) and math.isclose(r4_23["p"]["fpr"], 1 / 3),
                "un groupe sans positif : TPR nan, FPR 1/3.",
                f"le groupe p n'a aucun positif : son TPR vaut nan (dénominateur nul) et son FPR 1/3, reçu {tpr_p_23} et "
                f"{r4_23['p']['fpr']}.")
with checked("B6.23"):     # a group whose labels AND predictions are all 0: a 1 x 1 matrix without labels=[0, 1]
    E = mylearn.explain                        # inside the block: a library not created yet gives ⏳
    try:
        r5_23 = E.group_rates([0, 0, 1, 0], [0, 0, 1, 1], ["q", "q", "r", "r"])
    except ValueError as exc_23:
        if "unpack" in str(exc_23):
            print("❌ Ex B6.23 : un groupe dont les labels et les prédictions valent tous 0 fait échouer ta fonction ("
                  + str(exc_23) + ") : sa matrice de confusion n'a qu'une case ; relis le contrat : labels=[0, 1].")
            raise StopCheck
        raise
    if returned("B6.23", "group_rates", r5_23):
        q_23 = r5_23["q"]
        verdict("B6.23", q_23["n"] == 2 and math.isnan(q_23["tpr"]) and q_23["fpr"] == 0 and math.isnan(q_23["precision"]),
                "un groupe sans positif ni sélectionné : TPR et precision nan, FPR 0.",
                f"le groupe q (labels et prédictions tous à 0) doit avoir n = 2, un TPR et une precision nan, un FPR 0 ; "
                f"reçu {q_23}.")
with checked("B6.23"):
    E = mylearn.explain                        # inside the block: a library not created yet gives ⏳
    dp_23 = E.demographic_parity_difference([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], ["x"] * 4 + ["y"] * 4 + ["z"] * 4)
    eo_23 = E.equalized_odds_difference([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
                                        ["x"] * 4 + ["y"] * 4 + ["z"] * 4)
    if returned("B6.23", "demographic_parity_difference", dp_23) and returned("B6.23", "equalized_odds_difference", eo_23):
        verdict("B6.23", math.isclose(float(dp_23), 0.5), "parité démographique sur trois groupes : 0,75 − 0,25 = 0,5.",
                f"trois groupes de taux de sélection 0,5, 0,25 et 0,75 : l'écart vaut le plus grand moins le plus petit, "
                f"reçu {dp_23}.")
        verdict("B6.23", math.isclose(float(eo_23), 1.0), "equalized odds sur trois groupes : le plus grand des écarts de TPR et de FPR.",
                f"trois groupes de TPR 1, 0,5 et 1 et de FPR 0,5, 0 et 1 : l'écart vaut le plus grand des deux écarts "
                f"(max − min sur tous les groupes), reçu {eo_23}.")
        refuses("B6.23", "equalized_odds_difference d'un groupe sans positif", lambda: E.equalized_odds_difference(
            [0, 0, 1, 0], [0, 1, 1, 0], ["p", "p", "q", "q"]))
        refuses("B6.23", "demographic_parity_difference d'un seul groupe", lambda: E.demographic_parity_difference([1, 0], ["a", "a"]))
        refuses("B6.23", "group_rates avec un label 2", lambda: E.group_rates([0, 2], [0, 1], ["a", "b"]))'''

# --- 📦 B6.24 ---------------------------------------------------------------------------------------------------------
GIVEN_24 = r'''COLUMNS_24 = ["n", "positives", "base_rate", "selection_rate", "selection_rate_low", "selection_rate_high", "tpr",
              "tpr_low", "tpr_high", "fpr", "fpr_low", "fpr_high", "precision", "accuracy"]
N_BOOT_24 = 500 if FAST_MODE else 2000                # bootstrap draws (the fiche: 2 000)'''

TODO_24 = r'''def rate_ci_24(y_true, y_pred, rate, n_boot=500, rng=None):
    """(low, high): the 95 % percentile bootstrap interval of the rate `rate` ("selection_rate", "tpr" or "fpr") of
    ONE group, from its pairs (y_true, y_pred), with your bootstrap_ci (ch. 2); a draw without a denominator: nan."""
    raise NotImplementedError("rate_ci_24")


def audit_24(y_true, y_pred, groups, n_boot=500, seed=0):
    """pd.DataFrame, one row per group (the order of np.unique, index named "group"), the columns COLUMNS_24: n,
    positives (the y = 1 of the group), the rates of your group_rates, and the interval of selection_rate, tpr and fpr
    (rate_ci_24 on the rows of the group, one rng = np.random.default_rng(seed) for the whole table)."""
    raise NotImplementedError("audit_24")


def gap_ci_24(y_true, y_pred, groups, kind, n_boot=500, seed=0):
    """(gap, low, high): the gap `kind` over the groups ("dp": your demographic_parity_difference, "eo": your
    equalized_odds_difference, "tpr" or "fpr": the largest minus the smallest rate) and its 95 % bootstrap interval,
    the rows of each group resampled with replacement within their group (rng = np.random.default_rng(seed))."""
    raise NotImplementedError("gap_ci_24")'''

SOLUTION_24 = r'''def _rate_24(y, p, rate):
    """A rate of one group from its labels and decisions (nan when its denominator is 0)."""
    y, p = np.asarray(y), np.asarray(p)
    if rate == "selection_rate":
        return float(p.mean()) if len(p) else float("nan")
    keep = y == (1 if rate == "tpr" else 0)
    return float(p[keep].mean()) if keep.any() else float("nan")


def rate_ci_24(y_true, y_pred, rate, n_boot=500, rng=None):
    """(low, high): the 95 % percentile bootstrap interval of the rate `rate` ("selection_rate", "tpr" or "fpr") of
    ONE group, from its pairs (y_true, y_pred), with your bootstrap_ci (ch. 2); a draw without a denominator: nan."""
    pairs = np.column_stack([np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)])   # rows stay paired
    return mylearn.stats.bootstrap_ci(pairs, lambda s: _rate_24(s[:, 0], s[:, 1], rate), n_boot=n_boot, rng=rng)


def audit_24(y_true, y_pred, groups, n_boot=500, seed=0):
    """pd.DataFrame, one row per group (the order of np.unique, index named "group"), the columns COLUMNS_24: n,
    positives (the y = 1 of the group), the rates of your group_rates, and the interval of selection_rate, tpr and fpr
    (rate_ci_24 on the rows of the group, one rng = np.random.default_rng(seed) for the whole table)."""
    y, p, g = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(groups)
    rates = mylearn.explain.group_rates(y, p, g)
    rng = np.random.default_rng(seed)
    rows = {}
    for key, r in rates.items():
        m = g == key
        row = {"n": r["n"], "positives": int(y[m].sum()), **{k: r[k] for k in ("base_rate", "selection_rate", "tpr", "fpr",
                                                                               "precision", "accuracy")}}
        for rate in ("selection_rate", "tpr", "fpr"):
            row[f"{rate}_low"], row[f"{rate}_high"] = rate_ci_24(y[m], p[m], rate, n_boot=n_boot, rng=rng)
        rows[key] = row
    table = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS_24]
    table.index.name = "group"
    return table


def gap_ci_24(y_true, y_pred, groups, kind, n_boot=500, seed=0):
    """(gap, low, high): the gap `kind` over the groups ("dp": your demographic_parity_difference, "eo": your
    equalized_odds_difference, "tpr" or "fpr": the largest minus the smallest rate) and its 95 % bootstrap interval,
    the rows of each group resampled with replacement within their group (rng = np.random.default_rng(seed))."""
    y, p, g = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(groups)
    E = mylearn.explain

    def gap(yy, pp, gg):
        if kind == "dp":
            return float(E.demographic_parity_difference(pp, gg))
        if kind == "eo":
            return float(E.equalized_odds_difference(yy, pp, gg))
        values = [r[kind] for r in E.group_rates(yy, pp, gg).values()]
        return float(max(values) - min(values))

    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(g == key) for key in np.unique(g)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([m[rng.integers(0, len(m), size=len(m))] for m in members])   # within each group
        draws.append(gap(y[idx], p[idx], g[idx]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return gap(y, p, g), float(low), float(high)'''

CHECK_24 = r'''ok_24 = {}


with checked("B6.24"):
    start_24 = time.perf_counter()
    lo_24, hi_24 = rate_ci_24(YA_VAL[SEX_VAL == "Female"].copy(), PRED_VAL[SEX_VAL == "Female"].copy(), "tpr",
                              n_boot=N_BOOT_24, rng=np.random.default_rng(0))
    m_24 = (SEX_VAL == "Female") & (YA_VAL == 1)
    tpr_f_24 = PRED_VAL[m_24].mean()
    half_24 = 1.96 * math.sqrt(tpr_f_24 * (1 - tpr_f_24) / m_24.sum())          # the normal approximation
    verdict("B6.24", lo_24 <= tpr_f_24 <= hi_24 and 0.7 < (hi_24 - lo_24) / (2 * half_24) < 1.3,
            f"rate_ci_24 : TPR des femmes {fr(tpr_f_24, 3)}, intervalle [{fr(lo_24, 3)} ; {fr(hi_24, 3)}], de la largeur "
            "attendue (approximation normale ± 30 %).",
            f"rate_ci_24 : TPR des femmes {fr(tpr_f_24, 3)}, intervalle [{fr(lo_24, 3)} ; {fr(hi_24, 3)}] ; attendu un "
            f"intervalle qui le contient, large d'environ {fr(2 * half_24, 3)} (approximation normale) : sur quelles lignes "
            "tires-tu, et quels centiles gardes-tu ?")
    for name_24, groups_24 in (("sexe", SEX_VAL), ("âge", AGE_VAL)):
        table_24 = audit_24(YA_VAL.copy(), PRED_VAL.copy(), groups_24.copy(), n_boot=N_BOOT_24, seed=0)
        if not isinstance(table_24, pd.DataFrame) or list(table_24.index) != list(np.unique(groups_24)):
            print(f"❌ Ex B6.24 : audit_24 doit rendre un DataFrame indexé par les groupes, dans l'ordre de np.unique "
                  f"({list(np.unique(groups_24))}), reçu {type(table_24).__name__} "
                  f"{list(getattr(table_24, 'index', []))[:6]}.")
            raise StopCheck
        missing_24 = [c for c in COLUMNS_24 if c not in table_24.columns]
        if missing_24:
            print(f"❌ Ex B6.24 : audit_24 ({name_24}) : il manque les colonnes {missing_24}.")
            raise StopCheck
        want_24 = {k: {c: r[c] for c in ("n", "positives", "selection_rate", "tpr", "fpr")}
                   for k, r in sk_audit(YA_VAL, PRED_VAL, groups_24)[0].items()}
        bad_24 = [f"{c}[{k}] = {table_24.loc[k, c]} au lieu de {fr(float(v), 4)}" for k, w in want_24.items()
                  for c, v in w.items() if not np.isclose(float(table_24.loc[k, c]), float(v))]
        if bad_24:
            print(f"❌ Ex B6.24 : audit_24 ({name_24}) : " + " ; ".join(bad_24[:3]) + ".")
            raise StopCheck
        outside_24 = [f"{rate}[{k}]" for k in want_24 for rate in ("selection_rate", "tpr", "fpr")
                      if not table_24.loc[k, f"{rate}_low"] - 1e-9 <= table_24.loc[k, rate] <= table_24.loc[k, f"{rate}_high"] + 1e-9]
        widths_24 = []
        for k, w in want_24.items():
            m_k = groups_24 == k
            den_24 = {"selection_rate": m_k.sum(), "tpr": (m_k & (YA_VAL == 1)).sum(), "fpr": (m_k & (YA_VAL == 0)).sum()}
            for rate in ("selection_rate", "tpr", "fpr"):
                r_24 = float(w[rate])
                if den_24[rate] >= 100 and 0.05 < r_24 < 0.95:
                    widths_24.append(((table_24.loc[k, f"{rate}_high"] - table_24.loc[k, f"{rate}_low"])
                                      / (2 * 1.96 * math.sqrt(r_24 * (1 - r_24) / den_24[rate])), f"{rate}[{k}]"))
        odd_24 = [label for ratio, label in widths_24 if not 0.7 < ratio < 1.3]
        verdict("B6.24", not outside_24 and not odd_24,
                f"audit_24 ({name_24}) : taux de scikit-learn, intervalles qui les contiennent et de la largeur attendue.",
                f"audit_24 ({name_24}) : les taux sont justes, mais les intervalles de {(outside_24 + odd_24)[:3]} "
                + ("ne contiennent pas leur taux" if outside_24 else "n'ont pas la largeur attendue (approximation normale "
                   "± 30 %) : tires-tu dans les lignes du groupe, et gardes-tu les centiles 2,5 et 97,5 ?") + ".")
        if not outside_24 and not odd_24:
            ok_24[name_24] = table_24
    again_24 = audit_24(YA_VAL.copy(), PRED_VAL.copy(), SEX_VAL.copy(), n_boot=N_BOOT_24, seed=0)
    other_24 = audit_24(YA_VAL.copy(), PRED_VAL.copy(), SEX_VAL.copy(), n_boot=N_BOOT_24, seed=1)
    verdict("B6.24", again_24.equals(ok_24.get("sexe", again_24)) and not np.allclose(
        other_24["tpr_low"].to_numpy(float), again_24["tpr_low"].to_numpy(float)),
        "la même graine redonne le même tableau, une autre graine d'autres intervalles.",
        "audit_24 doit être reproductible (même graine, même tableau) et ses intervalles changer avec la graine.")

with checked("B6.24"):
    if len(ok_24) < 2:
        raise StopCheck
    want_s_24 = sk_audit(YA_VAL, PRED_VAL, SEX_VAL)[0]
    tpr_gap_24 = want_s_24["Male"]["tpr"] - want_s_24["Female"]["tpr"]
    gap_24 = gap_ci_24(YA_VAL.copy(), PRED_VAL.copy(), SEX_VAL.copy(), "tpr", n_boot=N_BOOT_24, seed=0)
    if not returned("B6.24", "gap_ci_24", gap_24):
        raise StopCheck
    g_24, l_24, h_24 = (float(v) for v in gap_24)
    half_g_24 = 1.96 * math.sqrt(sum(w["tpr"] * (1 - w["tpr"]) / w["positives"] for w in want_s_24.values()))
    verdict("B6.24", math.isclose(g_24, abs(tpr_gap_24)) and l_24 <= g_24 <= h_24 and 0.7 < (h_24 - l_24) / (2 * half_g_24) < 1.4,
            f"gap_ci_24 : écart de TPR {fr(g_24, 3)}, intervalle [{fr(l_24, 3)} ; {fr(h_24, 3)}].",
            f"gap_ci_24 (« tpr ») rend {fr(g_24, 3)} [{fr(l_24, 3)} ; {fr(h_24, 3)}] ; attendu l'écart {fr(abs(tpr_gap_24), 3)} "
            f"(le plus grand moins le plus petit TPR), dans un intervalle large d'environ {fr(2 * half_g_24, 3)} : "
            "rééchantillonnes-tu chaque groupe dans ses propres lignes ?")
    for kind_24, want_k_24 in (("dp", sk_audit(YA_VAL, PRED_VAL, SEX_VAL)[1]["dp"]),
                               ("eo", sk_audit(YA_VAL, PRED_VAL, SEX_VAL)[1]["eo"])):
        gk_24 = [float(v) for v in gap_ci_24(YA_VAL.copy(), PRED_VAL.copy(), SEX_VAL.copy(), kind_24, n_boot=100, seed=0)]
        verdict("B6.24", math.isclose(gk_24[0], float(want_k_24)) and gk_24[1] <= gk_24[0] <= gk_24[2],
                f"gap_ci_24 (« {kind_24} ») : {fr(gk_24[0], 3)}, dans son intervalle.",
                f"gap_ci_24 (« {kind_24} ») rend {[fr(v, 3) for v in gk_24]} ; attendu l'écart de ta fonction de B6.23 "
                f"({fr(float(want_k_24), 3)}, celui de scikit-learn), dans son intervalle.")
    ok_24["gap"] = True
    print(f"ℹ️ {time.perf_counter() - start_24:.1f} s")'''

EXPERIMENT_24 = r'''# The audit of the reference model on the validation part (threshold 0.5), by sex and by age band, with your functions
if len(ok_24) == 3:
    SHOW_24 = ["n", "positives", "base_rate", "selection_rate", "tpr", "fpr", "precision"]

    def with_ci_24(table):
        out = table[["n", "positives"]].copy()
        for rate in ("base_rate", "selection_rate", "tpr", "fpr", "precision"):
            out[rate] = table[rate].map(lambda v: fr(v, 3))
            if f"{rate}_low" in table:
                out[rate] = [f"{fr(v, 3)} [{fr(lo, 3)} ; {fr(hi, 3)}]" for v, lo, hi in
                             zip(table[rate], table[f"{rate}_low"], table[f"{rate}_high"])]
        return out

    AUDIT_24 = {"sexe": ok_24["sexe"], "âge": ok_24["âge"]}
    for name_24, table_24 in AUDIT_24.items():
        print(f"audit par {name_24} (validation, {N_BOOT_24} tirages ; [intervalle à 95 %]) :")
        print(with_ci_24(table_24).to_string())
        print()
    GAPS_24 = {}
    for name_24, groups_24 in (("sexe", SEX_VAL), ("âge", AGE_VAL)):
        for kind_24 in ("dp", "eo", "tpr", "fpr"):
            GAPS_24[name_24, kind_24] = gap_ci_24(YA_VAL, PRED_VAL, groups_24, kind_24, n_boot=N_BOOT_24, seed=0)
    labels_24 = {"dp": "ΔDP (sélection)", "eo": "ΔEO", "tpr": "écart de TPR", "fpr": "écart de FPR"}
    for name_24 in ("sexe", "âge"):
        print(f"écarts par {name_24} : " + " ; ".join(f"{labels_24[k]} {fr(GAPS_24[name_24, k][0], 3)} "
                                                       f"[{fr(GAPS_24[name_24, k][1], 3)} ; {fr(GAPS_24[name_24, k][2], 3)}]"
                                                       for k in labels_24))
    fig_24, axes_24 = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax_24, rate_24, title_24 in zip(axes_24, ("selection_rate", "tpr", "fpr"), ("taux de sélection", "TPR", "FPR")):
        tables_24 = [ok_24["sexe"], ok_24["âge"]]
        names_24 = [*ok_24["sexe"].index, *ok_24["âge"].index]
        values_24 = np.concatenate([t[rate_24].to_numpy(float) for t in tables_24])
        lows_24 = np.concatenate([t[f"{rate_24}_low"].to_numpy(float) for t in tables_24])
        highs_24 = np.concatenate([t[f"{rate_24}_high"].to_numpy(float) for t in tables_24])
        colors_24 = ["C1"] * len(ok_24["sexe"]) + ["C0"] * len(ok_24["âge"])
        ax_24.errorbar(range(len(names_24)), values_24, yerr=[values_24 - lows_24, highs_24 - values_24], fmt="none",
                       ecolor="gray", capsize=3)
        ax_24.scatter(range(len(names_24)), values_24, c=colors_24, zorder=3)
        ax_24.set_xticks(range(len(names_24)), names_24, rotation=45, fontsize=8)
        ax_24.set_title(f"{title_24} par groupe (intervalles à 95 %)", fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("⏳ Ex B6.24 : l'audit attend tes trois fonctions (cellule précédente) et ta librairie (B6.23, ch. 2).")'''

# --- 📈 B6.25 ---------------------------------------------------------------------------------------------------------
GIVEN_25 = r'''# The audit table to read: the reference model of part C on the validation part (threshold 0.5), by sex and by age
# band, 2 000 bootstrap draws (the solution of B6.24, computed once by tools/chapters/history_chB6.py)
TABLE_25 = pd.DataFrame(«ROWS_25»).set_index("groupe")
GAPS_25 = pd.DataFrame(«GAPS_25»).set_index("écart")
print(TABLE_25.to_string())
print()
print(GAPS_25.to_string())'''

TODO_25 = r'''lowest_tpr_25 = ...    # a) the letter of the AGE band with the smallest TPR: "A" <25, "B" 25-34, "C" 35-44, "D" 45-54, "E" 55-64, "F" 65+
positives_25 = ...     # b) on how many positives of the validation part does this TPR rest? (a whole number)
widest_25 = ...        # c) among the FIVE OTHER age bands, the letter of the one whose TPR interval is the WIDEST
tpr_sex_25 = ...       # d) does the 95 % interval of the TPR gap between the sexes exclude 0? True or False
overlap_25 = ...       # e) do the TPR intervals of the women and of the men overlap? True or False
conclusion_25 = ...    # f) the letter of the conclusion that the table supports (statement)'''

SOLUTION_25 = r'''lowest_tpr_25, positives_25, widest_25 = "«A25»", «B25», "«C25»"
tpr_sex_25, overlap_25, conclusion_25 = «D25», «E25», "«F25»"'''

CHECK_25 = r'''with wb.attempt("B6.25a"):
    wb.check("B6.25a", lowest_tpr_25)
with wb.attempt("B6.25b"):
    wb.check("B6.25b", positives_25)
with wb.attempt("B6.25c"):
    wb.check("B6.25c", widest_25)
with wb.attempt("B6.25d"):
    wb.check("B6.25d", tpr_sex_25)
with wb.attempt("B6.25e"):
    wb.check("B6.25e", overlap_25)
with wb.attempt("B6.25f"):
    wb.check("B6.25f", conclusion_25)'''

# --- 🔮 B6.26 ---------------------------------------------------------------------------------------------------------
TODO_26 = r'''proxy_auc_26 = ...       # a) ROC-AUC of a gradient boosting that predicts the sex from the 10 other features:
                         #    "A" below 0.6, "B" 0.6 to 0.75, "C" 0.75 to 0.9, "D" above 0.9
proxy_auc_few_26 = ...   # b) the same without relationship and marital-status (8 features): same letters
eo_ratio_26 = ...        # c) ΔEO by sex of the model retrained without sex, divided by the ΔEO of the reference model:
                         #    "A" below 0.5, "B" 0.5 to 1.5, "C" above 1.5
four_fifths_26 = ...     # d) model without sex: selection rate of the women / that of the men:
                         #    "A" below 0.4, "B" 0.4 to 0.6, "C" 0.6 to 0.8, "D" 0.8 or more'''

EXPERIMENT_26 = r'''start_26 = time.perf_counter()
NOSEX_26 = adult_pipeline(drop=("sex",)).fit(XA_TRAIN, YA_TRAIN)              # the reference recipe, without sex
PRED_NOSEX_26 = (NOSEX_26.predict_proba(XA_VAL)[:, 1] >= 0.5).astype(int)
female_tr_26, female_va_26 = (XA_TRAIN["sex"] == "Female").to_numpy(int), (SEX_VAL == "Female").astype(int)
AUC_26 = {}
for name_26, drop_26 in (("les 10 autres features", ("sex",)),
                         ("sans relationship ni marital-status", ("sex", "relationship", "marital-status"))):
    spy_26 = adult_pipeline(drop=drop_26).fit(XA_TRAIN, female_tr_26)          # predicts the sex
    AUC_26[name_26] = roc_auc_score(female_va_26, spy_26.predict_proba(XA_VAL)[:, 1])
rows_26 = {}
for name_26, pred_26 in (("avec sex (référence)", PRED_VAL), ("sans sex", PRED_NOSEX_26)):
    rates_26, gaps_26 = sk_audit(YA_VAL, pred_26, SEX_VAL)
    rows_26[name_26] = {"accuracy": np.mean(pred_26 == YA_VAL), "sélection F": rates_26["Female"]["selection_rate"],
                        "sélection M": rates_26["Male"]["selection_rate"],
                        "F / M": rates_26["Female"]["selection_rate"] / rates_26["Male"]["selection_rate"],
                        "TPR F": rates_26["Female"]["tpr"], "TPR M": rates_26["Male"]["tpr"],
                        "FPR F": rates_26["Female"]["fpr"], "FPR M": rates_26["Male"]["fpr"],
                        "ΔDP": gaps_26["dp"], "ΔEO": gaps_26["eo"]}
print("prédire le sexe à partir des autres features (validation) : " + " ; ".join(
    f"{name} : ROC-AUC {fr(v, 3)}" for name, v in AUC_26.items()))
print(pd.DataFrame(rows_26).T.map(lambda v: fr(v, 3)).to_string())
eo_ratio_value_26 = rows_26["sans sex"]["ΔEO"] / rows_26["avec sex (référence)"]["ΔEO"]
print(f"ΔEO sans sex / ΔEO avec : {fr(eo_ratio_value_26, 2)} ; ℹ️ {time.perf_counter() - start_26:.0f} s")
print()
for id_26, answer_26 in (("B6.26a", proxy_auc_26), ("B6.26b", proxy_auc_few_26), ("B6.26c", eo_ratio_26),
                         ("B6.26d", four_fifths_26)):
    wb.check(id_26, answer_26)'''

# --- 🔬 B6.27 ---------------------------------------------------------------------------------------------------------
GIVEN_27 = r'''# The validation part cut in two halves: the thresholds by group are chosen on CAL_27, every strategy is measured on
# EVAL_27 (choosing and measuring on the same people would flatter the thresholds)
order_27 = np.random.default_rng(27).permutation(len(XA_VAL))
CAL_27, EVAL_27 = np.sort(order_27[: len(order_27) // 2]), np.sort(order_27[len(order_27) // 2:])
print(f"moitié de calibrage : {len(CAL_27)} personnes ; moitié d'évaluation : {len(EVAL_27)} personnes")'''

TODO_27 = r'''def reweigh_27(groups, y):
    """np.ndarray (n,): the weight w(g, y) = P(g) P(y) / P(g, y) of each example (Kamiran and Calders, 206.10), the
    probabilities estimated by the shares in these data."""
    raise NotImplementedError("reweigh_27")


def tpr_thresholds_27(scores, y, groups, target_tpr):
    """{group: threshold} (the groups of np.unique): for each group, the quantile 1 - target_tpr of the scores of its
    positives (np.quantile, as in 206.10), so that its TPR on these data is about target_tpr."""
    raise NotImplementedError("tpr_thresholds_27")


def decide_27(scores, groups, thresholds):
    """np.ndarray of 0/1 ints: 1 when the score of an example reaches the threshold of its group
    (scores >= thresholds[group])."""
    raise NotImplementedError("decide_27")'''

SOLUTION_27 = r'''def reweigh_27(groups, y):
    """np.ndarray (n,): the weight w(g, y) = P(g) P(y) / P(g, y) of each example (Kamiran and Calders, 206.10), the
    probabilities estimated by the shares in these data."""
    g, y = np.asarray(groups), np.asarray(y)
    w = np.empty(len(y), dtype=float)
    for key in np.unique(g):
        for label in np.unique(y):
            cell = (g == key) & (y == label)
            if cell.any():
                w[cell] = np.mean(g == key) * np.mean(y == label) / np.mean(cell)
    return w


def tpr_thresholds_27(scores, y, groups, target_tpr):
    """{group: threshold} (the groups of np.unique): for each group, the quantile 1 - target_tpr of the scores of its
    positives (np.quantile, as in 206.10), so that its TPR on these data is about target_tpr."""
    s, y, g = np.asarray(scores, dtype=float), np.asarray(y), np.asarray(groups)
    return {key: float(np.quantile(s[(g == key) & (y == 1)], 1 - target_tpr)) for key in np.unique(g)}


def decide_27(scores, groups, thresholds):
    """np.ndarray of 0/1 ints: 1 when the score of an example reaches the threshold of its group
    (scores >= thresholds[group])."""
    s, g = np.asarray(scores, dtype=float), np.asarray(groups)
    return (s >= np.array([thresholds[key] for key in g])).astype(int)'''

CHECK_27 = r'''ok_27 = {}
with checked("B6.27"):
    g_27 = np.array(["u"] * 6 + ["v"] * 4)
    y_27 = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    w_27 = reweigh_27(g_27.copy(), y_27.copy())
    if returned("B6.27", "reweigh_27", w_27):
        w_27 = np.asarray(w_27, dtype=float)
        # P(u) = 0.6, P(y=1) = 0.5: u-positives 0.6 * 0.5 / 0.4 = 0.75, u-negatives 0.6 * 0.5 / 0.2 = 1.5, ...
        want_w_27 = np.array([0.75] * 4 + [1.5] * 2 + [2.0] + [2 / 3] * 3)
        if w_27.shape != want_w_27.shape:
            print(f"❌ Ex B6.27 : reweigh_27 doit rendre un poids par exemple, forme (10,), reçu {w_27.shape}.")
        elif np.allclose(w_27, 1 / want_w_27):
            print("❌ Ex B6.27 : ce sont les inverses des poids : w(g, y) = P(g) P(y) / P(g, y), les cases rares pèsent plus.")
        else:
            ok_27["reweigh"] = same("B6.27", "reweigh_27 sur deux groupes de 6 et 4", w_27, want_w_27, rtol=1e-9, atol=1e-12)
        base_27 = [np.average(y_27[g_27 == k], weights=w_27[g_27 == k]) for k in ("u", "v")] if w_27.shape == (10,) else []
        verdict("B6.27", len(base_27) == 2 and np.allclose(base_27, 0.5),
                "une fois repondérés, les deux groupes ont le même taux de base (0,5).",
                "une fois repondérés, les deux groupes doivent avoir le même taux de base que l'ensemble des données.")

with checked("B6.27"):
    rng_27 = np.random.default_rng(270)
    s_27 = rng_27.random(400)
    gg_27 = np.where(rng_27.random(400) < 0.4, "a", "b")
    yy_27 = (rng_27.random(400) < np.where(gg_27 == "a", 0.3, 0.5)).astype(int)
    thr_27 = tpr_thresholds_27(s_27.copy(), yy_27.copy(), gg_27.copy(), 0.75)
    if returned("B6.27", "tpr_thresholds_27", thr_27):
        if not isinstance(thr_27, dict) or sorted(thr_27) != ["a", "b"]:
            print(f"❌ Ex B6.27 : tpr_thresholds_27 doit rendre un dictionnaire {{groupe: seuil}}, reçu {type(thr_27).__name__}.")
            raise StopCheck
        want_t_27 = {k: np.quantile(s_27[(gg_27 == k) & (yy_27 == 1)], 0.25) for k in ("a", "b")}
        if all(np.isclose(thr_27[k], np.quantile(s_27[(gg_27 == k) & (yy_27 == 1)], 0.75)) for k in "ab"):
            print("❌ Ex B6.27 : c'est le quantile target_tpr : pour garder 75 % des positifs, le seuil laisse 25 % des scores "
                  "des positifs au-dessous de lui.")
        elif all(np.isclose(thr_27[k], np.quantile(s_27[gg_27 == k], 0.25)) for k in "ab"):
            print("❌ Ex B6.27 : le quantile est pris sur tous les scores du groupe : le TPR ne regarde que ses positifs.")
        else:
            ok_27["thresholds"] = same("B6.27", "tpr_thresholds_27 (TPR visé 0,75)", [thr_27["a"], thr_27["b"]],
                                       [want_t_27["a"], want_t_27["b"]], rtol=1e-9, atol=1e-12)
    d_27 = decide_27(np.array([0.2, 0.5, 0.7, 0.5, 0.9]), np.array(["a", "a", "a", "b", "b"]), {"a": 0.5, "b": 0.6})
    if returned("B6.27", "decide_27", d_27):
        ok_27["decide"] = same("B6.27", "decide_27 (un seuil égal au score décide 1)", np.asarray(d_27).astype(int),
                               [0, 1, 1, 0, 1])'''

EXPERIMENT_27 = r'''# Five strategies, measured on EVAL_27: accuracy, the gaps by sex (with their interval, your gap_ci_24 of B6.24) and
# the gaps by age band, re-audited after each correction
if len(ok_27) == 3:
    start_27 = time.perf_counter()
    nosex_27 = globals().get("NOSEX_26")                                  # the model of B6.26, if it ran
    if nosex_27 is None:
        nosex_27 = adult_pipeline(drop=("sex",)).fit(XA_TRAIN, YA_TRAIN)
    noproxy_27 = adult_pipeline(drop=("sex", "relationship", "marital-status")).fit(XA_TRAIN, YA_TRAIN)
    w_27 = reweigh_27(XA_TRAIN["sex"].to_numpy(), YA_TRAIN)
    reweighed_27 = adult_pipeline().fit(XA_TRAIN, YA_TRAIN, model__sample_weight=w_27)
    target_27 = float(PRED_VAL[CAL_27][YA_VAL[CAL_27] == 1].mean())        # the TPR of the reference on CAL_27
    thr_27 = tpr_thresholds_27(P_VAL[CAL_27], YA_VAL[CAL_27], SEX_VAL[CAL_27], target_27)
    DECISIONS_27 = {
        "référence (seuil 0,5)": PRED_VAL,
        "sans sex": (nosex_27.predict_proba(XA_VAL)[:, 1] >= 0.5).astype(int),
        "sans sex, relationship, marital-status": (noproxy_27.predict_proba(XA_VAL)[:, 1] >= 0.5).astype(int),
        "repondération (sexe × label)": (reweighed_27.predict_proba(XA_VAL)[:, 1] >= 0.5).astype(int),
        "seuils par sexe (même TPR)": np.asarray(decide_27(P_VAL, SEX_VAL, thr_27)).astype(int),
    }
    print("seuils par sexe, choisis sur CAL_27 pour un TPR de", fr(target_27, 3), ":",
          ", ".join(f"{k} {fr(v, 3)}" for k, v in thr_27.items()))
    y_e_27, s_e_27, a_e_27 = YA_VAL[EVAL_27], SEX_VAL[EVAL_27], AGE_VAL[EVAL_27]
    use_ci_27 = len(globals().get("ok_24", {})) == 3
    if not use_ci_27:
        print("⏳ Ex B6.27 : les intervalles attendent ta gap_ci_24 (B6.24) : seuls les écarts sont affichés.")
    TABLE_27 = {}
    for name_27, pred_27 in DECISIONS_27.items():
        p_e_27 = pred_27[EVAL_27]
        row_27 = {"accuracy": fr(np.mean(p_e_27 == y_e_27), 4)}
        for label_27, groups_27 in (("sexe", s_e_27), ("âge", a_e_27)):
            for kind_27 in ("dp", "eo"):
                if use_ci_27 and label_27 == "sexe":
                    g_27, lo_27, hi_27 = gap_ci_24(y_e_27, p_e_27, groups_27, kind_27, n_boot=200, seed=0)
                    row_27[f"Δ{kind_27.upper()} {label_27}"] = f"{fr(g_27, 3)} [{fr(lo_27, 3)} ; {fr(hi_27, 3)}]"
                else:
                    row_27[f"Δ{kind_27.upper()} {label_27}"] = fr(sk_audit(y_e_27, p_e_27, groups_27)[1][kind_27], 3)
        rates_27 = sk_audit(y_e_27, p_e_27, s_e_27)[0]
        row_27["TPR F / M"] = f"{fr(rates_27['Female']['tpr'], 2)} / {fr(rates_27['Male']['tpr'], 2)}"
        row_27["FPR F / M"] = f"{fr(rates_27['Female']['fpr'], 3)} / {fr(rates_27['Male']['fpr'], 3)}"
        TABLE_27[name_27] = row_27
    print(pd.DataFrame(TABLE_27).T.to_string())
    print(f"ℹ️ {time.perf_counter() - start_27:.0f} s")
else:
    print("⏳ Ex B6.27 : l'expérience attend tes trois fonctions (cellule précédente).")'''

# --- 🛠️ B6.28 ---------------------------------------------------------------------------------------------------------
GIVEN_28 = r'''# What the fiche must say about the reference model of part C (FACTS_28), and the eleven sections it must have
import re
import tempfile
import unicodedata
from pathlib import Path

rates_sex_28, gaps_sex_28 = sk_audit(YA_VAL, PRED_VAL, SEX_VAL)
rates_age_28, gaps_age_28 = sk_audit(YA_VAL, PRED_VAL, AGE_VAL)
FACTS_28 = {
    "modèle": "HistGradientBoostingClassifier(random_state=0) dans le pipeline du MP3 (encodage ordinal), seuil 0,5",
    "données": f"Adult (recensement américain de 1994, UCI / OpenML, licence CC BY 4.0) : {len(XA_TRAIN)} personnes "
               f"d'entraînement, {len(XA_VAL)} de validation",
    "accuracy (validation)": round(float(np.mean(PRED_VAL == YA_VAL)), 4),
    "ROC-AUC (validation)": round(float(roc_auc_score(YA_VAL, P_VAL)), 4),
    "groupes": {k: {"n": r["n"], "TPR": round(float(r["tpr"]), 2), "FPR": round(float(r["fpr"]), 2)}
                for k, r in {**rates_sex_28, **rates_age_28}.items()},
}
perm_28 = permutation_importance(HGB_C, XA_VAL, YA_VAL, n_repeats=3, random_state=0, scoring="accuracy")
PERM_28 = pd.Series(perm_28.importances_mean, index=XA_VAL.columns).sort_values(ascending=False)
FACTS_28["permutation (validation, baisse d'accuracy, 3 répétitions)"] = {k: round(float(v), 4) for k, v in PERM_28.head(5).items()}
REQUIRED_28 = ["Détails du modèle", "Usage prévu", "Facteurs", "Métriques", "Données d'évaluation",
               "Données d'entraînement", "Analyses quantitatives", "Explicabilité", "Équité", "Considérations éthiques",
               "Mises en garde et recommandations"]


def _plain_28(text):
    """Lower case, without accents nor apostrophes of any kind (to compare section titles)."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def sections_28(card):
    """{normalised title: text} of the sections `## ...` of a Markdown text, in order."""
    out, current = {}, None
    for line in str(card).splitlines():
        if line.startswith("## "):
            current = " ".join(_plain_28(line[3:]))
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return out


def check_card_28(card):
    """The list of what is missing in the fiche (empty: everything is there)."""
    if not isinstance(card, str):
        return ["CARD_28 doit être une chaîne de caractères (du Markdown)."]
    problems, found = [], sections_28(card)
    titles = list(found)
    wanted = []
    for title in REQUIRED_28:                       # a title may go on: "## Usage prévu (et hors périmètre)"
        words = " ".join(_plain_28(title))
        wanted.append(next((t for t in titles if t == words or t.startswith(words + " ")), words))
    missing = [t for t, w in zip(REQUIRED_28, wanted) if w not in found]
    if missing:
        problems.append(f"il manque les sections (titres « ## … ») : {', '.join(missing)}.")
    else:
        order = [titles.index(w) for w in wanted]
        if order != sorted(order):
            problems.append("les onze sections doivent venir dans l'ordre de REQUIRED_28.")
        text = {t: found[w] for t, w in zip(REQUIRED_28, wanted)}
        if "hors" not in _plain_28(text["Usage prévu"]):
            problems.append("« Usage prévu » doit aussi dire les usages hors périmètre (le mot « hors »).")
        if not re.search(r"0[.,]5\b", text["Métriques"]):
            problems.append("« Métriques » doit donner le seuil de décision (0,5).")
        acc = FACTS_28["accuracy (validation)"]
        if not re.search(rf"0[.,]{str(acc)[2:]}", card):
            problems.append(f"la fiche doit donner l'accuracy de validation à 4 décimales ({str(acc).replace('.', ',')}).")
        table = text["Analyses quantitatives"] + text["Équité"]
        for group, facts in FACTS_28["groupes"].items():
            lines = [line for line in table.splitlines() if group in line]
            numbers = {f"{facts[k]:.2f}" for k in ("TPR", "FPR")}
            if not any(str(facts["n"]) in line and all(re.search(v.replace(".", "[.,]"), line) for v in numbers)
                       for line in lines):
                problems.append(f"aucune ligne des sections « Analyses quantitatives » et « Équité » ne donne, pour le "
                                f"groupe {group}, son effectif ({facts['n']}), son TPR et son FPR (2 décimales).")
                break
        words = set(_plain_28(text["Équité"]))
        if not ({"equalized", "odds"} <= words or {"egalite", "chances"} <= words or {"parite", "demographique"} <= words):
            problems.append("« Équité » doit nommer le critère retenu (parité démographique, égalité des chances ou "
                            "equalized odds) et dire pourquoi.")
        if "intervalle" not in " ".join(_plain_28(text["Équité"])):
            problems.append("« Équité » doit parler des intervalles (bootstrap) de ses écarts.")
        if not set(_plain_28(text["Explicabilité"])) & {"shap", "shapley", "permutation", "dependance", "ice"}:
            problems.append("« Explicabilité » doit nommer une méthode (SHAP, permutation, dépendance partielle, ICE…).")
        if not {"fond", "validation"} & set(_plain_28(text["Explicabilité"])):
            problems.append("« Explicabilité » doit dire sur quelles données l'explication est calculée (la validation, "
                            "ou le fond d'une valeur SHAP).")
    if len(re.findall(r"\w+", card)) < 300:
        problems.append(f"la fiche fait {len(re.findall(r'\w+', card))} mots : il en faut au moins 300.")
    return problems


print("ce que la fiche doit dire (FACTS_28) :")
for key_28, value_28 in FACTS_28.items():
    print(f"  {key_28} : {value_28}")
print("sections, dans cet ordre (REQUIRED_28) :", " · ".join(REQUIRED_28))'''

TODO_28 = r'''CARD_28 = ...   # your model card, in Markdown: a title "# ...", then the eleven sections of REQUIRED_28 ("## ...")'''

SOLUTION_28 = r'''_g = FACTS_28["groupes"]
_perm = ", ".join(f"`{k}` {fr(v, 4)}" for k, v in PERM_28.head(4).items())
_rows = "\n".join(f"| {k} | {v['n']} | {v['TPR']:.2f} | {v['FPR']:.2f} |".replace(".", ",") for k, v in _g.items())
CARD_28 = f"""# Fiche modèle : revenu supérieur à 50 000 $ (Adult), gradient boosting audité (workbook, B6)

## Détails du modèle
Gradient boosting `HistGradientBoostingClassifier(random_state=0)` de scikit-learn 1.6, dans le pipeline sans fuite
du MP3 : catégories codées 0, 1, 2… (`OrdinalEncoder`, catégories rares regroupées), colonnes numériques telles
quelles, `fnlwgt` et `education` écartées. Entraîné le 3 novembre 2026 pour le chapitre B6 du workbook ; version 1 ;
code sous licence du dépôt. Seuil de décision : 0,5.

## Usage prévu
Exercice pédagogique : illustrer un audit d'équité et des corrections. Hors périmètre : toute décision sur une
personne réelle (crédit, embauche, logement, aide sociale), et toute population d'aujourd'hui : les données datent de
1994 et ne représentent que des adultes interrogés aux États-Unis.

## Facteurs
Le sexe (deux valeurs dans les données), la tranche d'âge (six tranches du MP3), et, à auditer ensuite, la catégorie
raciale et le pays d'origine, aux effectifs très inégaux. Les performances varient d'abord avec le taux de base de chaque
groupe (validation) : {fr(rates_sex_28["Female"]["positives"] / rates_sex_28["Female"]["n"], 3)} pour les femmes,
{fr(rates_sex_28["Male"]["positives"] / rates_sex_28["Male"]["n"], 3)} pour les hommes,
{fr(rates_age_28["<25"]["positives"] / rates_age_28["<25"]["n"], 3)} pour les moins de 25 ans
({rates_age_28["<25"]["positives"]} positifs sur {rates_age_28["<25"]["n"]}).

## Métriques
Accuracy au seuil 0,5 et ROC-AUC sur la validation ; par groupe : taux de sélection, TPR, FPR et precision ; écarts :
parité démographique (ΔDP) et equalized odds (ΔEO, le plus grand des écarts de TPR et de FPR), avec des intervalles
bootstrap à 95 % (tirages dans chaque groupe).

## Données d'évaluation
La partie validation de `wbx.adult()` : {len(XA_VAL)} personnes, stratifiée sur le label ; le test du MP3
({len(ADULT.X_test)} personnes) reste à part pour le défi B6.29.

## Données d'entraînement
{FACTS_28["données"]} ; taux de base 0,239 ; data card : `data/cards/adult.md` (provenance, valeurs manquantes,
biais connus de la collecte).

## Analyses quantitatives
Validation : accuracy {str(FACTS_28["accuracy (validation)"]).replace(".", ",")}, ROC-AUC
{str(FACTS_28["ROC-AUC (validation)"]).replace(".", ",")}.

| groupe | n | TPR | FPR |
|---|---|---|---|
{_rows}

## Explicabilité
Importance par permutation sur la validation (baisse d'accuracy, 3 répétitions) : {_perm}. Une valeur SHAP
par personne demanderait de dire son fond (la valeur de base et les signes en dépendent, 🔬 B6.16). Limites : des
features corrélées se partagent l'information (`relationship`, `marital-status` et `sex`), et une importance n'est pas
une cause.

## Équité
Critère retenu : l'equalized odds par sexe, parce qu'une erreur de classement fausse les statistiques publiées d'un
groupe dans les deux sens (faux positifs et faux négatifs). Au seuil 0,5, ΔEO par sexe vaut
{fr(gaps_sex_28["eo"], 3)} et ΔDP {fr(gaps_sex_28["dp"], 3)} ; les intervalles bootstrap (B6.24) excluent 0 pour
l'écart de TPR. Corrections essayées (B6.27, moitié d'évaluation de la validation) : retirer `sex` ne réduit pas les
écarts (les autres colonnes prédisent le sexe avec une ROC-AUC de 0,93) ; retirer aussi `relationship` et
`marital-status` coûte 2,5 points d'accuracy et réduit ΔDP, pas ΔEO ; des seuils par sexe choisis pour un même TPR
le dépassent chez les femmes sur d'autres personnes (175 positives par moitié de validation, c'est peu), et utilisent
l'attribut dans la décision (risque juridique, 206.10). Les écarts de sélection par âge suivent surtout les taux de
base, très différents d'une tranche à l'autre.

## Considérations éthiques
Le sexe et l'âge sont des attributs protégés ; les catégories du recensement de 1994 (sexe binaire, catégories
raciales) ne décrivent pas tout le monde. Les labels (revenu déclaré) portent eux-mêmes les inégalités de l'époque :
un modèle qui les reproduit fidèlement n'est pas « neutre ».

## Mises en garde et recommandations
Ne pas déployer. Réauditer après toute correction, sur tous les groupes, avec les effectifs et les intervalles ; les
moins de 25 ans n'ont qu'une poignée de positifs, et leurs taux sont très incertains. Refaire l'audit sur des données
récentes (ACS, `folktables`) avant toute conclusion sur aujourd'hui.
"""'''

CHECK_28 = r'''if CARD_28 is ... or len(re.findall(r"\w+", str(CARD_28))) < 10:
    print("⏳ Ex B6.28 : pas encore fait (rédige CARD_28).")
else:
    problems_28 = check_card_28(CARD_28)
    for problem_28 in problems_28:
        print(f"❌ Ex B6.28 : {problem_28}")
    if not problems_28:
        print("✅ Ex B6.28 : les onze sections, les chiffres par groupe, le critère et ses limites sont là.")
        card_path_28 = (Path("fiche_modele_adult_b6.md") if "mon_travail" in str(Path.cwd())   # next to your notebook
                        else Path(tempfile.mkdtemp(prefix="wb_b6_28_")) / "fiche_modele_adult_b6.md")
        card_path_28.write_text(CARD_28, encoding="utf-8")
        print(f"   écrite dans {card_path_28.name}" + (", à côté de ton notebook : garde-la pour ton portfolio."
                                                        if "mon_travail" in str(Path.cwd()) else "."))'''

# --- 🏆 B6.29 ---------------------------------------------------------------------------------------------------------
GIVEN_29 = r'''# The challenge: ΔEO by sex below GOAL_EO_29 on the test of the MP3, losing less than MAX_LOSS_29 of accuracy against
# the reference gradient boosting (no correction, threshold 0.5) on the same test; the same on a control split
GOAL_EO_29 = «GOAL_29»
MAX_LOSS_29 = 0.02
REFERENCE_29 = «REF_29»       # the reference solution on «N_CAL_29» splits and seeds: ΔEO (best, worst), drop of accuracy (smallest, largest)
CONTROL_SEED_29 = «CONTROL_29»   # the control split: wbx.adult(seed=CONTROL_SEED_29), whose test rows are other people
MAX_BALANCED_29 = 0.56        # balanced accuracy allowed to a method trained on shuffled labels (chance: 0.5)


def _make_evaluate_29():
    """evaluate_29, with the test parts and their labels kept here only (the global variables that hold data, a fitted
    model or a split are replaced by a decoy while your method runs and decides)."""
    import re
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_is_fitted
    from wb import datasets as wbdata

    splits = {"test du MP3": wbx.adult(), "contrôle": wbx.adult(seed=CONTROL_SEED_29)}

    def baseline(split):
        model = adult_pipeline().fit(split.X_train, split.y_train.to_numpy())
        return float(np.mean((model.predict_proba(split.X_test)[:, 1] >= 0.5) == split.y_test.to_numpy()))

    base = {name: baseline(split) for name, split in splits.items()}

    class Hidden:
        """What a hidden global variable holds during the run: any use of it says why."""

        def __init__(self, name):
            self._name = name

        def _refuse(self, *args, **kwargs):
            raise RuntimeError(f"pendant le défi, {self._name} est masqué : ta méthode ne reçoit que X_train, y_train, "
                               "X_val et y_val, et decide que les features à décider")

        __getattr__ = __getitem__ = __call__ = __iter__ = __len__ = _refuse

    ipython_names = re.compile(r"^(_+|_i+|_i?\d+|_[ido]h|In|Out|exit|quit|get_ipython)$")

    def fitted(v):
        try:
            check_is_fitted(v)
            return True
        except Exception:  # noqa: BLE001 - not fitted, or not an estimator scikit-learn knows
            return False

    def hideable(v, level=0):
        """True for what could hold Adult (a table, an array, a split) or a fitted model, up to three levels down."""
        if isinstance(v, wbx.TabularSplit):
            return True
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return len(v) >= 1000
        if isinstance(v, np.ndarray):
            return v.size >= 1000
        if isinstance(v, BaseEstimator):
            return fitted(v)
        if isinstance(v, dict):
            return level < 3 and any(hideable(x, level + 1) for x in list(v.values())[:10_000])
        if isinstance(v, (list, tuple, set, frozenset)):
            return level < 3 and any(hideable(x, level + 1) for x in list(v)[:10_000])
        return False

    def captured(method):
        kept = list(getattr(method, "__defaults__", None) or ()) + list((getattr(method, "__kwdefaults__", None)
                                                                          or {}).values())
        kept += [cell.cell_contents for cell in (getattr(method, "__closure__", None) or ())]
        return any(hideable(v) for v in kept)

    def refused(*args, **kwargs):
        raise RuntimeError("pendant le défi, ta méthode ne reçoit que ses arguments (ni chargeur de données du workbook, "
                           "ni lecture du disque par wb)")

    def run(method, split, shuffle_seed=None):
        """Fit the method cold on the train and validation parts (labels shuffled for the control), then decide the
        test part, whole and row by row (on 200 rows), every hideable global variable hidden."""
        X_tr, y_tr = split.X_train.copy(), split.y_train.to_numpy().copy()
        X_va, y_va = split.X_val.copy(), split.y_val.to_numpy().copy()
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            y_tr, y_va = rng.permutation(y_tr), rng.permutation(y_va)
        X_te = split.X_test.copy()
        g = globals()
        hidden = {n: Hidden(n) for n, v in list(g.items()) if not ipython_names.match(n) and hideable(v)}
        originals = {n: g[n] for n in hidden}
        saved = {name: getattr(wbx, name) for name in ("adult", "california", "penguins")}
        saved_data = {name: getattr(wbdata, name) for name in ("load_adult",)}
        cache = dict(wbx._TABULAR)
        try:
            g.update(hidden)
            wbx._TABULAR.clear()
            for name in saved:
                setattr(wbx, name, refused)
            for name in saved_data:
                setattr(wbdata, name, refused)
            start = time.perf_counter()
            decide = method(X_tr, y_tr, X_va, y_va)
            seconds = time.perf_counter() - start
            if not callable(decide):
                return None, f"method_29 doit rendre une fonction decide(X) ; reçu {type(decide).__name__}."
            whole = np.asarray(decide(X_te.copy()))
            alone = np.concatenate([np.asarray(decide(X_te.iloc[[i]].copy())).reshape(-1) for i in range(200)])
        finally:
            g.update(originals)
            wbx._TABULAR.update(cache)
            for name, f in saved.items():
                setattr(wbx, name, f)
            for name, f in saved_data.items():
                setattr(wbdata, name, f)
        if whole.shape != (len(X_te),) or not np.isin(whole, (0, 1)).all():
            return None, (f"decide(X) doit rendre un tableau de 0 et de 1 de forme ({len(X_te)},) ; reçu la forme "
                          f"{whole.shape}.")
        if not np.array_equal(alone.astype(int), whole[:200].astype(int)):
            return None, ("decide ne décide pas chaque personne seule : sur les 200 premières, décider une ligne à la fois "
                          "ne donne pas les mêmes décisions que tout le test d'un coup (une décision ne doit dépendre "
                          "que de la personne).")
        return {"decisions": whole.astype(int), "seconds": seconds}, ""

    def measure(split, decisions, name):
        y = split.y_test.to_numpy()
        rates, gaps = sk_audit(y, decisions, split.X_test["sex"].to_numpy())
        acc = float(np.mean(decisions == y))
        return {"split": name, "accuracy": acc, "loss": base[name] - acc, "eo": gaps["eo"], "tpr_gap": gaps["tpr"],
                "fpr_gap": gaps["fpr"], "balanced": float(balanced_accuracy_score(y, decisions))}

    def evaluate_29(method):
        """method(X_train, y_train, X_val, y_val) -> decide(X): the rules of the statement, then the two measures."""
        if captured(method):
            print("❌ Ex B6.29 : method_29 garde, dans ses arguments par défaut ou sa fermeture, des données d'Adult ou un "
                  "modèle déjà entraîné : elle ne doit utiliser que ses arguments.")
            return None
        results = {}
        for name, split in splits.items():
            out, why = run(method, split)
            if out is None:
                print(f"❌ Ex B6.29 ({name}) : {why}")
                return None
            results[name] = measure(split, out["decisions"], name) | {"seconds": out["seconds"]}
        out, why = run(method, splits["test du MP3"], shuffle_seed=29)
        if out is None:
            print(f"❌ Ex B6.29 (labels mélangés) : {why}")
            return None
        shuffled = measure(splits["test du MP3"], out["decisions"], "test du MP3")
        first = globals().get("FIRST_29")
        if first is None:
            globals()["FIRST_29"] = first = dict(results["test du MP3"])
        for name, r in results.items():
            print(f"{name} : accuracy {fr(r['accuracy'], 4)} (référence sans correction {fr(base[name], 4)}, "
                  f"{fr(100 * r['loss'], 2)} point{'s' if abs(r['loss']) >= 0.02 else ''} de moins) ; ΔEO par sexe {fr(r['eo'], 3)} "
                  f"(écart de TPR {fr(r['tpr_gap'], 3)}, de FPR {fr(r['fpr_gap'], 3)}) ; ajustement en {fr(r['seconds'], 1)} s.")
        if first["eo"] != results["test du MP3"]["eo"] or first["accuracy"] != results["test du MP3"]["accuracy"]:
            print(f"ℹ️ le test du MP3 a déjà été révélé dans cette session : ΔEO {fr(first['eo'], 3)}, accuracy "
                  f"{fr(first['accuracy'], 4)} à la première révélation ; c'est elle qui compte (règle d'honneur).")
        rules = [(shuffled["balanced"] <= MAX_BALANCED_29,
                  f"contrôle des labels mélangés : balanced accuracy {fr(shuffled['balanced'], 3)} (au plus "
                  f"{fr(MAX_BALANCED_29, 2)} ; le hasard : 0,5).",
                  f"entraînée sur des labels mélangés, ta méthode décide encore le test avec une balanced accuracy de "
                  f"{fr(shuffled['balanced'], 3)} (au plus {fr(MAX_BALANCED_29, 2)}) : apprend-elle d'une autre source "
                  "que les labels qu'elle reçoit ?"),
                 (all(r["accuracy"] <= base[n] + 0.02 for n, r in results.items()),
                  "aucune accuracy trop belle (au plus 2 points au-dessus de la référence sans correction).",
                  "une accuracy dépasse de plus de 2 points celle de la référence sans correction : une correction "
                  "d'équité ne fait pas mieux qu'elle d'autant ; une fuite ?")]
        for ok, success, failure in rules:
            verdict("B6.29", ok, success, failure)
        goals = {n: r["eo"] < GOAL_EO_29 and r["loss"] < MAX_LOSS_29 for n, r in results.items()}
        ok = all(rule[0] for rule in rules) and all(goals.values())
        verdict("B6.29", ok,
                f"défi réussi : ΔEO {fr(results['test du MP3']['eo'], 3)} et {fr(results['contrôle']['eo'], 3)} (objectif "
                f"sous {fr(GOAL_EO_29, 2)}), accuracy en baisse de {fr(100 * results['test du MP3']['loss'], 2)} et "
                f"{fr(100 * results['contrôle']['loss'], 2)} point (moins de {fr(100 * MAX_LOSS_29, 0)}) ; référence : ΔEO de "
                f"{fr(REFERENCE_29['eo'][0], 3)} à {fr(REFERENCE_29['eo'][1], 3)}.",
                "pas encore : " + " ; ".join(
                    f"{n} : ΔEO {fr(r['eo'], 3)} (sous {fr(GOAL_EO_29, 2)}), accuracy en baisse de {fr(100 * r['loss'], 2)} point "
                    f"(moins de {fr(100 * MAX_LOSS_29, 0)})" for n, r in results.items()) + ".")
        return {"results": results, "shuffled": shuffled, "ok": ok}

    return evaluate_29


evaluate_29 = _make_evaluate_29()
print(f"Objectif : ΔEO par sexe sous {fr(GOAL_EO_29, 2)} et moins de {fr(100 * MAX_LOSS_29, 0)} points d'accuracy perdus, sur "
      f"le test du MP3 et sur le découpage de contrôle (seed={CONTROL_SEED_29}) ; référence : ΔEO de "
      f"{fr(REFERENCE_29['eo'][0], 3)} à {fr(REFERENCE_29['eo'][1], 3)}, accuracy en baisse de "
      f"{fr(100 * REFERENCE_29['loss'][0], 2)} à {fr(100 * REFERENCE_29['loss'][1], 2)} point.")'''

TODO_29 = r'''def method_29(X_train, y_train, X_val, y_val):
    """Your method: fit a NEW model on these data only (X_*: DataFrames of the 14 columns of Adult, sex included;
    y_*: int arrays of 0/1) and return decide(X), which returns the 0/1 decisions (an int array (len(X),)) of the
    people of the DataFrame X, each person decided alone."""
    raise NotImplementedError("method_29")'''

SOLUTION_29 = r'''TARGET_EO_29 = 0.02     # the gap aimed at on the validation part: below the goal, for a margin on new people


def method_29(X_train, y_train, X_val, y_val):
    """Your method: fit a NEW model on these data only (X_*: DataFrames of the 14 columns of Adult, sex included;
    y_*: int arrays of 0/1) and return decide(X), which returns the 0/1 decisions (an int array (len(X),)) of the
    people of the DataFrame X, each person decided alone."""
    model = adult_pipeline().fit(X_train, y_train)                 # the reference recipe, without correction
    scores, sex = model.predict_proba(X_val)[:, 1], X_val["sex"].to_numpy()
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curves = {}
    for group in ("Female", "Male"):                               # TPR, FPR and right decisions at each threshold
        pos, neg = scores[(sex == group) & (y_val == 1)], scores[(sex == group) & (y_val == 0)]
        tpr = (pos[None, :] >= grid[:, None]).mean(axis=1)
        fpr = (neg[None, :] >= grid[:, None]).mean(axis=1)
        curves[group] = (tpr, fpr, tpr * len(pos) + (1 - fpr) * len(neg))
    (tpr_f, fpr_f, right_f), (tpr_m, fpr_m, right_m) = curves["Female"], curves["Male"]
    eo = np.maximum(np.abs(tpr_f[:, None] - tpr_m[None, :]), np.abs(fpr_f[:, None] - fpr_m[None, :]))
    acc = (right_f[:, None] + right_m[None, :]) / len(y_val)
    if (eo <= TARGET_EO_29).any():                                 # the best accuracy among the fair enough pairs
        i, j = np.unravel_index(np.argmax(np.where(eo <= TARGET_EO_29, acc, -1.0)), acc.shape)
    else:                                                          # else the smallest gap
        i, j = np.unravel_index(np.argmin(eo), eo.shape)
    thresholds = {"Female": float(grid[i]), "Male": float(grid[j])}

    def decide(X):
        t = np.array([thresholds.get(value, 0.5) for value in X["sex"]])
        return (model.predict_proba(X)[:, 1] >= t).astype(int)

    return decide'''

CHECK_29 = r'''result_29 = None
with checked("B6.29"):
    if globals().get("method_29") is None:
        raise StopCheck
    start_29 = time.perf_counter()
    result_29 = evaluate_29(method_29)
    print(f"ℹ️ {time.perf_counter() - start_29:.0f} s")'''
