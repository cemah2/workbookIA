"""Code cells of the chapter B5 notebook, second half (part C after the U-Net: B5.22 to B5.31), shared by build_chB5.py
(through chB5_parts2.py) and history_chB5.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
A few cells hold placeholders «NAME», filled by chB5_parts2.py with the precomputed results of history_chB5.json.
"""

# --- 🐛 B5.22 ---------------------------------------------------------------------------------------------------------
GIVEN_22 = r'''# 500 MNIST training digits as they are stored (uint8 in [0, 255]), the schedule of the notebook, and the tools of
# the exercise: the four chained diagnoses and a PERFECT denoiser for these 500 images (no network to train)
RAW_22, _ = wbd.raw_subset("mnist", 500)
AB_22 = wbd.alpha_bars(BETAS)


def coefficients_22(noisy_fn):
    """The two coefficients of a forward process x_t = c1[t] x0 + c2[t] eps, read from noisy_fn at every index."""
    T = len(AB_22)
    t = torch.arange(T)
    ones, zeros = torch.ones(T, 1, 28, 28), torch.zeros(T, 1, 28, 28)
    c1 = torch.as_tensor(noisy_fn(ones, t, zeros, AB_22)).reshape(T, -1)[:, 0].double().numpy()
    c2 = torch.as_tensor(noisy_fn(zeros, t, ones, AB_22)).reshape(T, -1)[:, 0].double().numpy()
    return c1, c2


def diagnose_22(prepare_fn, noisy_fn, sample_fn):
    """Four diagnoses of the pipeline, in a row: the first one that fails stops the chain (one bug can hide another).
    Returns True when the four pass."""

    def point1():
        x = torch.as_tensor(prepare_fn(RAW_22[:64]))
        if tuple(x.shape) != (64, 1, 28, 28) or not x.is_floating_point():
            return False, f"prepare_22 rend {tuple(x.shape)} en {x.dtype}", "attendu un tenseur de flottants (64, 1, 28, 28)"
        lo, hi = float(x.min()), float(x.max())
        if abs(lo + 1) < 1e-6 and abs(hi - 1) < 1e-6:
            return True, "les pixels préparés vont de −1 (noir) à 1 (blanc)", ""
        return False, (f"les pixels préparés vont de {fr(lo, 3)} à {fr(hi, 3)} (un pixel noir de MNIST vaut 0, un blanc "
                       "255)"), "sur quel intervalle le bruit et l'écrêtage de x̂₀ supposent-ils les images ?"

    def point2():
        _, c2 = coefficients_22(noisy_fn)
        want = np.sqrt(1 - AB_22)
        bad = np.flatnonzero(np.abs(c2 - want) > 1e-5)
        if not len(bad):
            return True, "à chaque indice, la part du bruit dans x_t est celle de la forme fermée", ""
        i = int(bad[0])
        return False, (f"à l'indice {i}, la part du bruit dans x_t vaut {fr(c2[i], 4)} (forme fermée à cet indice : "
                       f"{fr(want[i], 4)})"), "quel ᾱ ta fonction lit-elle pour un exemple d'indice t ?"

    def point3():
        c1, c2 = coefficients_22(noisy_fn)
        total = c1 ** 2 + c2 ** 2
        bad = np.flatnonzero(np.abs(total - 1) > 1e-5)
        if not len(bad):
            return True, "à chaque indice, c1² + c2² = 1 : la variance d'un pixel de variance 1 est préservée", ""
        i = int(bad[len(bad) // 2])
        return False, (f"à l'indice {i}, x_t = {fr(c1[i], 4)} x₀ + {fr(c2[i], 4)} ε : un pixel de variance 1 et un "
                       f"bruit indépendant de variance 1 donnent une variance {fr(total[i], 4)}, pas 1"), \
            "relis la forme fermée (fiche §205.2) : que vaut le coefficient de x₀ ?"

    def point4():
        x0 = torch.as_tensor(prepare_fn(RAW_22[:8])).float()
        eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(22))
        last = torch.full((8,), len(AB_22) - 1)
        x_T = torch.as_tensor(noisy_fn(x0, last, eps, AB_22)).float()

        def true_noise(x, t):                     # the noise of the current x, for the image x0 (a perfect network)
            a = float(AB_22[int(t)])
            return (x - math.sqrt(a) * x0) / math.sqrt(1 - a)

        out = torch.as_tensor(sample_fn(true_noise, x_T, AB_22, torch.Generator().manual_seed(23)))
        err = float((out.double() - x0.double()).abs().max())
        if err < 1e-3:
            return True, f"avec le vrai bruit du x courant, la chaîne finit à {fr_e(err)} de x₀", ""
        return False, (f"avec le vrai bruit du x courant à chaque pas, la chaîne finit à {fr(err, 4)} de x₀ (B5.18 : "
                       "elle devrait le retrouver)"), "qu'ajoute ta boucle à chaque pas, et à quels indices ?"

    points = [("échelle des images", point1), ("indices du bruitage", point2), ("coefficients du bruitage", point3),
              ("aller-retour avec le vrai bruit", point4)]
    for k, (name, run) in enumerate(points, start=1):
        try:
            ok, measured, why = run()
        except Exception as error:  # noqa: BLE001 - the learner's code: one ❌ line, then the chain stops
            ok, measured, why = False, f"une erreur {type(error).__name__} : {error}", "relis la fonction en cause"
        if ok:
            print(f"✅ Ex B5.22 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B5.22 : point {k}, {name} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B5.22 : point {later} : corrige d'abord le point {k}.")
            return False
    print("✅ Ex B5.22 : les quatre diagnostics passent.")
    return True


def show_pipeline_22(prepare_fn, noisy_fn, sample_fn, title):
    """Eight samples of the pipeline with the perfect denoiser of its own forward process for the 500 digits."""
    try:
        data = torch.as_tensor(prepare_fn(RAW_22)).float()
        c1, c2 = coefficients_22(noisy_fn)
        eps_fn = wbd.ideal_denoiser(data, c1, c2)
        x_T = torch.randn((8, 1, 28, 28), generator=torch.Generator().manual_seed(220))
        out = torch.as_tensor(sample_fn(eps_fn, x_T, AB_22, torch.Generator().manual_seed(221))).float()
    except Exception as error:  # noqa: BLE001 - the learner's code
        print(f"⏳ Ex B5.22 : ton pipeline lève {type(error).__name__} ({error}) : rien à montrer.")
        return None
    show_images(out, ncol=8, size=1.2, suptitle=title)
    print(f"pixels générés : de {fr(float(out.min()), 2)} à {fr(float(out.max()), 2)}, moyenne {fr(float(out.mean()), 2)} "
          f"(images de MNIST dans [−1, 1] : moyenne {fr(float(wbd.to_signed(RAW_22).mean()), 2)})")
    return out'''

TODO_22 = r'''# The pipeline to repair IN PLACE (same names, same signatures): its author wanted images in the range of the noise,
# the closed form of the fiche and Algorithm 2 of Ho et al. (variance beta_t); four bugs slipped in.
def prepare_22(raw):
    """uint8 images (N, 28, 28) in [0, 255] -> a float32 tensor (N, 1, 28, 28) in the range of the noise."""
    return torch.as_tensor(np.asarray(raw), dtype=torch.float32)[:, None] / 255


def noisy_22(x0, t, eps, alpha_bars):
    """x_t of the closed form, for a batch x0 (N, 1, 28, 28), the step index t of each image ((N,) int64, from 0
    to T - 1) and the noise eps (N, 1, 28, 28)."""
    ab = torch.as_tensor(np.asarray(alpha_bars), dtype=torch.float32)[t - 1].view(-1, 1, 1, 1)   # step t of the paper
    return ab * x0 + (1 - ab).sqrt() * eps


def sample_22(eps_fn, x, alpha_bars, generator):
    """Ancestral DDPM sampling (Algorithm 2, variance beta_t) from x, the noise at index T - 1, down to the image;
    eps_fn(x, t) is the noise predicted at the index t; the draws z come from generator."""
    ab = np.asarray(alpha_bars, dtype=np.float64)
    betas = 1 - ab / np.concatenate([[1.0], ab[:-1]])
    for t in range(len(ab) - 1, -1, -1):
        eps = eps_fn(x, t)
        x = (x - betas[t] / math.sqrt(1 - ab[t]) * eps) / math.sqrt(1 - betas[t])
        x = x + math.sqrt(betas[t]) * torch.randn(x.shape, generator=generator)
    return x'''

SOLUTION_22 = r'''def prepare_22(raw):
    """uint8 images (N, 28, 28) in [0, 255] -> a float32 tensor (N, 1, 28, 28) in the range of the noise."""
    return torch.as_tensor(np.asarray(raw), dtype=torch.float32)[:, None] / 127.5 - 1      # 1. [-1, 1], like the noise


def noisy_22(x0, t, eps, alpha_bars):
    """x_t of the closed form, for a batch x0 (N, 1, 28, 28), the step index t of each image ((N,) int64, from 0
    to T - 1) and the noise eps (N, 1, 28, 28)."""
    ab = torch.as_tensor(np.asarray(alpha_bars), dtype=torch.float32)[t].view(-1, 1, 1, 1)   # 2. t is already an index
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps                                            # 3. sqrt(alpha_bar)


def sample_22(eps_fn, x, alpha_bars, generator):
    """Ancestral DDPM sampling (Algorithm 2, variance beta_t) from x, the noise at index T - 1, down to the image;
    eps_fn(x, t) is the noise predicted at the index t; the draws z come from generator."""
    ab = np.asarray(alpha_bars, dtype=np.float64)
    betas = 1 - ab / np.concatenate([[1.0], ab[:-1]])
    for t in range(len(ab) - 1, -1, -1):
        eps = eps_fn(x, t)
        x = (x - betas[t] / math.sqrt(1 - ab[t]) * eps) / math.sqrt(1 - betas[t])
        if t > 0:                                                                            # 4. no noise at index 0
            x = x + math.sqrt(betas[t]) * torch.randn(x.shape, generator=generator)
    return x'''

CHECK_22 = r'''repaired_22 = False
with checked("B5.22"):
    repaired_22 = diagnose_22(prepare_22, noisy_22, sample_22)'''

EXPERIMENT_22 = r'''# Your pipeline, as it is now, with a perfect denoiser for the 500 digits (it shows the symptom before the repair)
samples_22 = show_pipeline_22(prepare_22, noisy_22, sample_22,
                              "ton pipeline " + ("réparé" if repaired_22 else "tel quel") + ", avec un débruiteur parfait")'''

# --- 📦 B5.23 ---------------------------------------------------------------------------------------------------------
GIVEN_23 = r'''# The recipe of the training (wb.diffusion.unet_settings), without its EMA decay: you choose yours (RHO_23)
SETTINGS_23 = {k: v for k, v in wbd.unet_settings(FAST_MODE).items() if k != "ema"}
CONFIG_23 = {"n_steps": T_STEPS, "channels": list(SETTINGS_23["channels"]), "t_dim": SETTINGS_23["t_dim"],
             "n_classes": None}                        # the configuration of the U-Net (UNET_CLASS(**CONFIG_23))
DATA_23 = wbd.image_subset("mnist", SETTINGS_23["n_train"])   # the frozen training subset, in [-1, 1]
AB = wbd.alpha_bars(BETAS)                             # alpha_bar of the schedule of the notebook, indices 0 .. T - 1
N_UPDATES_23 = wbd.n_updates(len(DATA_23), SETTINGS_23)
REAL_MNIST = wbd.image_subset("mnist", 2000, split="test")    # 2 000 test digits: the reference of the distances
REAL_FEATS = wbd.features(CLF, REAL_MNIST.images)
X_FIXED = torch.randn((16, 1, 28, 28), generator=torch.Generator().manual_seed(2323))   # the fixed starting noise


def key_23(rho, source="learner"):
    """The key of the checkpoint of B5.23: the configuration, the recipe with your rho, the mode, who trained it."""
    return wbd.checkpoint_key(CONFIG_23, SETTINGS_23 | {"ema": float(rho)}, fast=FAST_MODE, source=source,
                              data=DATA_23)


def frechet_mnist(images):
    """The Fréchet distance between the features (CLF, 64 values) of images in [-1, 1] and those of REAL_FEATS, with
    your frechet_distance_from_features of ch. 27 (the reference when your gen_metrics.py does not exist)."""
    feats = wbd.features(CLF, torch.as_tensor(images).clamp(-1, 1))
    return float(mylearn.gen_metrics.frechet_distance_from_features(REAL_FEATS, feats))


print(f"recette : {SETTINGS_23['epochs']} epochs sur {len(DATA_23)} images, batches de {SETTINGS_23['batch_size']}, "
      f"AdamW au learning rate {SETTINGS_23['lr']} : {N_UPDATES_23} pas d'optimisation ; U-Net de canaux "
      f"{tuple(SETTINGS_23['channels'])} ({wbd.count_params(wbd.TinyUNet(**CONFIG_23)):_} paramètres), T = {T_STEPS}."
      .replace("_", " "))'''

TODO_23 = r'''RHO_23 = ...    # the decay rho of your EMA, chosen for N_UPDATES_23 optimizer steps (say why in your notes)


def ema_update_23(ema, model, rho):
    """theta_ema <- rho * theta_ema + (1 - rho) * theta for each parameter of ema, in place and without a graph; the
    buffers of model are copied into those of ema."""
    raise NotImplementedError("ema_update_23")


def train_23(model, ema, images, settings, generator, alpha_bars):
    """Train model, and its EMA copy ema, in place on images (float32 (N, 1, 28, 28) in [-1, 1]) with the recipe of
    the contract; returns {"loss": the list of the losses of every step (Python floats)}."""
    raise NotImplementedError("train_23")'''

SOLUTION_23 = r'''RHO_23 = 0.99 if FAST_MODE else 0.999   # 1 252 steps: 0.99 ** 1252 ≈ 3e-6 of the initial weights left, a memory of
                                         # about 100 steps (8 % of the training); full mode, 9 375 steps: 0.999


def ema_update_23(ema, model, rho):
    """theta_ema <- rho * theta_ema + (1 - rho) * theta for each parameter of ema, in place and without a graph; the
    buffers of model are copied into those of ema."""
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.mul_(rho).add_(p, alpha=1 - rho)
        for b_ema, b in zip(ema.buffers(), model.buffers()):
            b_ema.copy_(b)


def train_23(model, ema, images, settings, generator, alpha_bars):
    """Train model, and its EMA copy ema, in place on images (float32 (N, 1, 28, 28) in [-1, 1]) with the recipe of
    the contract; returns {"loss": the list of the losses of every step (Python floats)}."""
    ab = torch.tensor(np.asarray(alpha_bars), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    losses, bs = [], settings["batch_size"]
    model.train()
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(images), generator=generator)
        for i in range(0, len(images), bs):
            x0 = images[order[i:i + bs]]
            t = torch.randint(0, len(ab), (len(x0),), generator=generator)
            eps = torch.randn(x0.shape, generator=generator)
            a = ab[t].view(-1, 1, 1, 1)
            loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update_23(ema, model, settings["ema"])
            losses.append(loss.item())
    model.eval()
    ema.eval()
    return {"loss": losses}'''

CHECK_23 = r'''probes_ok_23 = False
with checked("B5.23"):
    torch.manual_seed(0)
    small_23 = wbd.TinyUNet(20, channels=(8, 8, 16), t_dim=16)            # a small U-Net of the contract
    with torch.no_grad():
        for p_ in small_23.parameters():
            p_.add_(0.1 * torch.randn(p_.shape))                            # weights far from the initialisation
    # 1. ema_update_23 against the formula, on two copies
    ema_a, target_23 = copy.deepcopy(small_23), copy.deepcopy(small_23)
    with torch.no_grad():
        for p_ in target_23.parameters():
            p_.add_(torch.randn(p_.shape))
    ids_23 = [id(p_) for p_ in ema_a.parameters()]
    ema_b = copy.deepcopy(ema_a)
    ema_update_23(ema_a, target_23, 0.75)
    wbd.ema_update(ema_b, target_23, 0.75)
    ema_ok_23 = all(torch.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(ema_a.parameters(), ema_b.parameters()))
    inplace_23 = [id(p_) for p_ in ema_a.parameters()] == ids_23 and all(p_.grad is None for p_ in ema_a.parameters())
    verdict("B5.23", ema_ok_23 and inplace_23, "ema_update_23 : 0,75 θ_EMA + 0,25 θ, sur place et sans gradient.",
            "ema_update_23(ema, model, 0.75) doit rendre chaque paramètre de ema égal à 0,75 × lui-même + 0,25 × celui de "
            "model, sur place (les mêmes objets Parameter) et sans gradient : " +
            ("les valeurs diffèrent." if not ema_ok_23 else "les paramètres ont été remplacés ou ont un gradient."))
    # 2. train_23 against the recipe of the contract: 6 digits in batches of 4 (a last batch of 2), 2 epochs
    probe_23 = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "ema": 0.5}
    ab20_23 = wbd.alpha_bars(np.linspace(1e-3, 0.2, 20))   # a short schedule
    m1, e1, m2, e2 = (copy.deepcopy(small_23) for _ in range(4))
    hist_23 = train_23(m1, e1, DATA_23.images[:6].clone(), probe_23, torch.Generator().manual_seed(5), ab20_23)
    want_23 = wbd.train_unet(m2, e2, DATA_23.images[:6].clone(), probe_23, torch.Generator().manual_seed(5), ab20_23)
    if not returned("B5.23", "train_23", hist_23):
        raise StopCheck
    losses_23 = hist_23.get("loss") if isinstance(hist_23, dict) else None
    if not (isinstance(losses_23, list) and len(losses_23) == 4 and all(isinstance(v, float) for v in losses_23)):
        print(f"❌ Ex B5.23 : train_23 doit rendre {{'loss': [...]}}, une loss (float Python) par pas : 2 epochs de 6 "
              f"images en batches de 4 font 4 pas ; reçu {hist_23!r:.120}.")
        raise StopCheck
    loss_ok_23 = np.allclose(losses_23, want_23["loss"], rtol=1e-4, atol=1e-6)
    weights_ok_23 = all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(m1.parameters(), m2.parameters()))
    emaw_ok_23 = all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(e1.parameters(), e2.parameters()))
    if loss_ok_23 and weights_ok_23 and emaw_ok_23:
        print("✅ Ex B5.23 : 4 pas de train_23 (2 epochs, un dernier batch de 2 images) : les loss, les poids et la copie "
              "EMA de la recette du contrat.")
    elif not math.isclose(losses_23[0], want_23["loss"][0], rel_tol=1e-4):
        print(f"❌ Ex B5.23 : la loss du 1ᵉʳ pas, aux mêmes poids, vaut {fr(losses_23[0], 5)} au lieu de "
              f"{fr(want_23['loss'][0], 5)} : vérifie l'ordre des tirages (l'ordre des images, les pas, le bruit), la "
              "forme fermée et la cible.")
    elif not loss_ok_23:
        print("❌ Ex B5.23 : la 1ʳᵉ loss est juste, pas les suivantes : vérifie l'optimiseur (AdamW, son learning rate), "
              "les batches d'une epoch (le dernier fait 2 images) et l'ordre des images à chaque epoch.")
    elif not weights_ok_23:
        print("❌ Ex B5.23 : les loss sont justes, mais pas les poids après le dernier pas : un pas d'optimisation manque-t-il ?")
    else:
        print("❌ Ex B5.23 : les poids entraînés sont justes, pas la copie EMA : est-elle mise à jour après chaque pas, avec "
              "settings['ema'] ?")
    # 3. your decay rho
    rho_ok_23 = False
    if not filled(RHO_23):
        print("⏳ Ex B5.23 : choisis RHO_23.")
    elif not (isinstance(RHO_23, (int, float)) and 0 < RHO_23 < 1):
        print(f"❌ Ex B5.23 : RHO_23 est un nombre strictement entre 0 et 1 ; reçu {RHO_23!r}.")
    else:
        left_23, memory_23 = RHO_23 ** N_UPDATES_23, 1 / (1 - RHO_23)
        rho_ok_23 = left_23 <= 0.01 and memory_23 >= 20
        verdict("B5.23", rho_ok_23,
                f"ρ = {RHO_23} : après {N_UPDATES_23} pas, il reste {fr(100 * left_23, 4)} % des poids de départ dans la "
                f"copie EMA, qui moyenne environ {memory_23:.0f} pas.",
                f"ρ = {RHO_23} : après {N_UPDATES_23} pas, il resterait {fr(100 * left_23, 2)} % des poids de départ dans la "
                f"copie EMA (au plus 1 %), et elle moyennerait environ {memory_23:.0f} pas (au moins 20).")
    probes_ok_23 = ema_ok_23 and inplace_23 and loss_ok_23 and weights_ok_23 and emaw_ok_23 and rho_ok_23'''

EXPERIMENT_23 = r'''# Train your U-Net (UNET_CLASS) on DATA_23 and save it ONCE, with its EMA copy and its history (about 2 min on CPU in
# fast mode; 🚀 in full mode)
model_23 = ema_23 = history_23 = None
if not globals().get("probes_ok_23"):
    print("⏳ Ex B5.23 : l'entraînement attend les contrôles de la cellule précédente (et RHO_23).")
else:
    with checked("B5.23"):
        KEY_23 = key_23(RHO_23, globals().get("CKPT_SOURCE_23", "learner"))
        if wbd.delete_checkpoint(KEY_23):                    # a new training: the old model goes, even if this run fails
            print("ℹ️ l'ancienne sauvegarde de ton U-Net est effacée : la suite du chapitre relira celui-ci.")
        torch.manual_seed(SETTINGS_23["seed"])
        net_23 = UNET_CLASS(**CONFIG_23)
        ema_net_23 = copy.deepcopy(net_23)
        started_23 = time.perf_counter()
        hist_23 = train_23(net_23, ema_net_23, DATA_23.images, SETTINGS_23 | {"ema": float(RHO_23)},
                           torch.Generator().manual_seed(SETTINGS_23["seed"] + 1), AB)
        seconds_23 = time.perf_counter() - started_23
        if returned("B5.23", "train_23", hist_23):
            model_23, ema_23, history_23 = net_23, ema_net_23, hist_23
            path_23 = wbd.save_checkpoint(KEY_23, model_23, ema=ema_23, history=history_23,
                                          extra={"seconds": seconds_23})
            print(f"{len(history_23['loss'])} pas en {seconds_23:.0f} s ; sauvegardé : {path_23.name}, dans "
                  "wbd.checkpoints_dir()")'''

EXPERIMENT_23B = r'''# Judge the training on samples from a FIXED noise (the same X_FIXED every time), not on the loss
if model_23 is None:
    print("⏳ Ex B5.23 : le jugement attend l'entraînement de la cellule précédente.")
else:
    with checked("B5.23"):
        ckpt_23 = wbd.load_checkpoint(KEY_23)
        verdict("B5.23", ckpt_23 is not None, f"la sauvegarde se relit (sauvegardée le {ckpt_23['saved_at'] if ckpt_23 else ''}).",
                "la sauvegarde ne se relit pas.")
        losses_23 = np.array(history_23["loss"])
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(losses_23, lw=0.5, alpha=0.4, label="loss de chaque pas")
        ax.plot(np.arange(99, len(losses_23)), np.convolve(losses_23, np.ones(100) / 100, mode="valid"), lw=2,
                label="moyenne glissante sur 100 pas")
        ax.set_yscale("log")
        ax.set_xlabel("pas d'optimisation")
        ax.set_ylabel("MSE sur le bruit")
        ax.legend()
        plt.show()
        print(f"loss moyenne des 100 premiers pas {fr(losses_23[:100].mean(), 3)}, des 100 derniers {fr(losses_23[-100:].mean(), 3)}")
        for name_23, net in (("copie EMA", ema_23), ("poids bruts", model_23)):
            show_images(wbd.sample_ddim(net, X_FIXED, AB, 50), ncol=16, size=0.75,
                        suptitle=f"{name_23} : 16 échantillons (DDIM, 50 pas) du même bruit X_FIXED")
        x256_23 = torch.randn((256, 1, 28, 28), generator=torch.Generator().manual_seed(2324))
        FD_23 = frechet_mnist(wbd.sample_ddim(ema_23, x256_23, AB, 50))
        print(f"distance de Fréchet de 256 échantillons de la copie EMA (DDIM, 50 pas) aux 2 000 chiffres de test : "
              f"{fr(FD_23, 1)} (deux lots de 256 vrais chiffres : environ {fr(FLOOR_23, 0)})")
        verdict("B5.23", FD_23 < GOAL_23,
                f"la copie EMA génère des formes de chiffres : distance {fr(FD_23, 1)}, sous {GOAL_23:.0f} (la référence : de "
                f"{fr(REF_23[0], 0)} à {fr(REF_23[1], 0)} selon la graine).",
                f"distance {fr(FD_23, 1)} : attendu moins de {GOAL_23:.0f} (référence de {fr(REF_23[0], 0)} à "
                f"{fr(REF_23[1], 0)}) ; regarde les échantillons, puis relance les contrôles de la cellule de vérification.")'''

# --- Part C, second half: the trained model ---------------------------------------------------------------------------
TOOLS_C2 = r'''# The model of the rest of part C: your U-Net trained by B5.23, read from its checkpoint (never retrained here)
REFERENCE_FALLBACK = False   # True: if your model of B5.23 is missing, use the reference U-Net of wb.diffusion (trained
                             # once with the recipe of B5.23, about 2 min on CPU in fast mode, then reloaded)
MODEL = CKPT = None
if isinstance(globals().get("RHO_23"), (int, float)) and 0 < RHO_23 < 1:
    CKPT = wbd.load_checkpoint(key_23(RHO_23, globals().get("CKPT_SOURCE_23", "learner")))
if CKPT is not None:
    try:
        MODEL = wbd.build_from_checkpoint(UNET_CLASS, CKPT, which="ema")
        where_c2 = ""
    except Exception:  # noqa: BLE001 - UNET_CLASS not the contract in this session (a restarted kernel)
        MODEL = wbd.build_from_checkpoint(wbd.TinyUNet, CKPT, which="ema")
        where_c2 = " dans la classe TinyUNet de wb.diffusion (mêmes noms, mêmes poids)"
    print(f"ℹ️ la copie EMA de ton U-Net de B5.23, sauvegardée le {CKPT['saved_at']}, relue de {CKPT['path'].name}"
          f"{where_c2}.")
elif REFERENCE_FALLBACK:
    MODEL, CKPT = wbd.reference_model(FAST_MODE)
    print(f"ℹ️ ton U-Net de B5.23 manque : la copie EMA du U-Net de RÉFÉRENCE (sauvegardé le {CKPT['saved_at']}) le "
          "remplace dans la suite de la partie C.")
else:
    print("⏳ aucune sauvegarde de ton U-Net de B5.23 (" + ("mode rapide" if FAST_MODE else "mode complet") + ") : "
          "exécute les cellules de B5.23 (RHO_23 compris), ou mets REFERENCE_FALLBACK = True dans cette cellule pour "
          "prendre le U-Net de référence (2 min la première fois). Sur Colab, la sauvegarde s'efface avec la session.")


def need_model(ex_id):
    """Stop a check cell with ⏳ while no trained U-Net is loaded."""
    if MODEL is None:
        print(f"⏳ Ex {ex_id} : cet exercice attend ton U-Net entraîné (B5.23), ou REFERENCE_FALLBACK = True dans la "
              "cellule des outils.")
        raise StopCheck


class PassCounter:
    """Count the passes of a network: every call of ``model`` (the batch size of each call) while active."""

    def __init__(self, model):
        self.model, self.calls, self.images = model, 0, 0

    def __enter__(self):
        def hook(module, args):
            self.calls += 1
            self.images += int(args[0].shape[0])
        self.handle = self.model.register_forward_pre_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False'''

# --- 📦 B5.24 ---------------------------------------------------------------------------------------------------------
TODO_24 = r'''def sample_ddpm_24(model, x_T, schedule, rng, clip_x0=1.0):
    """Ancestral DDPM sampling with YOUR ddpm_step, from x_T (a tensor (N, 1, 28, 28), the noise at index T - 1) down
    to the images; a float32 tensor (N, 1, 28, 28)."""
    raise NotImplementedError("sample_ddpm_24")


def sample_ddim_24(model, x_T, alpha_bars, n_steps, clip_x0=1.0):
    """Deterministic DDIM with YOUR ddim_step on wbd.ddim_timesteps(T, n_steps), then to -1, x̂0 clipped and the
    noise recomputed from it; a float32 tensor (N, 1, 28, 28)."""
    raise NotImplementedError("sample_ddim_24")'''

SOLUTION_24 = r'''def _eps_24(model, x, t):
    """The network's noise at the index t for the float64 array x, as a float64 array (float32 inside, no graph)."""
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32), torch.full((len(x),), t, dtype=torch.long))
    return out.double().numpy()


def sample_ddpm_24(model, x_T, schedule, rng, clip_x0=1.0):
    """Ancestral DDPM sampling with YOUR ddpm_step, from x_T (a tensor (N, 1, 28, 28), the noise at index T - 1) down
    to the images; a float32 tensor (N, 1, 28, 28)."""
    x = x_T.double().numpy()
    for t in range(len(schedule["betas"]) - 1, -1, -1):
        x = mylearn.diffusion.ddpm_step(x, t, _eps_24(model, x, t), schedule, rng=rng, clip_x0=clip_x0)
    return torch.as_tensor(x, dtype=torch.float32)


def sample_ddim_24(model, x_T, alpha_bars, n_steps, clip_x0=1.0):
    """Deterministic DDIM with YOUR ddim_step on wbd.ddim_timesteps(T, n_steps), then to -1, x̂0 clipped and the
    noise recomputed from it; a float32 tensor (N, 1, 28, 28)."""
    D = mylearn.diffusion
    x = x_T.double().numpy()
    steps = [int(t) for t in wbd.ddim_timesteps(len(alpha_bars), n_steps)] + [-1]
    for t, t_prev in zip(steps[:-1], steps[1:]):
        eps = _eps_24(model, x, t)
        if clip_x0 is not None:
            x0 = np.clip(D.predict_x0_from_eps(x, np.full(len(x), t), eps, alpha_bars), -clip_x0, clip_x0)
            eps = (x - math.sqrt(alpha_bars[t]) * x0) / math.sqrt(1 - alpha_bars[t])
        x = D.ddim_step(x, t, t_prev, eps, alpha_bars)
    return torch.as_tensor(x, dtype=torch.float32)'''

CHECK_24 = r'''samplers_ok_24 = False
with checked("B5.24"):
    need_model("B5.24")
    SCHED = mylearn.diffusion.noise_schedule(BETAS)          # your noise_schedule (B5.12)
    x4_24 = torch.randn((4, 1, 28, 28), generator=torch.Generator().manual_seed(24))
    copy4_24 = x4_24.clone()
    ok_24 = True
    for label_24, call_24, want_24, passes_24 in (
            ("sample_ddim_24, 10 pas", lambda: sample_ddim_24(MODEL, x4_24, AB, 10),
             lambda: wbd.sample_ddim(MODEL, x4_24, AB, 10), 10),
            ("sample_ddim_24, 3 pas, sans écrêtage", lambda: sample_ddim_24(MODEL, x4_24, AB, 3, clip_x0=None),
             lambda: wbd.sample_ddim(MODEL, x4_24, AB, 3, clip_x0=None), 3),
            (f"sample_ddpm_24, {T_STEPS} pas", lambda: sample_ddpm_24(MODEL, x4_24, SCHED, np.random.default_rng(3)),
             lambda: wbd.sample_ddpm(MODEL, x4_24, AB, np.random.default_rng(3)), T_STEPS)):
        with PassCounter(MODEL) as count_24:
            got_24 = call_24()
        if not returned("B5.24", label_24.split(",")[0], got_24):
            ok_24 = False
            continue
        if not (torch.is_tensor(got_24) and got_24.dtype == torch.float32 and tuple(got_24.shape) == (4, 1, 28, 28)):
            print(f"❌ Ex B5.24 : {label_24} doit rendre un tenseur float32 (4, 1, 28, 28) ; reçu {type(got_24).__name__} "
                  f"{getattr(got_24, 'dtype', '')} {tuple(getattr(got_24, 'shape', ()))}.")
            ok_24 = False
            continue
        ok_24 &= same("B5.24", f"{label_24} contre l'échantillonneur du contrat (même bruit de départ)", got_24, want_24(),
                      rtol=1e-4, atol=1e-4)
        ok_24 &= count_24.calls == passes_24
        verdict("B5.24", count_24.calls == passes_24, f"{label_24} : {count_24.calls} passes du réseau.",
                f"{label_24} : {count_24.calls} passes du réseau, au lieu de {passes_24}.")
    verdict("B5.24", torch.equal(x4_24, copy4_24), "le bruit de départ x_T n'a pas changé.",
            "tes échantillonneurs ont modifié x_T, le bruit de départ.")
    samplers_ok_24 = bool(ok_24) and torch.equal(x4_24, copy4_24)'''

EXPERIMENT_24 = r'''# The same 32 starting noises, sampled by DDPM (all the T steps) and by DDIM (50 steps): images and time
if not samplers_ok_24:
    print("⏳ Ex B5.24 : la comparaison attend tes deux échantillonneurs (cellule précédente sans ❌).")
else:
    with checked("B5.24"):
        x32_24 = torch.randn((32, 1, 28, 28), generator=torch.Generator().manual_seed(240))
        RESULTS_24 = {}
        for name_24, run_24 in ((f"DDPM, {T_STEPS} pas", lambda: sample_ddpm_24(MODEL, x32_24, SCHED, np.random.default_rng(241))),
                                ("DDIM, 50 pas", lambda: sample_ddim_24(MODEL, x32_24, AB, 50))):
            start_24 = time.perf_counter()
            out_24 = run_24()
            seconds_24 = time.perf_counter() - start_24
            with torch.no_grad():
                proba_24 = torch.softmax(CLF(out_24.clamp(-1, 1)), dim=1).max(dim=1).values
            RESULTS_24[name_24] = {"images": out_24, "seconds": seconds_24, "confidence": float(proba_24.mean()),
                                   "sure": float((proba_24 > 0.9).float().mean())}
            show_images(out_24[:16], ncol=16, size=0.75, suptitle=f"{name_24} : les 16 premiers des 32 échantillons")
        print(f"{'échantillonneur':<16} {'secondes':>9} {'s / image':>10} {'confiance moyenne':>18} {'reconnus à 90 %':>16}")
        for name_24, r in RESULTS_24.items():
            print(f"{name_24:<16} {fr(r['seconds'], 1):>9} {fr(r['seconds'] / 32, 3):>10} {fr(r['confidence'], 3):>18} "
                  f"{fr(100 * r['sure'], 0) + ' %':>16}")
        print("(des mesures, pas des réponses : les durées dépendent de la machine ; la confiance est la probabilité que "
              "CLF donne à sa classe préférée, en moyenne sur les 32 images)")'''

# --- 🔬 B5.25 ---------------------------------------------------------------------------------------------------------
GIVEN_25 = r'''STEPS_25 = (5, 10, 20, 50, 100)                       # numbers of DDIM steps of the curve
N_25 = 128 if FAST_MODE else 1000                       # images per point (the same starting noise for every point)
for n_25 in (5, 10):
    print(f"sous-séquence de {n_25} pas, de l'indice {T_STEPS - 1} à 0 : {wbd.ddim_timesteps(T_STEPS, n_25).tolist()}")
# The full protocol (tools/chapters/history_chB5.py): the recipe of B5.23 with the linear schedule (L, the notebook's)
# and with the cosine schedule (C), «N_SEEDS_25» seeds each, the distance of 500 DDIM samples for each number of steps
FULL_25 = «FULL_25»'''

TODO_25 = r'''def fd_curve_25(model, alpha_bars, steps_list, x_T):
    """For each number of steps n of steps_list: (frechet_mnist of sample_ddim_24(model, x_T, alpha_bars, n), the
    seconds of the sampling); a list of pairs of floats."""
    raise NotImplementedError("fd_curve_25")


enough_25 = ...     # a) full protocol, linear schedule: the smallest number of steps of STEPS_25 from which the distance
                    #    is at most «RATIO_TEXT_25» times its distance at 100 steps, for every seed
better5_25 = ...    # b) full protocol, at 5 steps: which schedule has the smaller distance for every seed, "L" or "C"?'''

SOLUTION_25 = r'''def fd_curve_25(model, alpha_bars, steps_list, x_T):
    """For each number of steps n of steps_list: (frechet_mnist of sample_ddim_24(model, x_T, alpha_bars, n), the
    seconds of the sampling); a list of pairs of floats."""
    out = []
    for n in steps_list:
        start = time.perf_counter()
        images = sample_ddim_24(model, x_T, alpha_bars, n)
        seconds = time.perf_counter() - start
        out.append((frechet_mnist(images), seconds))
    return out'''

CHECK_25 = r'''curve_ok_25 = False
with checked("B5.25"):
    need_model("B5.25")
    x8_25 = torch.randn((80, 1, 28, 28), generator=torch.Generator().manual_seed(25))
    got_25 = fd_curve_25(MODEL, AB, [2, 4], x8_25)
    if returned("B5.25", "fd_curve_25", got_25):
        try:
            pairs_25 = [(float(a), float(b)) for a, b in got_25]
        except (TypeError, ValueError):
            pairs_25 = None
        if pairs_25 is None or len(pairs_25) != 2:
            print(f"❌ Ex B5.25 : fd_curve_25 doit rendre une liste de couples (distance, secondes), un par nombre de pas ; "
                  f"reçu {got_25!r:.100}.")
        else:
            want_25 = [frechet_mnist(wbd.sample_ddim(MODEL, x8_25, AB, n)) for n in (2, 4)]
            curve_ok_25 = same("B5.25", "les distances de fd_curve_25 (80 images, 2 et 4 pas) contre DDIM du contrat",
                               [a for a, _ in pairs_25], want_25, rtol=1e-3, atol=1e-3)
            verdict("B5.25", all(b > 0 for _, b in pairs_25), "les durées sont positives.", "une durée n'est pas positive.")
with wb.attempt("B5.25"):
    wb.check("B5.25a", enough_25)
with wb.attempt("B5.25"):
    wb.check("B5.25b", better5_25)'''

EXPERIMENT_25 = r'''# Your linear model: distance against time for each number of DDIM steps, over the bands of the full protocol
CURVE_25 = None
if not curve_ok_25:
    print("⏳ Ex B5.25 : la courbe attend fd_curve_25 (cellule précédente sans ❌).")
else:
    with checked("B5.25"):
        x_25 = torch.randn((N_25, 1, 28, 28), generator=torch.Generator().manual_seed(250))
        CURVE_25 = fd_curve_25(MODEL, AB, STEPS_25, x_25)
        curves_25 = {"L, ton modèle": CURVE_25}
        if not FAST_MODE:                                   # 🚀 full mode: a second model, with the cosine schedule
            torch.manual_seed(SETTINGS_23["seed"])
            cos_net_25 = UNET_CLASS(**CONFIG_23)
            cos_ema_25 = copy.deepcopy(cos_net_25)
            ab_cos_25 = wbd.alpha_bars(wbd.schedule_betas("cosine", T_STEPS))
            train_23(cos_net_25, cos_ema_25, DATA_23.images, SETTINGS_23 | {"ema": float(RHO_23)},
                     torch.Generator().manual_seed(SETTINGS_23["seed"] + 1), ab_cos_25)
            curves_25["C, ton modèle"] = fd_curve_25(cos_ema_25, ab_cos_25, STEPS_25, x_25)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for (name_25, curve), color in zip(curves_25.items(), ("C0", "C1")):
            ax1.plot([s for _, s in curve], [d for d, _ in curve], "o-", color=color, label=name_25)
            for (d, s), n in zip(curve, STEPS_25):
                ax1.annotate(str(n), (s, d), fontsize=7)
        for key_25, color in (("L", "C0"), ("C", "C1")):
            rows_25 = np.array(FULL_25[key_25])                 # (seeds, steps)
            ax2.fill_between(STEPS_25, rows_25.min(axis=0), rows_25.max(axis=0), color=color, alpha=0.2)
            ax2.plot(STEPS_25, np.median(rows_25, axis=0), "o-", color=color, label=f"{key_25}, protocole complet")
        if not FAST_MODE:
            ax2.plot(STEPS_25, [d for d, _ in curves_25["L, ton modèle"]], "x--", color="C0", label="L, ton modèle")
        for ax, xlabel in ((ax1, "secondes d'échantillonnage"), (ax2, "nombre de pas DDIM")):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("distance de Fréchet")
            ax.legend(fontsize=8)
        plt.show()
        print("pas :", "  ".join(f"{n:>6}" for n in STEPS_25))
        print(f"ta distance ({N_25} images) :", "  ".join(f"{fr(d, 1):>6}" for d, _ in CURVE_25))
        for key_25 in ("L", "C"):
            print(f"protocole complet {key_25} (médiane, 500 images) :",
                  "  ".join(f"{fr(v, 1):>6}" for v in np.median(np.array(FULL_25[key_25]), axis=0)))'''

# --- 🔮 B5.26 and 📦 B5.27: guidance on Fashion-MNIST -------------------------------------------------------------------
GIVEN_26 = r'''W_26 = (0, 1, 3, 7)                                        # the guidance scales (convention of Ho and Salimans)'''

TODO_26 = r'''fd7_26 = ...    # a) at w = 7, is the Fréchet distance SMALLER than at w = 0 for every seed? True or False
acc0_26 = ...   # b) at w = 0 (the conditional model, no guidance), does the classifier recognise the requested class in
                #    MORE than 70 % of the images, for every seed? True or False
div0_26 = ...   # c) at w = 0, is the diversity of the generated images LARGER than that of the real test images, for
                #    every seed? True or False'''

EXPERIMENT_26 = r'''TABLE_26 = wbd.precomputed("guidance_26")        # the full protocol, read here only (history_chB5.py)
print(f"protocole : {TABLE_26['n_seeds']} graines, {TABLE_26['n_images']} images par graine (autant par classe), DDIM "
      f"{TABLE_26['steps']} pas ; médiane, et de la plus petite à la plus grande valeur des graines")
print(f"{'w':>3} {'distance de Fréchet':>26} {'classe reconnue':>22} {'diversité':>20} {'pixels saturés':>20}")
for w_26 in W_26:
    row_26 = TABLE_26["by_w"][str(w_26)]
    cells_26 = []
    for key_26, scale_26, digits_26 in (("fd", 1, 1), ("acc", 100, 0), ("div", 1, 3), ("sat", 100, 1)):
        v_26 = np.array(row_26[key_26]) * scale_26
        cells_26.append(f"{fr(np.median(v_26), digits_26)} ({fr(v_26.min(), digits_26)}–{fr(v_26.max(), digits_26)})")
    print(f"{w_26:>3} {cells_26[0]:>26} {cells_26[1] + ' %':>22} {cells_26[2]:>20} {cells_26[3] + ' %':>20}")
print(f"vraies images de test : diversité {fr(TABLE_26['real']['div'], 3)}, pixels saturés "
      f"{fr(100 * TABLE_26['real']['sat'], 1)} % ; plancher de la distance (200 vraies images) {fr(TABLE_26['real']['fd'], 1)}")
wb.check("B5.26a", fd7_26)
wb.check("B5.26b", acc0_26)
wb.check("B5.26c", div0_26)'''

GIVEN_27 = r'''# Fashion-MNIST: the training subset, the classifier of the measures (the CNN of ch. 27, trained on Fashion-MNIST),
# 2 000 test images for the reference, the recipe (the one of B5.23 with the classes, EMA decay fixed for its steps)
FASHION = wbd.image_subset("fashion", wbd.n_train(FAST_MODE))
CLF_F = wbd.classifier("fashion")
REAL_F = wbd.image_subset("fashion", 2000, split="test")
REAL_F_FEATS = wbd.features(CLF_F, REAL_F.images)
SETTINGS_27 = SETTINGS_23 | {"epochs": 2 if FAST_MODE else 20, "ema": 0.99 if FAST_MODE else 0.999, "p_uncond": 0.1,
                             "seed": 2700}
CONFIG_27 = CONFIG_23 | {"n_classes": 10}
PER_CLASS_27 = 20 if FAST_MODE else 100
Y_27 = torch.arange(10).repeat_interleave(PER_CLASS_27)     # the requested classes: PER_CLASS_27 images per class
NAMES_F = ["T-shirt", "pantalon", "pull", "robe", "manteau", "sandale", "chemise", "basket", "sac", "bottine"]


def key_27(source="learner"):
    """The key of the checkpoint of the conditional U-Net of B5.27."""
    return wbd.checkpoint_key(CONFIG_27, SETTINGS_27, fast=FAST_MODE, source=source, data=FASHION)


def guidance_measures(images, y):
    """The four measures of B5.26 for images in [-1, 1] generated for the classes y: Fréchet distance to REAL_F
    (features of CLF_F), share of images whose predicted class is y, diversity (the mean, over the classes, of the
    mean standard deviation of the features), share of saturated pixels (|x| > 0.98)."""
    images = torch.as_tensor(images).clamp(-1, 1)
    feats = wbd.features(CLF_F, images)
    yy = np.asarray(y)
    return {"fd": float(mylearn.gen_metrics.frechet_distance_from_features(REAL_F_FEATS, feats)),
            "acc": float(np.mean(wbd.predict(CLF_F, images) == yy)),
            "div": float(np.mean([feats[yy == c].std(axis=0).mean() for c in np.unique(yy)])),
            "sat": float((images.abs() > 0.98).float().mean())}


print(f"{len(FASHION)} images de Fashion-MNIST ; le classifieur des mesures reconnaît "
      f"{fr(100 * np.mean(wbd.predict(CLF_F, REAL_F.images) == REAL_F.labels.numpy()), 1)} % des 2 000 images de test.")'''

TODO_27 = r'''def train_cond_27(model, ema, images, labels, settings, generator, alpha_bars):
    """train_23 with the classes: after the noise, the dropout torch.rand(b, generator=generator) < settings["p_uncond"]
    replaces the label by the empty class model.n_classes; the model receives (x_t, t, y). Returns {"loss": [...]}."""
    raise NotImplementedError("train_cond_27")


def guided_eps_27(model, x, t, y, w):
    """The guided noise at the step index t (an int): (1 + w) eps(x, t, y) - w eps(x, t, ∅), the two predictions in
    ONE call of the model on 2 N images; for w = 0, one call on the N images. No graph."""
    raise NotImplementedError("guided_eps_27")


def sample_cond_27(model, x_T, y, w, alpha_bars, n_steps):
    """sample_ddim_24 (x̂0 clipped to [-1, 1]) with guided_eps_27(model, x, t, y, w) as the network's noise."""
    raise NotImplementedError("sample_cond_27")'''

SOLUTION_27 = r'''def train_cond_27(model, ema, images, labels, settings, generator, alpha_bars):
    """train_23 with the classes: after the noise, the dropout torch.rand(b, generator=generator) < settings["p_uncond"]
    replaces the label by the empty class model.n_classes; the model receives (x_t, t, y). Returns {"loss": [...]}."""
    ab = torch.tensor(np.asarray(alpha_bars), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    losses, bs = [], settings["batch_size"]
    model.train()
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(images), generator=generator)
        for i in range(0, len(images), bs):
            idx = order[i:i + bs]
            x0, y = images[idx], labels[idx]
            t = torch.randint(0, len(ab), (len(x0),), generator=generator)
            eps = torch.randn(x0.shape, generator=generator)
            drop = torch.rand(len(x0), generator=generator) < settings["p_uncond"]
            y = torch.where(drop, torch.full_like(y, model.n_classes), y)          # the empty class
            a = ab[t].view(-1, 1, 1, 1)
            loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, y), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update_23(ema, model, settings["ema"])
            losses.append(loss.item())
    model.eval()
    ema.eval()
    return {"loss": losses}


def guided_eps_27(model, x, t, y, w):
    """The guided noise at the step index t (an int): (1 + w) eps(x, t, y) - w eps(x, t, ∅), the two predictions in
    ONE call of the model on 2 N images; for w = 0, one call on the N images. No graph."""
    tt = torch.full((len(x),), t, dtype=torch.long)
    with torch.no_grad():
        if w == 0:
            return model(x, tt, y)
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, model.n_classes)]))
    cond, uncond = both.chunk(2)
    return (1 + w) * cond - w * uncond


def sample_cond_27(model, x_T, y, w, alpha_bars, n_steps):
    """sample_ddim_24 (x̂0 clipped to [-1, 1]) with guided_eps_27(model, x, t, y, w) as the network's noise."""
    D = mylearn.diffusion
    x = x_T.double().numpy()
    steps = [int(t) for t in wbd.ddim_timesteps(len(alpha_bars), n_steps)] + [-1]
    for t, t_prev in zip(steps[:-1], steps[1:]):
        eps = guided_eps_27(model, torch.as_tensor(x, dtype=torch.float32), t, y, w).double().numpy()
        x0 = np.clip(D.predict_x0_from_eps(x, np.full(len(x), t), eps, alpha_bars), -1, 1)
        eps = (x - math.sqrt(alpha_bars[t]) * x0) / math.sqrt(1 - alpha_bars[t])
        x = D.ddim_step(x, t, t_prev, eps, alpha_bars)
    return torch.as_tensor(x, dtype=torch.float32)'''

CHECK_27 = r'''cond_ok_27 = False
with checked("B5.27"):
    torch.manual_seed(1)
    small_27 = wbd.TinyUNet(20, channels=(8, 8, 16), t_dim=16, n_classes=3)
    with torch.no_grad():
        for p_ in small_27.parameters():
            p_.add_(0.1 * torch.randn(p_.shape))
    ab20_27 = wbd.alpha_bars(np.linspace(1e-3, 0.2, 20))   # a short schedule
    probe_27 = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "ema": 0.5, "p_uncond": 0.4}
    imgs_27, labels_27 = FASHION.images[:6].clone(), FASHION.labels[:6] % 3
    m1, e1, m2, e2 = (copy.deepcopy(small_27) for _ in range(4))
    hist_27 = train_cond_27(m1, e1, imgs_27, labels_27.clone(), probe_27, torch.Generator().manual_seed(6), ab20_27)
    want_27 = wbd.train_unet(m2, e2, imgs_27, probe_27, torch.Generator().manual_seed(6), ab20_27, labels=labels_27)
    ok_27 = returned("B5.27", "train_cond_27", hist_27) and isinstance(hist_27, dict)
    if ok_27:
        ok_27 = same("B5.27", "les 4 loss de train_cond_27 contre la recette du contrat (mêmes tirages, abandon du label "
                     "compris)", hist_27.get("loss"), want_27["loss"], rtol=1e-4, atol=1e-6)
        ok_27 &= all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(e1.parameters(), e2.parameters()))
        verdict("B5.27", ok_27, "les poids et la copie EMA après 4 pas : ceux de la recette.",
                "les poids ou la copie EMA après 4 pas ne sont pas ceux de la recette.")
    m2.eval()
    x5_27, y5_27 = torch.randn((5, 1, 28, 28), generator=torch.Generator().manual_seed(7)), torch.tensor([0, 1, 2, 0, 3])
    for w_27, calls_27, rows_27 in ((0.0, 1, 5), (2.5, 1, 10)):
        with PassCounter(m2) as count_27:
            got_27 = guided_eps_27(m2, x5_27, 13, y5_27, w_27)
        ok_27 &= same("B5.27", f"guided_eps_27 à w = {fr(w_27, 1)} (dont une classe vide) contre la formule", got_27,
                      wbd.predict_eps(m2, x5_27, 13, y5_27, w_27), rtol=1e-4, atol=1e-5)
        ok_27 &= (count_27.calls, count_27.images) == (calls_27, rows_27)
        verdict("B5.27", (count_27.calls, count_27.images) == (calls_27, rows_27),
                f"à w = {fr(w_27, 1)} : {calls_27} appel du réseau sur {rows_27} images.",
                f"à w = {fr(w_27, 1)}, attendu {calls_27} appel du réseau sur {rows_27} images ; vu {count_27.calls} "
                f"appel(s) sur {count_27.images} images.")
    ok_27 &= same("B5.27", "sample_cond_27 (5 pas, w = 2) contre DDIM guidé du contrat",
                  sample_cond_27(m2, x5_27, y5_27, 2.0, ab20_27, 5),
                  wbd.sample_ddim(m2, x5_27, ab20_27, 5, y=y5_27, w=2.0), rtol=1e-4, atol=1e-4)
    cond_ok_27 = bool(ok_27)'''

EXPERIMENT_27 = r'''# Train your conditional U-Net on Fashion-MNIST (about 1 min on CPU in fast mode; 🚀 in full mode), save it once
MODEL_27 = None
if not cond_ok_27:
    print("⏳ Ex B5.27 : l'entraînement attend les contrôles de la cellule précédente.")
else:
    with checked("B5.27"):
        KEY_27 = key_27(globals().get("CKPT_SOURCE_23", "learner"))
        wbd.delete_checkpoint(KEY_27)
        torch.manual_seed(SETTINGS_27["seed"])
        net_27 = UNET_CLASS(**CONFIG_27)
        ema_27 = copy.deepcopy(net_27)
        start_27 = time.perf_counter()
        hist_27 = train_cond_27(net_27, ema_27, FASHION.images, FASHION.labels, SETTINGS_27,
                                torch.Generator().manual_seed(SETTINGS_27["seed"] + 1), AB)
        print(f"{len(hist_27['loss'])} pas en {time.perf_counter() - start_27:.0f} s")
        wbd.save_checkpoint(KEY_27, net_27, ema=ema_27, history=hist_27)
        MODEL_27 = ema_27
        del net_27
        wbd.trim_memory()'''

EXPERIMENT_27B = r'''# Your model, for w = 0, 1, 3, 7: the same starting noise and the same classes for every w (DDIM, 20 steps)
MEASURES_27 = None
if MODEL_27 is None:
    print("⏳ Ex B5.27 : les mesures attendent ton modèle entraîné (cellule précédente).")
else:
    with checked("B5.27"):
        x_27 = torch.randn((len(Y_27), 1, 28, 28), generator=torch.Generator().manual_seed(271))
        MEASURES_27, grid_27 = {}, []
        for w_27 in W_26:
            out_27 = sample_cond_27(MODEL_27, x_27, Y_27, w_27, AB, 20)
            MEASURES_27[w_27] = guidance_measures(out_27, Y_27)
            grid_27.append(out_27[::PER_CLASS_27])                  # the first image of each class
        show_images(torch.cat(grid_27), ncol=10, size=0.9,
                    titles=[f"{NAMES_F[c]}, w={w}" for w in W_26 for c in range(10)],
                    suptitle="une image par classe (colonnes), pour w = 0, 1, 3 et 7 (lignes), du même bruit")
        table_27 = wbd.precomputed("guidance_26")["by_w"]
        print(f"{'w':>3} {'distance':>9} {'(protocole)':>12} {'reconnue':>9} {'(protocole)':>12} {'diversité':>10} "
              f"{'saturés':>8}")
        for w_27, m in MEASURES_27.items():
            print(f"{w_27:>3} {fr(m['fd'], 1):>9} {fr(np.median(table_27[str(w_27)]['fd']), 1):>12} "
                  f"{fr(100 * m['acc'], 0) + ' %':>9} {fr(100 * np.median(table_27[str(w_27)]['acc']), 0) + ' %':>12} "
                  f"{fr(m['div'], 3):>10} {fr(100 * m['sat'], 1) + ' %':>8}")
        verdict("B5.27", MEASURES_27[3]["acc"] > MEASURES_27[0]["acc"] and MEASURES_27[3]["div"] < MEASURES_27[0]["div"],
                "la guidance fait reconnaître la classe plus souvent, au prix de la diversité (w = 3 contre w = 0).",
                "à w = 3, la classe devrait être reconnue plus souvent et la diversité plus petite qu'à w = 0 : relis "
                "guided_eps_27 (le signe de l'extrapolation, la classe vide).")'''

# --- 🔬 B5.28 ---------------------------------------------------------------------------------------------------------
GIVEN_28 = r'''# The full protocol (tools/chapters/history_chB5.py): the recipe of B5.23 with the SAME number of optimizer steps
# (N_UPDATES_23) on 500, 10 000 and 60 000 images, «N_SEEDS_28» seeds each; 256 DDIM samples (50 steps) per model and,
# for each sample, the ratio r of copy_ratio_28; the share of samples with r < 1/3
FULL_28 = «FULL_28»


class IdealNet(nn.Module):
    """The perfect denoiser of a finite dataset (wbd.ideal_denoiser, B5.22) as a network model(x, t)."""

    def __init__(self, images):
        super().__init__()
        self.eps_fn = wbd.ideal_denoiser(images, np.sqrt(AB), np.sqrt(1 - AB))

    def forward(self, x, t):
        return self.eps_fn(x, int(t[0]))'''

TODO_28 = r'''def nearest_28(samples, train):
    """For each sample (N, 1, 28, 28), its nearest image of train (M, 1, 28, 28) in pixels: (distances, indices), two
    NumPy arrays (N,), the Euclidean distance of the flattened images and the index in train (the first one on ties)."""
    raise NotImplementedError("nearest_28")


def copy_ratio_28(samples, train):
    """For each sample, r = d(sample, a) / d(a, b), where a is its nearest training image and b the nearest image of
    train OTHER than a (the closest neighbour of a in its own dataset); a float64 NumPy array (N,)."""
    raise NotImplementedError("copy_ratio_28")


copies500_28 = ...   # a) full protocol, 500 images: the share of samples with r < 1/3, for every seed:
                     #    "A" less than 10 %, "B" from 10 to 50 %, "C" more than 50 %
closer500_28 = ...   # b) full protocol: is the median r of the models trained on 500 images SMALLER than that of the
                     #    models trained on 60 000 images, seed by seed? True or False'''

SOLUTION_28 = r'''def nearest_28(samples, train):
    """For each sample (N, 1, 28, 28), its nearest image of train (M, 1, 28, 28) in pixels: (distances, indices), two
    NumPy arrays (N,), the Euclidean distance of the flattened images and the index in train (the first one on ties)."""
    a = torch.as_tensor(samples).reshape(len(samples), -1).double()
    b = torch.as_tensor(train).reshape(len(train), -1).double()
    d, i = torch.cdist(a, b).min(dim=1)
    return d.numpy(), i.numpy()


def copy_ratio_28(samples, train):
    """For each sample, r = d(sample, a) / d(a, b), where a is its nearest training image and b the nearest image of
    train OTHER than a (the closest neighbour of a in its own dataset); a float64 NumPy array (N,)."""
    d, idx = nearest_28(samples, train)
    flat = torch.as_tensor(train).reshape(len(train), -1).double()
    to_all = torch.cdist(flat[idx], flat)                          # distances of each a to every training image
    to_all[torch.arange(len(idx)), torch.as_tensor(idx)] = float("inf")   # not a itself
    return d / to_all.min(dim=1).values.numpy()'''

CHECK_28 = r'''ratio_ok_28 = False
with checked("B5.28"):
    train_c = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [10.0, 10.0]]).reshape(4, 1, 1, 2)
    queries_c = torch.tensor([[0.0, 0.5], [3.0, 3.0], [9.0, 9.0]]).reshape(3, 1, 1, 2)
    nd_28 = nearest_28(queries_c, train_c)
    if returned("B5.28", "nearest_28", nd_28):
        d_c, i_c = pair("B5.28", "nearest_28", nd_28, names=("distances", "indices"))
        ok_28 = same("B5.28", "nearest_28 sur 4 points du plan : les distances", d_c, [0.5, 1.0, math.sqrt(2)])
        ok_28 &= same("B5.28", "nearest_28 : les indices", i_c, [0, 2, 3])
        ok_28 &= same("B5.28", "copy_ratio_28 sur les mêmes points (0,5/3 ; 1/4 ; √2/√85)", copy_ratio_28(queries_c, train_c),
                      [0.5 / 3, 1.0 / 4, math.sqrt(2) / math.sqrt(85)])
        imgs_28 = DATA_23.images[:300]
        ok_28 &= same("B5.28", "copy_ratio_28 de 5 images d'entraînement à peine bruitées (σ = 0,01) contre la référence",
                      copy_ratio_28(imgs_28[:5] + 0.01 * torch.randn(5, 1, 28, 28, generator=torch.Generator().manual_seed(8)),
                                    imgs_28),
                      wbd.copy_ratio(imgs_28[:5] + 0.01 * torch.randn(5, 1, 28, 28, generator=torch.Generator().manual_seed(8)),
                                     imgs_28), rtol=1e-6, atol=1e-9)
        ratio_ok_28 = bool(ok_28)
with wb.attempt("B5.28"):
    wb.check("B5.28a", copies500_28)
with wb.attempt("B5.28"):
    wb.check("B5.28b", closer500_28)'''

EXPERIMENT_28 = r'''# Your model (B5.23) and the perfect denoiser of 500 images (B5.22): how close are their samples to their training images?
if not ratio_ok_28:
    print("⏳ Ex B5.28 : l'expérience attend nearest_28 et copy_ratio_28 (cellule précédente sans ❌).")
else:
    with checked("B5.28"):
        need_model("B5.28")
        x_28 = torch.randn((128, 1, 28, 28), generator=torch.Generator().manual_seed(280))
        small_set_28 = DATA_23.images[:500]                       # the first 500 images of the frozen subset
        runs_28 = {f"ton U-Net ({len(DATA_23)} images)": (MODEL, DATA_23.images),
                   "débruiteur parfait (500 images)": (IdealNet(small_set_28), small_set_28)}
        if not FAST_MODE:                     # 🚀 full mode: a U-Net trained with the same number of steps on 500 images
            torch.manual_seed(SETTINGS_23["seed"])
            net500_28, ema500_28 = UNET_CLASS(**CONFIG_23), None
            ema500_28 = copy.deepcopy(net500_28)
            epochs_28 = math.ceil(N_UPDATES_23 / math.ceil(500 / SETTINGS_23["batch_size"]))
            train_23(net500_28, ema500_28, small_set_28, SETTINGS_23 | {"ema": float(RHO_23), "epochs": epochs_28},
                     torch.Generator().manual_seed(SETTINGS_23["seed"] + 1), AB)
            runs_28["ton U-Net (500 images)"] = (ema500_28, small_set_28)
        RATIOS_28 = {}
        for name_28, (net, train) in runs_28.items():
            samples_28 = wbd.sample_ddim(net, x_28, AB, 50)
            r_28 = copy_ratio_28(samples_28, train)
            RATIOS_28[name_28] = r_28
            d_28, i_28 = nearest_28(samples_28, train)
            order_28 = np.argsort(r_28)[:8]
            show_images(torch.cat([samples_28[order_28], train[i_28[order_28]]]), ncol=8, size=0.9,
                        titles=[f"r={fr(r_28[k], 2)}" for k in order_28] + ["voisin"] * 8,
                        suptitle=f"{name_28} : les 8 échantillons de plus petit r, et leur plus proche voisin")
            print(f"{name_28} : r médian {fr(np.median(r_28), 2)} ; {fr(100 * np.mean(r_28 < 1 / 3), 1)} % des 128 "
                  "échantillons ont r < 1/3")
        fig, ax = plt.subplots(figsize=(7, 3))
        for name_28, r_28 in RATIOS_28.items():
            ax.hist(np.clip(r_28, 0, 2), bins=40, range=(0, 2), alpha=0.5, label=name_28)
        ax.axvline(1 / 3, color="k", ls=":", lw=1)
        ax.set_xlabel("r = d(échantillon, voisin) / d(voisin, son propre voisin)")
        ax.legend(fontsize=8)
        plt.show()
        print("protocole complet, part des échantillons de r < 1/3 (une valeur par graine) :")
        for size_28, shares in FULL_28["copies"].items():
            print(f"  {int(size_28):>6} images : " + " ; ".join(f"{fr(100 * s, 1)} %" for s in shares)
                  + f" (r médian {fr(float(np.median(FULL_28['median_r'][size_28])), 2)})")'''

# --- 📦 B5.29 ---------------------------------------------------------------------------------------------------------
GIVEN_29 = r'''# The code of the mini-project MP6 (projets/partie_6_generatif/solution/genbench: its VAE of ch. 25, its DCGAN of
# ch. 27, their trainings and its evaluation), on MNIST; the features are those of CLF (64 values), as in B5.23
import sys
from pathlib import Path

MP6_29 = wb.repo_root() / "projets" / "partie_6_generatif" / "solution"
if str(MP6_29) not in sys.path:
    sys.path.insert(0, str(MP6_29))
from genbench import evaluate as gb_eval, models as gb_models, train as gb_train  # noqa: E402

TRAIN01_29 = (DATA_23.images + 1) / 2                    # genbench works in [0, 1]
GENBENCH_29 = {"vae": {"latent_dim": 16, "width": 32, "epochs": 3, "beta": 0.5, "seed": 0},
               "dcgan": {"latent_dim": 64, "width": 32, "epochs": 3, "seed": 0}}


class Clf01(nn.Module):
    """CLF seen by genbench: images in [0, 1] (CLF works in [-1, 1]), features() and the 10 logits."""

    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def features(self, x):
        return self.clf.features(2 * x - 1)

    def forward(self, x):
        return self.clf(2 * x - 1)


CLF01 = Clf01(CLF).eval()


def train_vae_29():
    s = GENBENCH_29["vae"]
    gb_train.set_seed(s["seed"])
    vae = gb_models.ConvVAE(latent_dim=s["latent_dim"], width=s["width"])
    gb_train.train_vae(vae, TRAIN01_29, epochs=s["epochs"], beta=s["beta"])
    return vae


def train_dcgan_29():
    import tempfile
    s = GENBENCH_29["dcgan"]
    gb_train.set_seed(s["seed"])
    G, D = gb_models.Generator(latent_dim=s["latent_dim"], width=s["width"]), gb_models.Discriminator(width=s["width"])
    G.apply(gb_models.init_dcgan_weights)
    D.apply(gb_models.init_dcgan_weights)
    gb_train.train_gan(G, D, TRAIN01_29, epochs=s["epochs"], folder=Path(tempfile.mkdtemp(prefix="wb_b5_29_")))
    return G


# trained once (≈ 35 s on CPU), then read from wbd.checkpoints_dir() (a key with the settings and the data)
VAE_29 = wbd.cached_module({"model": "genbench-vae", "settings": GENBENCH_29["vae"], "data": DATA_23.describe()},
                           lambda: gb_models.ConvVAE(latent_dim=16, width=32), train_vae_29)
GAN_29 = wbd.cached_module({"model": "genbench-dcgan", "settings": GENBENCH_29["dcgan"], "data": DATA_23.describe()},
                           lambda: gb_models.Generator(latent_dim=64, width=32), train_dcgan_29)'''

TODO_29 = r'''def evaluate_29(images01, train01):
    """The evaluation of the MP6 for generated images in [0, 1] (N, 1, 28, 28), with the functions of
    genbench.evaluate and the classifier CLF01: {"fd": frechet_distance between the features of the 2 000 test digits
    (REAL_MNIST, brought to [0, 1]) and those of images01, "entropy" and "confidence": those of class_report,
    "nn": the median of the distances of nearest_neighbors(images01, train01)}."""
    raise NotImplementedError("evaluate_29")'''

SOLUTION_29 = r'''def evaluate_29(images01, train01):
    """The evaluation of the MP6 for generated images in [0, 1] (N, 1, 28, 28), with the functions of
    genbench.evaluate and the classifier CLF01: {"fd": frechet_distance between the features of the 2 000 test digits
    (REAL_MNIST, brought to [0, 1]) and those of images01, "entropy" and "confidence": those of class_report,
    "nn": the median of the distances of nearest_neighbors(images01, train01)}."""
    real = gb_eval.classifier_features(CLF01, (REAL_MNIST.images + 1) / 2)
    fake = gb_eval.classifier_features(CLF01, images01)
    report = gb_eval.class_report(CLF01, images01)
    distances, _ = gb_eval.nearest_neighbors(images01, train01)
    return {"fd": gb_eval.frechet_distance(real, fake), "entropy": report["entropy"],
            "confidence": report["confidence"], "nn": float(np.median(distances))}'''

CHECK_29 = r'''eval_ok_29 = False
with checked("B5.29"):
    probe_29 = (DATA_23.images[-100:] + 1) / 2
    got_29 = evaluate_29(probe_29, TRAIN01_29[:2000])
    if returned("B5.29", "evaluate_29", got_29):
        keys_29 = set(got_29) if isinstance(got_29, dict) else set()
        if keys_29 != {"fd", "entropy", "confidence", "nn"}:
            print(f"❌ Ex B5.29 : evaluate_29 rend un dictionnaire aux clés fd, entropy, confidence et nn ; reçu {got_29!r:.120}.")
        else:
            real_29 = wbd.features(CLF, REAL_MNIST.images)
            want_29 = {"fd": mylearn.gen_metrics.frechet_distance_from_features(real_29, wbd.features(CLF, 2 * probe_29 - 1)),
                       "nn": float(np.median(torch.cdist(probe_29.reshape(100, -1).double(),
                                                         TRAIN01_29[:2000].reshape(len(TRAIN01_29[:2000]), -1).double()).min(dim=1).values.numpy()))}
            proba_29 = torch.softmax(CLF(2 * probe_29 - 1), dim=1).detach()
            shares_29 = np.bincount(proba_29.argmax(dim=1).numpy(), minlength=10) / 100
            nz_29 = shares_29[shares_29 > 0]
            want_29["entropy"] = float(-(nz_29 * np.log(nz_29)).sum() / np.log(10))
            want_29["confidence"] = float(proba_29.max(dim=1).values.mean())
            eval_ok_29 = same("B5.29", "evaluate_29 sur 100 vrais chiffres (fd, entropy, confidence, nn) contre le calcul "
                              "direct", [float(got_29[k]) for k in ("fd", "entropy", "confidence", "nn")],
                              [want_29[k] for k in ("fd", "entropy", "confidence", "nn")], rtol=1e-4, atol=1e-6)'''

EXPERIMENT_29 = r'''# 256 images of each generator, measured by YOUR evaluate_29 (the same features, the same number of images)
if not eval_ok_29:
    print("⏳ Ex B5.29 : la comparaison attend evaluate_29 (cellule précédente sans ❌).")
else:
    with checked("B5.29"):
        need_model("B5.29")
        x_29 = torch.randn((256, 1, 28, 28), generator=torch.Generator().manual_seed(290))
        GENERATED_29 = {"DDPM (ton U-Net, DDIM 50 pas)": (wbd.sample_ddim(MODEL, x_29, AB, 50).clamp(-1, 1) + 1) / 2,
                        "VAE (ch. 25, genbench)": gb_eval.sample_images(VAE_29, 256, seed=291),
                        "DCGAN (ch. 27, genbench)": gb_eval.sample_images(GAN_29, 256, seed=291),
                        "vraies images (d'autres chiffres d'entraînement)": TRAIN01_29[-256:]}
        RESULTS_29 = {name: evaluate_29(images, TRAIN01_29[:-256]) for name, images in GENERATED_29.items()}
        show_images(torch.cat([2 * images[:8] - 1 for images in GENERATED_29.values()]), ncol=8, size=0.9,
                    suptitle="8 images de chaque générateur (lignes : DDPM, VAE, DCGAN, vraies images)")
        print(f"{'générateur':<50} {'distance':>9} {'entropie':>9} {'confiance':>10} {'voisin médian':>14}")
        for name, r in RESULTS_29.items():
            print(f"{name:<50} {fr(r['fd'], 1):>9} {fr(r['entropy'], 3):>9} {fr(r['confidence'], 3):>10} {fr(r['nn'], 2):>14}")
        print("(256 images par générateur, features de CLF, distance aux 2 000 chiffres de test ; « voisin médian » : la "
              "distance en pixels médiane au plus proche chiffre d'entraînement)")'''

# --- 🛠️ B5.30 --------------------------------------------------------------------------------------------------------
TODO_30 = r'''OUT_30 = Path("b5_echantillonneur")      # the folder of your script and of its images (created if needed)

SAMPLE_PY_30 = r"""# sample.py: images of a diffusion checkpoint of chapter B5 (DDIM, x̂0 clipped), with their parameters.
#
#     python sample.py CHECKPOINT [--steps 50] [--seed 0] [--guidance 0] [--label K] [--n 16] [--out samples]
import argparse
import sys

from wb import diffusion as wbd


def main(argv=None):
    raise NotImplementedError("sample.py")


if __name__ == "__main__":
    sys.exit(main())
"""'''

SOLUTION_30 = r'''import tempfile
from pathlib import Path

OUT_30 = Path(tempfile.mkdtemp(prefix="wb_b5_30_"))   # the solutions notebook writes nothing in the repository

SAMPLE_PY_30 = r"""# sample.py: images of a diffusion checkpoint of chapter B5 (DDIM, x̂0 clipped), with their parameters.
#
#     python sample.py CHECKPOINT [--steps 50] [--seed 0] [--guidance 0] [--label K] [--n 16] [--out samples]
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wb import diffusion as wbd


def parse_args(argv):
    p = argparse.ArgumentParser(description="DDIM samples of a checkpoint of chapter B5")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("samples"))
    return p.parse_args(argv)


def fail(message):
    print(f"sample.py: {message}", file=sys.stderr)
    return 2


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        return fail(f"no checkpoint {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings, config = json.loads(payload["key"])["settings"], payload["config"]
    T, n_classes = config["n_steps"], config["n_classes"]
    if not 1 <= args.steps <= T:
        return fail(f"--steps must be in [1, {T}], got {args.steps}")
    if args.n < 1 or args.guidance < 0:
        return fail("--n must be >= 1 and --guidance >= 0")
    if n_classes is None and (args.guidance > 0 or args.label is not None):
        return fail("this checkpoint has no classes: no --guidance, no --label")
    if args.label is not None and not 0 <= args.label < n_classes:
        return fail(f"--label must be in [0, {n_classes})")
    model = wbd.TinyUNet(**config)
    model.load_state_dict(payload["ema_state_dict"] or payload["state_dict"])
    model.eval()
    alpha_bars = wbd.alpha_bars(wbd.schedule_betas(settings["schedule"], T))
    x_T = torch.randn((args.n, 1, 28, 28), generator=torch.Generator().manual_seed(args.seed))
    y = None
    if n_classes is not None:
        y = torch.full((args.n,), args.label) if args.label is not None else torch.arange(args.n) % n_classes
    start = time.perf_counter()
    images = wbd.sample_ddim(model, x_T, alpha_bars, args.steps, y=y, w=args.guidance).clamp(-1, 1)
    seconds = time.perf_counter() - start
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "samples.npy", images.numpy())
    ncol = min(8, args.n)
    nrow = -(-args.n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img in zip(axes.ravel(), wbd.to_unit(images)):
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    fig.savefig(args.out / "samples.png", dpi=80)
    params = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "steps": args.steps, "subsequence": [int(t) for t in wbd.ddim_timesteps(T, args.steps)],
              "seed": args.seed, "guidance": args.guidance, "label": args.label, "n": args.n,
              "seconds": round(seconds, 3), "torch": torch.__version__, "date": datetime.date.today().isoformat()}
    (args.out / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    print(f"{args.n} images -> {args.out / 'samples.png'} ({seconds:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""'''

CHECK_30 = r'''import json
import subprocess
import sys


def run_30(*args, out="run"):
    """Run YOUR sample.py in OUT_30 with the arguments; returns (exit code, stderr, folder of the images)."""
    folder = OUT_30 / out
    done = subprocess.run([sys.executable, "sample.py", *map(str, args), "--out", str(folder)], cwd=OUT_30,
                          capture_output=True, text=True, timeout=300)
    return done.returncode, done.stderr.strip().splitlines()[-1:] if done.stderr.strip() else [], folder


script_ok_30 = False
with checked("B5.30"):
    need_model("B5.30")
    OUT_30.mkdir(parents=True, exist_ok=True)
    (OUT_30 / "sample.py").write_text(SAMPLE_PY_30, encoding="utf-8")
    print(f"ℹ️ ton script : {(OUT_30 / 'sample.py').resolve()}")
    ckpt_30 = CKPT["path"]
    code_30, err_30, dir_a = run_30(ckpt_30, "--steps", 10, "--seed", 1, "--n", 4, out="a")
    ok_30 = code_30 == 0
    verdict("B5.30", ok_30, "python sample.py <checkpoint> --steps 10 --seed 1 --n 4 : code de sortie 0.",
            f"python sample.py <checkpoint> --steps 10 --seed 1 --n 4 : code de sortie {code_30} ; {err_30}")
    if not ok_30:
        raise StopCheck
    files_30 = {p.name for p in dir_a.iterdir()}
    verdict("B5.30", {"samples.npy", "samples.png", "params.json"} <= files_30,
            "le dossier --out contient samples.npy, samples.png et params.json.",
            f"le dossier --out doit contenir samples.npy, samples.png et params.json ; il contient {sorted(files_30)}.")
    images_a = np.load(dir_a / "samples.npy")
    want_30 = wbd.sample_ddim(MODEL, torch.randn((4, 1, 28, 28), generator=torch.Generator().manual_seed(1)), AB, 10)
    ok_30 &= same("B5.30", "samples.npy : les 4 images de DDIM à 10 pas du bruit de graine 1 (copie EMA)", images_a,
                  want_30.clamp(-1, 1).numpy(), rtol=1e-4, atol=1e-4)
    params_30 = json.loads((dir_a / "params.json").read_text(encoding="utf-8"))
    need_30 = {"checkpoint_sha256", "steps", "seed", "guidance", "n", "subsequence"}
    ok_params_30 = need_30 <= set(params_30) and params_30.get("steps") == 10 and params_30.get("seed") == 1 and \
        params_30.get("subsequence") == wbd.ddim_timesteps(T_STEPS, 10).tolist()
    verdict("B5.30", ok_params_30, "params.json : l'empreinte du checkpoint, les pas, leur sous-séquence, la graine, la guidance.",
            f"params.json doit contenir au moins {sorted(need_30)}, avec les valeurs de l'appel ; reçu {params_30!r:.200}.")
    code_b, _, dir_b = run_30(ckpt_30, "--steps", 10, "--seed", 1, "--n", 4, out="b")
    same_seed_30 = code_b == 0 and np.array_equal(np.load(dir_b / "samples.npy"), images_a)
    verdict("B5.30", same_seed_30, "même graine, mêmes images, au bit près.", "deux appels de même graine doivent rendre les "
            "mêmes images, au bit près.")
    code_c, _, dir_c = run_30(ckpt_30, "--steps", 10, "--seed", 2, "--n", 4, out="c")
    other_seed_30 = code_c == 0 and not np.allclose(np.load(dir_c / "samples.npy"), images_a)
    verdict("B5.30", other_seed_30, "une autre graine donne d'autres images.", "--seed 2 doit donner d'autres images que --seed 1.")
    code_d, err_d, _ = run_30(ckpt_30, "--guidance", 2, "--n", 4, out="d")
    verdict("B5.30", code_d != 0, f"--guidance sur un modèle sans classes : refusé (code {code_d}, {err_d}).",
            "--guidance 2 sur un checkpoint sans classes doit échouer avec un message, pas générer.")
    code_e, err_e, _ = run_30(ckpt_30, "--steps", 0, out="e")
    verdict("B5.30", code_e != 0, f"--steps 0 : refusé (code {code_e}, {err_e}).", "--steps 0 doit échouer avec un message.")
    ok_30 &= ok_params_30 and same_seed_30 and other_seed_30 and code_d != 0 and code_e != 0
    ckpt27_30 = wbd.load_checkpoint(key_27(globals().get("CKPT_SOURCE_23", "learner")))
    if ckpt27_30 is None:
        print("⏳ Ex B5.30 : --guidance et --label se vérifient avec le modèle conditionnel de B5.27 (pas encore sauvegardé).")
    else:
        code_f, err_f, dir_f = run_30(ckpt27_30["path"], "--steps", 10, "--seed", 3, "--n", 4, "--guidance", 3, "--label", 7,
                                      out="f")
        model27_30 = wbd.build_from_checkpoint(wbd.TinyUNet, ckpt27_30)
        want_f = wbd.sample_ddim(model27_30, torch.randn((4, 1, 28, 28), generator=torch.Generator().manual_seed(3)), AB,
                                 10, y=torch.full((4,), 7), w=3.0).clamp(-1, 1).numpy()
        ok_30 &= code_f == 0 and same("B5.30", "--guidance 3 --label 7 sur le modèle de B5.27 : la guidance de la fiche "
                                      "(w = 3)", np.load(dir_f / "samples.npy") if code_f == 0 else None, want_f,
                                      rtol=1e-4, atol=1e-4)
    script_ok_30 = bool(ok_30)'''

# --- 🏆 B5.31 ---------------------------------------------------------------------------------------------------------
GIVEN_31 = r'''# The challenge: the smallest Fréchet distance on Fashion-MNIST with at most 50 network passes per image
MAX_PASSES_31 = 50
LIMIT_31 = «LIMIT_FAST_31» if FAST_MODE else «LIMIT_FULL_31»   # seconds of training: a reduced run in fast mode, THE challenge in full mode
N_31 = 256 if FAST_MODE else 1000                     # generated images, compared to 2 000 test images
GOAL_31 = «GOAL_FAST_31» if FAST_MODE else «GOAL_FULL_31»    # the distance to reach (worst seed of the reference + 15 %)
REFERENCE_31 = «REF_FAST_31» if FAST_MODE else «REF_FULL_31»   # the reference solution: (best, worst) of its seeds
COPIES_31 = «COPIES_FAST_31» if FAST_MODE else «COPIES_FULL_31»   # largest share of copies (r < 1/3) allowed: reference + margin


def _make_evaluate_31():
    """evaluate_31, with the test images and their features kept here only (the global variables that hold images,
    models or checkpoints are replaced by a decoy while your method runs)."""
    import re
    import tempfile
    from pathlib import Path
    from wb import datasets as wbdata

    train = wbd.image_subset("fashion", wbd.n_train(FAST_MODE))
    test = wbd.image_subset("fashion", 2000, split="test")
    clf = wbd.classifier("fashion")
    test_feats = wbd.features(clf, test.images)

    class Hidden:
        """What a hidden global variable holds during the run: any use of it says why."""

        def __init__(self, name):
            self._name = name

        def _refuse(self, *args, **kwargs):
            raise RuntimeError(f"pendant le défi, {self._name} est masqué : ta méthode ne reçoit que images, labels et "
                               "seconds")

        __getattr__ = __getitem__ = __call__ = __iter__ = __len__ = _refuse

    ipython_names = re.compile(r"^(_+|_i+|_i?\d+|_[ido]h|In|Out|exit|quit|get_ipython)$")

    def hidden_globals():
        """The global variables that could hold images, features, a trained model or a checkpoint."""
        g, found = globals(), {}
        for n, v in list(g.items()):
            if ipython_names.match(n) or n.startswith("_"):
                continue
            if isinstance(v, wbd.ImageSet):
                found[n] = Hidden(n)
            elif isinstance(v, dict) and ("state_dict" in v or "ema_state_dict" in v):
                found[n] = Hidden(n)
            elif isinstance(v, nn.Module) and any(p.numel() >= 1000 for p in v.parameters()):
                found[n] = Hidden(n)
            elif isinstance(v, torch.Tensor) and v.numel() >= 784 * 10:
                found[n] = Hidden(n)
            elif isinstance(v, np.ndarray) and v.size >= 784 * 10:
                found[n] = Hidden(n)
        return found

    def refused(*args, **kwargs):
        raise RuntimeError("pendant le défi, ta méthode ne reçoit que images, labels et seconds (ni données du workbook, "
                           "ni sauvegarde, ni modèle déjà entraîné, ni classifieur des mesures)")

    def run(method, seconds):
        """Train the method cold, then generate N_31 images, counting the passes of every network."""
        g = globals()
        hidden = hidden_globals()
        before = [m for m in list(g.values()) if isinstance(m, nn.Module)]
        old_params = [p.detach() for m in before for p in m.parameters() if p.numel() > 1000]
        saved = {name: getattr(wbd, name) for name in ("image_subset", "raw_subset", "_raw", "load_checkpoint",
                                                        "reference_model", "classifier", "features", "predict",
                                                        "checkpoints_dir")}
        saved_data = {name: getattr(wbdata, name) for name in ("load_mnist", "load_fashion_mnist")}
        saved_torch = (torch.load, torch.serialization.load)
        empty = Path(tempfile.mkdtemp(prefix="wb_b5_31_"))
        originals = {n: g[n] for n in hidden}
        caches = [(cache, dict(cache)) for cache in (wbd._RAW, wbd._CLASSIFIERS, wbdata._ARRAYS)]
        counts, depth, used = {"passes": 0}, [0], []

        def pre(module, args):
            if depth[0] == 0 and args and torch.is_tensor(args[0]):
                counts["passes"] += int(args[0].shape[0])
                used.append(module)
            depth[0] += 1

        def post(module, args, output):
            depth[0] -= 1

        out = elapsed = None
        try:
            g.update(hidden)
            for cache, _ in caches:                           # the images and models kept in memory by wb
                cache.clear()
            for name in saved:
                setattr(wbd, name, refused)
            wbd.checkpoints_dir = lambda: empty
            for name in saved_data:
                setattr(wbdata, name, refused)
            torch.load = torch.serialization.load = refused
            torch.manual_seed(3131)
            start = time.perf_counter()
            generate = method(train.images.clone(), train.labels.clone(), seconds)
            elapsed = time.perf_counter() - start
            if not callable(generate):
                print(f"❌ Ex B5.31 : method_31 doit rendre une fonction generate(n, generator) ; reçu {type(generate).__name__}.")
                return None
            handles = (nn.modules.module.register_module_forward_pre_hook(pre),
                       nn.modules.module.register_module_forward_hook(post))
            try:
                with torch.no_grad():
                    out = generate(N_31, torch.Generator().manual_seed(3132))
            finally:
                for h in handles:
                    h.remove()
        finally:
            g.update(originals)
            for cache, content in caches:
                cache.clear()
                cache.update(content)
            for name, f in saved.items():
                setattr(wbd, name, f)
            for name, f in saved_data.items():
                setattr(wbdata, name, f)
            torch.load, torch.serialization.load = saved_torch
        if not (torch.is_tensor(out) and tuple(out.shape) == (N_31, 1, 28, 28) and torch.isfinite(out).all()):
            print(f"❌ Ex B5.31 : generate(n, generator) doit rendre un tenseur fini de forme ({N_31}, 1, 28, 28) ; reçu "
                  f"{type(out).__name__} {tuple(getattr(out, 'shape', ()))}.")
            return None
        reused = any(any(m is b for b in before) for m in used) or any(
            p.numel() > 1000 and any(q.shape == p.shape and torch.equal(q, p.detach()) for q in old_params)
            for m in used for p in m.parameters())
        if reused:
            print("❌ Ex B5.31 : generate doit utiliser un modèle neuf, entraîné pendant la course : un réseau appelé existait "
                  "déjà, ou il a les poids d'un modèle du notebook.")
            return None
        return out.float().clamp(-1, 1), elapsed, counts["passes"] / N_31

    def evaluate_31(method):
        """method(images, labels, seconds) -> generate(n, generator): the rules of the statement, then the distance."""
        result = run(method, LIMIT_31)
        if result is None:
            return None
        images, seconds, passes = result
        feats = wbd.features(clf, images)
        fd = float(mylearn.gen_metrics.frechet_distance_from_features(test_feats, feats))
        r = wbd.copy_ratio(images, train.images)
        copies = float(np.mean(r < 1 / 3))
        to_test = float(torch.cdist(images.reshape(N_31, -1).double(), test.images.reshape(len(test), -1).double()).min())
        show_images(images[:32], ncol=16, size=0.75, suptitle="32 des images générées par ta méthode")
        print(f"entraînement en {fr(seconds, 1)} s ; {fr(passes, 1)} passes de réseau par image ; distance de Fréchet "
              f"{fr(fd, 2)} ({N_31} images contre 2 000 images de test) ; copies (r < 1/3) : {fr(100 * copies, 1)} %.")
        rules = [(seconds <= LIMIT_31 * 1.05 + 2, f"entraînement en {fr(seconds, 1)} s, dans les {LIMIT_31:.0f} s.",
                  f"entraînement en {fr(seconds, 1)} s : le défi en permet {LIMIT_31:.0f} (ta méthode doit respecter seconds)."),
                 (passes <= MAX_PASSES_31, f"{fr(passes, 1)} passes de réseau par image, au plus {MAX_PASSES_31}.",
                  f"{fr(passes, 1)} passes de réseau par image (la guidance compte double) : le défi en permet "
                  f"{MAX_PASSES_31}."),
                 (copies <= COPIES_31, f"contrôle des copies : {fr(100 * copies, 1)} % d'images de r < 1/3 (au plus "
                                       f"{fr(100 * COPIES_31, 0)} %, la référence plus une marge).",
                  f"contrôle des copies : {fr(100 * copies, 1)} % des images sont anormalement proches d'une image "
                  f"d'entraînement (r < 1/3 ; au plus {fr(100 * COPIES_31, 0)} %) : ta méthode recopie-t-elle ?"),
                 (to_test > 1.0, "aucune image n'est une image de test.",
                  "une image générée est une image de test, à moins de 1 en distance de pixels.")]
        for ok, success, failure in rules:
            verdict("B5.31", ok, success, failure)
        ok = all(rule[0] for rule in rules)
        verdict("B5.31", ok and fd <= GOAL_31,
                f"défi réussi : distance {fr(fd, 2)} (objectif {fr(GOAL_31, 1)} ; référence de {fr(REFERENCE_31[0], 1)} à "
                f"{fr(REFERENCE_31[1], 1)}).",
                f"pas encore : distance {fr(fd, 2)}, il en faut au plus {fr(GOAL_31, 1)} en respectant les règles.")
        return {"fd": fd, "seconds": seconds, "passes": passes, "copies": copies, "ok": ok and fd <= GOAL_31}

    return evaluate_31


evaluate_31 = _make_evaluate_31()
print(f"Course de {LIMIT_31:.0f} s d'entraînement, {MAX_PASSES_31} passes par image ; objectif : une distance de Fréchet "
      f"d'au plus {fr(GOAL_31, 1)} sur {N_31} images.")'''

TODO_31 = r'''def method_31(images, labels, seconds):
    """Your method: build and train, in at most `seconds` seconds, a NEW model on images (float32 (N, 1, 28, 28) in
    [-1, 1], Fashion-MNIST) and labels (int64 (N,)) ONLY; return a function generate(n, generator) that returns n
    images, a float32 tensor (n, 1, 28, 28) in [-1, 1], with at most 50 network passes per image."""
    raise NotImplementedError("method_31")'''

SOLUTION_31 = r'''SEED_31 = 3100          # the seed of the reference (history_chB5.py calibrates the goal on several seeds)
W_31, STEPS_31 = «W_31», «STEPS_31»     # guidance and DDIM steps: 2 passes per step with guidance, so 25 steps = 50 passes


def method_31(images, labels, seconds):
    """Your method: build and train, in at most `seconds` seconds, a NEW model on images (float32 (N, 1, 28, 28) in
    [-1, 1], Fashion-MNIST) and labels (int64 (N,)) ONLY; return a function generate(n, generator) that returns n
    images, a float32 tensor (n, 1, 28, 28) in [-1, 1], with at most 50 network passes per image."""
    start = time.perf_counter()
    torch.manual_seed(SEED_31)
    generator = torch.Generator().manual_seed(SEED_31 + 1)
    T = T_STEPS
    ab_np = AB                                                         # the schedule of the notebook
    ab = torch.tensor(ab_np, dtype=torch.float32)
    model = UNET_CLASS(T, channels=(16, 32, 64), n_classes=10)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    budget, step = 0.95 * seconds, 0
    model.train()
    while time.perf_counter() - start < budget:
        idx = torch.randint(len(images), (32,), generator=generator)
        x0, y = images[idx], labels[idx]
        t = torch.randint(0, T, (32,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        y = torch.where(torch.rand(32, generator=generator) < 0.1, torch.full_like(y, 10), y)
        a = ab[t].view(-1, 1, 1, 1)
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, y), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update_23(ema, model, min(0.99, (1 + step) / (10 + step)))    # an EMA that warms up
        step += 1
    ema.eval()

    def generate(n, generator):
        x_T = torch.randn((n, 1, 28, 28), generator=generator)
        y = torch.arange(n) % 10                                        # as many images of each class
        return sample_cond_27(ema, x_T, y, W_31, ab_np, STEPS_31).clamp(-1, 1)

    return generate'''

CHECK_31 = r'''result_31 = None
with checked("B5.31"):
    if globals().get("method_31") is None:
        raise StopCheck
    result_31 = evaluate_31(method_31)'''
