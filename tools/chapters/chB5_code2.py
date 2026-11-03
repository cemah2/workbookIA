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
