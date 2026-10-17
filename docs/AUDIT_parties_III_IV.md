# Audit P5 des parties III et IV

*Chapitres 12 à 20, checkpoints III et IV, mini-projets MP3 et MP4. Prompt P5 (`docs/METHODE.md` §3), session 43.*

> **Statut : 🛠️ en cours (2026-10-17).** Onze relecteurs indépendants lancés en lecture seule sur le commit `5a17b91` ; leurs rapports bruts iront dans `docs/audit_P5_parties_III_IV/`, le tri et les propositions dans ce fichier.

## 1. Méthode

Onze relecteurs indépendants (sous-agents), en lecture seule, chacun sur une dimension de l'audit, avec les consignes communes de l'audit des parties 0 à II (`docs/AUDIT_parties_I_II.md`, annexe A), adaptées au périmètre (annexe A ci-dessous). Nouveauté : les relecteurs ont le texte du livre (volumes 1 et 2, usage interne de l'audit, jamais versionné) pour juger les encadrés « Le livre : » et « Le livre, corrigé », les renvois et la règle du §3.

| # | Relecteur | Dimension |
|---|---|---|
| A1 | termes | terminologie (§5, §22), titres, tutoiement, notations (§6), encadrés (§14), en-têtes (§11), typographie, d'un chapitre à l'autre |
| A2 | liens | renvois entre chapitres et leur sens, renvois vers l'avant, rappels 🔁, liens relatifs et externes, notions utilisées avant d'être enseignées (§15.2), remédiations, `requires` |
| A3 | structure | conformité aux §11, §12, §17 et §19 par script ; indices de niveau 3 ; échafaudage décroissant de la partie III |
| A4 | exactitude | ch. 12 et 13 |
| A5 | exactitude | ch. 14 et 15 |
| A6 | exactitude | ch. 16 et 17 |
| A7 | exactitude | ch. 18 et 19 |
| A8 | exactitude | ch. 20 et la cheatsheet PyTorch |
| A9 | checkpoints | CP3, CP4, MP3, MP4 : cohérence entre eux, avec CP1, CP2, MP1, MP2 et avec les chapitres ; corrigés recalculés ; kits essayés |
| A10 | code | API `mylearn` des parties III et IV, stubs ↔ référence ↔ tests ↔ énoncés, mutations, `wb`, cellules de vérification, « Run all » |
| A11 | parcours | expérience de l'apprenant de bout en bout (copie du dépôt), consignes, progression, parcours, suivi, Colab |

**Sévérités** (comme en session 24). MAJEUR : erreur de fond qui trompe ou bloque l'apprenant, ou correction qui changerait le contrat (énoncé ou réponse d'un exercice publié, ID, titre, type, durée, stub, règle de la bible, convention transversale) : **proposée, à valider par toi avant d'être appliquée**. MINEUR : défaut local sans effet sur le contrat : appliqué directement après vérification. SUGGESTION : facultatif.

## Annexe A — consignes des relecteurs

Les consignes communes reprennent celles de l'audit des parties 0 à II (`docs/AUDIT_parties_I_II.md`, annexe A.1), avec trois changements : le périmètre (ch. 12 à 20, CP3, CP4, MP3, MP4, modules `mylearn` et tests de ces chapitres ; les parties 0 à II ne sont relues que pour la cohérence) ; l'accès au texte du livre ; un rapport écrit au fur et à mesure, pour qu'une interruption ne perde rien.
