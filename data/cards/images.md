# Images du ch. 28 : photos libres et tableaux du domaine public

| | |
|---|---|
| **Chargement** | `wb.datasets.load_sample_image(name, size)` (photos) et `wb.datasets.load_style_image(name, size)` (tableaux) → `float32` `(3, H, W)` dans $[0, 1]$, grand côté `size`, proportions gardées |
| **Crédits** | `wb.datasets.image_credit(name)` ; dictionnaires `CONTENT_IMAGES` et `STYLE_IMAGES` |
| **Photos** | `china` et `flower`, livrées avec scikit-learn (`sklearn.datasets.load_sample_image`), 640 × 427 ; aucun téléchargement |
| **Tableaux** | `data/images/starry_night.jpg` (98 Ko), `great_wave.jpg` (76 Ko), `composition_vii.jpg` (92 Ko) : reproductions de 500 pixels de large tirées de Wikimedia Commons le 2026-10-30, empreintes SHA-256 dans `STYLE_IMAGES` |
| **Licences** | photos : CC BY 2.0 (attribution obligatoire) ; tableaux : domaine public (Van Gogh, mort en 1890 ; Hokusai, mort en 1849 ; Kandinsky, mort en 1944, libre dans l'Union européenne depuis le 1ᵉʳ janvier 2015), et reproductions fidèles non protégées (directive (UE) 2019/790, art. 14) |

## Provenance
- Photos : `sklearn/datasets/images/README.txt` les attribue toutes deux à danielbuechele (Flickr), sous CC BY 2.0, mais le lien de `flower.jpg` pointe vers le compte vultilion : `image_credit` donne les deux noms et les liens tels quels.
- Tableaux : miniatures de 500 pixels de Wikimedia Commons (*La Nuit étoilée*, version « Google Art Project » ; *La Grande Vague de Kanagawa* ; *Composition VII*). Elles sont versionnées (moins de 300 Ko au total) parce que Wikimedia limite les téléchargements automatiques (erreur 429 à la troisième requête rapprochée) ; sans la copie versionnée, `load_style_image` télécharge avec un `User-Agent` explicite et vérifie l'empreinte, puis, en dernier recours, rend une texture procédurale (`procedural_style`) avec un avertissement.

## Utilisation
Ch. 28 : le deep dream sur une photo, le transfert de style (contenu : une photo ; style : un tableau), la galerie documentée de 🛠️ 28.26.

## Biais et limites
- Trois tableaux seulement, tous très texturés ; deux photos seulement, de 640 × 427 pixels.
- **Jamais** d'œuvre encore protégée (Picasso, mort en 1973 : protégé dans l'UE jusqu'à la fin de 2043) : le livre en utilise, le workbook non.
