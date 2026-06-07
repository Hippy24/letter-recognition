# Guide de mise en ligne — Letter Recognition Demo

## Structure finale du projet

```
letter-recognition/
├── demo/
│   └── index.html          ← dashboard autonome (aucun serveur requis)
├── src/
│   ├── train.py
│   ├── clustering.py
│   └── predict.py
├── notebooks/
│   └── letter_recognition.ipynb
├── data/                   ← gitignored
├── outputs/                ← gitignored (modèles .pkl, graphiques)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Option 1 — GitHub Pages (recommandé, gratuit, lié au repo)

C'est l'option idéale : l'URL sera `https://TON_USERNAME.github.io/letter-recognition`

### Étapes

```bash
# 1. Initialiser le repo et pousser le code
git init
git add .
git commit -m "feat: initial project — letter recognition ML + live demo"
git branch -M main
git remote add origin https://github.com/TON_USERNAME/letter-recognition.git
git push -u origin main
```

Ensuite sur GitHub :
1. Ouvrir le repo → **Settings** → **Pages** (colonne gauche)
2. Source : **Deploy from a branch**
3. Branch : `main` / Folder : `/demo`
4. Cliquer **Save**

GitHub Pages sera actif en ~2 minutes.  
URL : `https://TON_USERNAME.github.io/letter-recognition`

### Mettre à jour la démo

```bash
# Modifier demo/index.html, puis :
git add demo/index.html
git commit -m "fix: update demo predictions"
git push
# → GitHub Pages se redéploie automatiquement en ~1 min
```

---

## Option 2 — Netlify (drag & drop, 30 secondes)

Idéal si tu veux une URL custom comme `letter-recognition.netlify.app`

1. Aller sur [netlify.com](https://netlify.com) → créer un compte gratuit
2. Aller sur [app.netlify.com/drop](https://app.netlify.com/drop)
3. **Glisser-déposer le dossier `demo/`** dans la zone
4. Netlify génère une URL immédiatement

Pour une URL personnalisée :
- Dans Netlify → Site settings → **Change site name**
- Choisir : `letter-recognition-demo` → URL : `letter-recognition-demo.netlify.app`

### Déploiement continu depuis GitHub (optionnel)

```bash
# Chez Netlify : New site → Import from Git → GitHub → letter-recognition
# Base directory : demo
# Publish directory : demo
# Build command : (laisser vide — c'est du HTML statique)
```

---

## Option 3 — Vercel (le plus rapide)

```bash
# Installer Vercel CLI
npm install -g vercel

# Depuis le dossier demo/
cd demo/
vercel

# Suivre les instructions :
# ? Set up and deploy "demo"? → Y
# ? Which scope? → ton compte
# ? Link to existing project? → N
# ? What's your project's name? → letter-recognition
# ? In which directory is your code located? → ./
# → URL générée immédiatement
```

---

## Après la mise en ligne : mettre à jour le README et le dashboard

### 1. Dans `demo/index.html`, remplacer :
```html
<!-- LIGNE 1 : le lien GitHub dans le header -->
href="https://github.com/VOTRE_USERNAME/letter-recognition"

<!-- LIGNE 2 : le lien dans le footer -->
href="https://github.com/VOTRE_USERNAME/letter-recognition"
```
Par ton vrai username.

### 2. Dans `README.md`, ajouter en haut :

```markdown
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://TON_USERNAME.github.io/letter-recognition)
```

Et dans la section "Démarrage rapide" :

```markdown
## Démo live

👉 **[Tester le modèle en ligne](https://TON_USERNAME.github.io/letter-recognition)**
```

### 3. Commit final

```bash
git add README.md demo/index.html
git commit -m "docs: add live demo link"
git push
```

---

## Vérifications avant de partager

- [ ] Ouvrir `demo/index.html` en local (double-clic) → ça marche ?
- [ ] L'URL GitHub Pages s'ouvre correctement ?
- [ ] Les préréglages (A, E, I, M, O…) donnent bien ces lettres ?
- [ ] Les sliders mettent à jour la prédiction en direct ?
- [ ] Le lien GitHub dans le header pointe vers ton repo ?

---

## Ce que tu peux dire en entretien

> "Le dashboard tourne entièrement en JavaScript, sans serveur. J'ai pré-calculé les centroïdes du modèle k-NN entraîné sur les 20 000 exemples UCI, et la prédiction se fait par distance euclidienne normalisée côté client. Ça permet de déployer la démo gratuitement sur GitHub Pages avec zéro infrastructure."

---

## Commandes git utiles pour la suite

```bash
# Voir l'état du repo
git status

# Ajouter et commiter en une ligne
git add -A && git commit -m "feat: ..."

# Pousser
git push

# Créer une branche pour une nouvelle feature
git checkout -b feat/streamlit-app

# Tags (pour marquer une version stable)
git tag -a v1.0.0 -m "Premier release — SVM 97.8%"
git push origin v1.0.0
```
