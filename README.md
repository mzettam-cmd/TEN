# Thermodynamic Equilibrium Network (TEN)

Prototype de recherche pour étudier un réseau à relaxation énergétique avec apprentissage contrastif local, sans rétropropagation globale.

## Objectifs

1. Vérifier le fonctionnement sur XOR.
2. Évaluer TEN sur MNIST.
3. Comparer TEN à un MLP entraîné par backpropagation.
4. Mesurer accuracy, loss, temps, paramètres et stabilité multi-seed.
5. Réaliser des ablations sur la température et le nombre d'itérations de relaxation.

> Ce dépôt est un prototype expérimental. Les propriétés théoriques (conservation d'énergie, stabilité des gradients, avantage énergétique matériel, etc.) ne sont pas considérées comme démontrées par le simple fonctionnement du code.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Exécution

```bash
python experiments/run_xor.py
python experiments/run_mnist.py
python experiments/run_baseline.py
```

Les résultats sont enregistrés dans `results/`.

## Reproductibilité

Les expériences acceptent plusieurs seeds. Une comparaison scientifique doit rapporter au minimum moyenne et écart-type sur plusieurs exécutions.

## Structure

- `src/models/`: modèles TEN et baseline MLP
- `src/learning/`: relaxation et règle d'apprentissage
- `src/datasets/`: chargement des données
- `src/training/`: boucles d'entraînement
- `src/evaluation/`: métriques et benchmarks
- `experiments/`: points d'entrée des expériences
- `tests/`: tests unitaires
