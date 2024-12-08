# building-perceptron

Ce projet applique un modèle d’apprentissage automatique pour prédire la présence de cancer du sein en utilisant les données de **Breast Cancer Wisconsin**. Le modèle choisi pour cette tâche est un **Perceptron**, un type de neurone artificiel, utilisé ici pour la classification binaire. Ce README décrit les étapes du projet, de l'analyse exploratoire à l'évaluation du modèle.

## Objectifs du Projet

1. **Récupération des Données** : Charger et préparer les données de **Breast Cancer Wisconsin**.
2. **Nettoyage et Analyse Exploratoire** : Assurer la qualité des données en appliquant un nettoyage rigoureux et en réalisant une analyse exploratoire approfondie.
3. **Réduction des données** : Utiliser une méthode de réduction des données pour simplifier les données, tout en préservant les informations pertinentes.
4. **Modélisation avec un Perceptron** : Développer un modèle basé sur le Perceptron pour classer les données en fonction du diagnostic de cancer.
5. **Évaluation et Conclusions** : Évaluer les performances du modèle et proposer des améliorations pour des résultats optimaux.

## Description des Données

Les données de **Breast Cancer Wisconsin** contiennent des informations cliniques sur des tumeurs du sein. Les variables incluent des mesures telles que la taille, la texture, la compacité, et la concavité des cellules, permettant de prédire si une tumeur est maligne ou bénigne.

## Méthodologie

### 1. Nettoyage des Données
   - Gérer les valeurs manquantes et s'assurer de la cohérence des données.
   - Normaliser ou standardiser les variables si nécessaire.

### 2. Analyse Exploratoire des Données (EDA)
   - Analyse des distributions des variables.
   - Visualisation des corrélations entre les variables.
   - Examen des relations entre les caractéristiques et le diagnostic.

### 3. Réduction de la Dimensionnalité
   - Gestion de la différence des classes
   - Sélection des paramètres fait par forward et backward selections.
   - Les variables sélectionnés sont enregistrés dans un fichier csv

### 4. Modélisation avec le Perceptron
Le modèle Perceptron est un classificateur linéaire simple qui met à jour ses poids en fonction de l'erreur de prédiction. Le modèle est entraîné en suivant les étapes suivantes:
   - Normaliser les caractéristiques d'entrée.
   - Initialiser les poids et le biais.
   - À chaque itération, calculer la sortie linéaire, appliquer la fonction d'activation et mettre à jour les poids et le biais en fonction de l'erreur de prédiction.
   - Arrêter l'entraînement si la perte est inférieure à un seuil spécifié.
   - Prédire le diagnostic de cancer (bénin ou malin) pour chaque observation.

### 5. Évaluation du Modèle
   - Calculer l'exactitude, le rappel, la précision et le F1-score pour évaluer les performances du modèle.
   - Comparer les résultats avec les attentes et identifier les zones d'amélioration.

## Resultats

Après avoir entraîné et évalué le modèle de Perceptron sur le jeu de données avec les caractéristiques sélectionnées à la fois par sélection avant et sélection arrière, nous avons obtenu les métriques suivantes :

### Forward Selected Features
- **Accuracy**: 0.9647
- **Precision**: 0.9286
- **Recall**: 1.0
- **F1-score**: 0.9630

### Backward Selected Features
- **Accuracy**: 0.9647
- **Precision**: 0.9286
- **Recall**: 1.0
- **F1-score**: 0.9630

## Conclusion

Le modèle Perceptron démontre des performances élevées sur l'ensemble de données, atteignant une précision d'environ 96,47 %. Les scores de précision et de rappel indiquent que le modèle est très efficace pour identifier correctement les instances positives tout en minimisant les faux positifs et les faux négatifs. Le score F1 équilibré confirme davantage la fiabilité et la robustesse du modèle.

Ces résultats suggèrent que le modèle Perceptron est bien adapté à cette tâche de classification, fournissant des prédictions précises et cohérentes. Les travaux futurs pourraient explorer des optimisations supplémentaires et des comparaisons avec d'autres modèles d'apprentissage automatique pour garantir la meilleure performance possible.

### Recommandations pour Améliorer le Modèle
**Amélioration du Modèle**
- Réseau de Perceptrons Multicouches (MLP) : Étant donné que le Perceptron est un modèle linéaire, tester un réseau de Perceptrons multicouches permettra de capturer des relations non linéaires dans les données. Cela peut améliorer la performance si les relations entre les caractéristiques sont complexes.
- Utilisez d'autre moyen de séléction des données: RandomForest

## Prérequis

- Python 3.x
- Bibliothèques : `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

