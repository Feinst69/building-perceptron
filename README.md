# building-perceptron

Ce projet applique un modèle d’apprentissage automatique pour prédire la présence de cancer du sein en utilisant les données de **Breast Cancer Wisconsin**. Le modèle choisi pour cette tâche est un **Perceptron**, un type de neurone artificiel, utilisé ici pour la classification binaire. Ce README décrit les étapes du projet, de l'analyse exploratoire à l'évaluation du modèle.

## Objectifs du Projet

1. **Récupération des Données** : Charger et préparer les données de **Breast Cancer Wisconsin**.
2. **Nettoyage et Analyse Exploratoire** : Assurer la qualité des données en appliquant un nettoyage rigoureux et en réalisant une analyse exploratoire approfondie.
3. **Réduction de la Dimensionnalité** : Utiliser une méthode de réduction de dimensionnalité pour simplifier les données, tout en préservant les informations pertinentes.
4. **Modélisation avec un Perceptron** : Développer un modèle basé sur le Perceptron pour classer les données en fonction du diagnostic de cancer.
5. **Évaluation et Conclusions** : Évaluer les performances du modèle et proposer des améliorations pour des résultats optimaux.

## Structure du Projet

- **Data Loading & Cleaning** : Importer les données et effectuer un prétraitement pour gérer les valeurs manquantes et les incohérences.
- **Exploratory Data Analysis (EDA)** : Visualiser et analyser les distributions des variables, les corrélations, et les éventuels schémas dans les données.
- **Dimensionality Reduction** : Appliquer une méthode (telle que **PCA** - Analyse en Composantes Principales) pour réduire le nombre de variables tout en conservant un maximum d'information.
- **Modélisation avec Perceptron** : Entraîner un modèle Perceptron pour la classification des données et ajuster les hyperparamètres.
- **Évaluation des Performances** : Utiliser des métriques telles que l'exactitude, la précision, le rappel, et le F1-score pour évaluer les performances du modèle.
- **Conclusion et Recommandations** : Résumer les résultats, évaluer l’efficacité du Perceptron pour cette tâche, et suggérer des améliorations potentielles.

## Description des Données

Les données de **Breast Cancer Wisconsin** contiennent des informations cliniques sur des tumeurs du sein. Les variables incluent des mesures telles que la taille, la texture, la compacité, et la concavité des cellules, permettant de prédire si une tumeur est maligne ou bénigne.

## Méthodologie

### 1. Nettoyage des Données
   - Gérer les valeurs manquantes et s'assurer de la cohérence des données.
   - Normaliser ou standardiser les variables si nécessaire.

### 2. Analyse Exploratoire des Données (EDA)
   - Analyse des distributions, détection des outliers.
   - Visualisation des corrélations entre les variables.
   - Examen des relations entre les caractéristiques et le diagnostic.

### 3. Réduction de la Dimensionnalité
   - Application de la méthode de réduction de dimensionnalité choisie (par exemple, PCA) pour simplifier les données.

### 4. Modélisation avec le Perceptron
   - Entraîner un modèle de Perceptron sur les données traitées.
   - Ajuster les hyperparamètres (tels que le taux d’apprentissage et le nombre d’itérations).
   - Prédire le diagnostic de cancer (bénin ou malin) pour chaque observation.

### 5. Évaluation du Modèle
   - Calculer l'exactitude, le rappel, la précision et le F1-score pour évaluer les performances du modèle.
   - Comparer les résultats avec les attentes et identifier les zones d'amélioration.

## Résultats et Conclusion



### Recommandations pour Améliorer le Modèle


## Prérequis

- Python 3.x
- Bibliothèques : `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

