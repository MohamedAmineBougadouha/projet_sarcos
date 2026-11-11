\# Projet 4 : Apprentissage de la dynamique inverse d'un bras robotique (SARCOS)



Module : IA / TASI (ML, DL, RL) pour la mécatronique, ENISO - 2026/2027

Binôme : Mohamed Amine Bougadouha et Achref Bouzidi



\## 1. Objectif

Prédire le couple moteur de l'\*\*articulation 1\*\* d'un bras robotique à 7 articulations à partir de \*\*21 entrées\*\* : positions, vitesses et accélérations des 7 articulations (dynamique inverse). Une régression linéaire sert de baseline, puis un MLP (réseau de neurones) doit la battre. L'évaluation se fait par le nMSE sur un jeu de test séparé.



\## 2. Données

SARCOS Robot Arm Inverse Dynamics : http://gaussianprocess.org/gpml/data/



| Fichier | Exemples | Colonnes |

|---|---|---|

| `sarcos\_inv.mat` (entraînement) | 44 484 | 28 |

| `sarcos\_inv\_test.mat` (test) | 4 449 | 28 |



\- Colonnes 0 à 6 : positions des 7 articulations

\- Colonnes 7 à 13 : vitesses

\- Colonnes 14 à 20 : accélérations

\- Colonnes 21 à 27 : couples moteurs (sorties)

\- Aucune valeur manquante.

\- Couple de l'articulation 1 (entraînement) : moyenne 13.65 N·m, écart-type 20.47 N·m, de -94.9 à 121.4 N·m.



\## 3. Méthode

1\. \*\*Découpage\*\* : sur le fichier d'entraînement, 90 % pour l'entraînement (40 035) et 10 % pour la validation (4 449), `random\_state=42`. Le fichier de test est utilisé \*\*une seule fois\*\*, à la fin, pour la note finale.

2\. \*\*Standardisation\*\* (`StandardScaler`) ajustée \*\*uniquement sur l'entraînement\*\*, pour éviter toute fuite de données. Elle est nécessaire car positions, vitesses et accélérations n'ont pas les mêmes échelles.

3\. \*\*Baseline\*\* : régression linéaire (`LinearRegression`).

4\. \*\*MLP\*\* (`MLPRegressor`) : activation ReLU, optimiseur Adam, taux d'apprentissage 1e-3, early stopping (`n\_iter\_no\_change=10`, `max\_iter=200`). Cinq architectures comparées sur la \*\*validation\*\* : (32), (64), (64,64), (128,128), (256,256).

5\. \*\*Sélection\*\* de l'architecture sur la validation, puis évaluation unique sur le test.



\## 4. Choix des métriques

\- \*\*nMSE\*\* (erreur quadratique moyenne normalisée) : MSE divisée par la variance du couple. 0 signifie parfait, 1 signifie équivalent à prédire toujours la moyenne (vérifié : prédire la moyenne donne 1.0002). Elle permet de comparer les modèles sur une échelle commune, sans dépendre de l'unité. C'est la métrique demandée par la fiche.

\- \*\*RMSE\*\* (en N·m) : erreur typique exprimée dans l'unité physique du couple, plus parlante pour un ingénieur.

\- \*\*Temps d'inférence\*\* (ms par échantillon) : pertinent pour une utilisation en commande.



\## 5. Résultats (articulation 1, jeu de test)



| Modèle | nMSE test | RMSE test (N·m) | Temps d'inférence (ms/échantillon) |

|---|---|---|---|

| Régression linéaire (baseline) | 0.0751 | 5.576 | 0.00016 |

| \*\*MLP (256, 256)\*\* | \*\*0.0107\*\* | \*\*2.105\*\* | 0.0069 |



Le MLP divise le nMSE par environ 7 et réduit l'erreur typique (RMSE) d'environ 62 %. Il explique environ 98.9 % de la variance du couple, contre 92.5 % pour la régression linéaire. Il est environ 40 fois plus lent à l'inférence, mais 0.0069 ms par échantillon reste très faible (temps mesuré sur un lot de 4 449 échantillons).



\### Comparaison des architectures du MLP (nMSE de validation)



| Architecture | nMSE validation | Epochs | Temps d'entraînement (s) |

|---|---|---|---|

| (32) | 0.0280 | 157 | 17.9 |

| (64) | 0.0241 | 169 | 24.1 |

| (64, 64) | 0.0197 | 96 | 24.7 |

| (128, 128) | 0.0157 | 124 | 57.4 |

| \*\*(256, 256)\*\* | \*\*0.0140\*\* | 84 | 91.0 |



L'erreur baisse quand le réseau grandit. La baseline linéaire a un nMSE de validation de 0.0771.



\### Fiabilité des résultats

\- Le nMSE de validation (0.0140) et celui de test (0.0107) sont du même ordre : pas de signe de surapprentissage.

\- La courbe d'apprentissage (`results/03\_courbe\_apprentissage.png`) chute rapidement puis se stabilise ; l'early stopping s'est arrêté à l'epoch 84.

\- La régression linéaire donne des scores très proches en validation (0.0771) et en test (0.0751).



\### Interprétation de la baseline

Les coefficients les plus importants de la régression linéaire sont l'accélération de l'articulation 1 (24.3), l'accélération de l'articulation 4 (10.0) et la position de l'articulation 1 (8.1). C'est cohérent avec la physique : le couple dépend surtout de l'inertie (couple proportionnel à l'accélération), avec un couplage venant de l'articulation 4 et un effet de la position (gravité). Un modèle linéaire ne peut pas représenter les effets non linéaires (Coriolis, forces centrifuges, frottements), ce qui explique le gain du MLP.



\## 6. Analyse des cas d'échec (MLP (256, 256), jeu de test)



\*\*Distribution des erreurs absolues :\*\* moyenne 1.501 N·m, médiane 1.094 N·m, 90 % des erreurs sous 3.331 N·m, maximum 20.035 N·m. L'erreur moyenne représente environ 7 % de l'écart-type du couple (20.47 N·m).



| Groupe (quartile) | Erreur moyenne selon \\|acc\_1\\| (N·m) | Erreur moyenne selon \\|vel\_1\\| (N·m) |

|---|---|---|

| Très faible | 1.301 | 1.522 |

| Faible | 1.364 | 1.640 |

| Forte | 1.534 | 1.529 |

| Très forte | 1.807 | 1.315 |



\*\*Observations :\*\*

\- \*\*Accélération\*\* : l'erreur augmente régulièrement avec l'accélération, de 1.301 à 1.807 N·m (environ +39 % entre le quartile le plus faible et le plus fort). Le modèle est donc moins précis dans les régimes à forte accélération. Hypothèses plausibles, non vérifiées ici : les effets non linéaires pèsent davantage et les exemples d'entraînement y sont plus rares (les accélérations vont jusqu'à environ 50 pour l'articulation 1, avec un écart-type de 9.6).

\- \*\*Vitesse\*\* : pas de tendance claire, l'erreur reste entre 1.3 et 1.6 N·m, et elle est même la plus faible dans le quartile des vitesses les plus fortes.

\- \*\*Erreurs extrêmes\*\* : elles existent (maximum 20.0 N·m) mais pèsent peu : les 1 % de pires échantillons représentent 6.0 % de l'erreur totale. La grande majorité des prédictions est donc précise, et l'erreur n'est pas dominée par quelques cas aberrants.

\- Les figures `results/04\_predit\_vs\_reel.png` (prédit contre réel) et `results/05\_analyse\_erreurs.png` (distribution et erreur selon l'accélération et la vitesse) illustrent ces points.



\## 7. Résultat complémentaire (non utilisé pour la sélection) : Random Forest

Pour information, un Random Forest (`RandomForestRegressor`, 100 arbres) a aussi été entraîné.



| Modèle | nMSE validation | nMSE test | RMSE test (N·m) |

|---|---|---|---|

| MLP (256, 256) | 0.0140 | 0.0107 | 2.105 |

| Random Forest (100 arbres) | 0.0251 | 0.0055 | 1.504 |



Observation : le Random Forest est meilleur que le MLP sur le test, mais moins bon sur la validation. Ces deux classements opposés montrent qu'un seul découpage ne suffit pas pour trancher. Conformément à la méthode, le modèle retenu reste le MLP, choisi sur la validation avant toute lecture du test. Un classement fiable demanderait une validation croisée et une recherche d'hyperparamètres pour les deux modèles (voir perspectives).



\## 8. Comment exécuter

1\. Télécharger `sarcos\_inv.mat` et `sarcos\_inv\_test.mat` depuis http://gaussianprocess.org/gpml/data/

2\. Sur Kaggle, créer un dataset nommé `sarcos-inverse-dynamics` contenant ces 2 fichiers.

3\. Ouvrir `notebooks/sarcos-joint1-baseline-mlp.ipynb` sur Kaggle, accélérateur \*\*None (CPU)\*\*, et ajouter le dataset en Input.

4\. Exécuter toutes les cellules (Run All). Durée totale : environ 10 à 15 minutes.

5\. Les figures et les tableaux sont enregistrés dans `results/`.



Bibliothèques : numpy, pandas, scipy, scikit-learn, matplotlib. Graine aléatoire : 42. Les fichiers `.mat` ne sont pas versionnés (voir `.gitignore`).



\## 9. Structure du dépôt

\- `notebooks/` : notebook complet

\- `results/` : figures 01 à 05 et tableaux CSV (`tableau\_final.csv`, `comparaison\_architectures.csv`)

\- `README.md`



\## 10. Limites et perspectives

\- Un seul découpage entraînement / validation et une seule graine : les résultats gagneraient à être moyennés sur plusieurs graines avec une validation croisée.

\- Une seule articulation est traitée (périmètre binôme). Perspectives : prédire les 7 couples, comparer rigoureusement Random Forest et MLP, étudier l'influence de la taille du jeu d'entraînement, et discuter du lien avec la commande par couple calculé.

\- Le temps d'inférence est mesuré sur un lot ; en commande réelle (un échantillon à la fois) il serait à remesurer.

