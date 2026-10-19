## Comment exécuter
1. Télécharger sarcos_inv.mat et sarcos_inv_test.mat (http://gaussianprocess.org/gpml/data/)
2. Les uploader comme dataset Kaggle nommé "sarcos-inverse-dynamics"
3. Ouvrir notebooks/sarcos_joint1_baseline_mlp.ipynb sur Kaggle (accélérateur : None / CPU)
4. Ajouter le dataset en Input, puis exécuter toutes les cellules (Run All)
Bibliothèques : numpy, pandas, scipy, scikit-learn, matplotlib

## Résultats (articulation 1, jeu de test)
| Modèle | nMSE | RMSE (N·m) |
|---|---|---|
| Régression linéaire | 0.0751 | 5.576 |
| MLP (256, 256) | 0.0107 | 2.105 |