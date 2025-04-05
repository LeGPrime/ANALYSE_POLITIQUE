import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# lecture du csv
df = pd.read_csv('resultats_premier_tour_2022_departements.csv')

# liste des candidats
candidats = ['arthaud', 'roussel', 'macron', 'lassalle', 'lepen', 'zemmour', 
             'melenchon', 'hidalgo', 'jadot', 'pecresse', 'poutou', 'dupont_aignan']

# calcul des pourcentages
for c in candidats:
    df[f'{c}_pct'] = df[c] / df['exprimes'] * 100

df['participation_pct'] = df['votants'] / df['inscrits'] * 100
df['abstention_pct'] = 100 - df['participation_pct']  

# pour les cartes
df['code_dept'] = df['code_departement'].astype(str)
for i in range(len(df)):
    if len(df.loc[i, 'code_dept']) == 1:
        df.loc[i, 'code_dept'] = '0' + df.loc[i, 'code_dept']

# couleurs des partis
colors = {
    "macron_pct": '#FFEB3B', "lepen_pct": '#212121', "melenchon_pct": '#F44336', 
    "zemmour_pct": '#3F51B5', "pecresse_pct": '#2196F3', "jadot_pct": '#4CAF50',
    "lassalle_pct": '#FF9800', "roussel_pct": '#B71C1C', "dupont_aignan_pct": '#795548',
    "hidalgo_pct": '#E91E63', "poutou_pct": '#9C27B0', "arthaud_pct": '#880E4F'
}

candidats_pct = [f'{c}_pct' for c in candidats]

# top 10 des départements les plus peuplés
top_depts = df.sort_values(by='inscrits', ascending=False).head(10)
data = top_depts[candidats_pct].copy()
data.index = top_depts['nom_departement']

plt.figure(figsize=(16, 10))
data.plot(kind='bar', stacked=True, color=[colors[col] for col in candidats_pct], figsize=(16, 10))
plt.title('Résultats présidentielles 2022 - Top 10 départements')
plt.xlabel('Département')
plt.ylabel('%')
plt.xticks(rotation=45, ha='right')
plt.legend(labels=[c.replace('_pct', '') for c in candidats_pct], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('resultats_top10.png')

# scores moyens des candidats
moy = {}
for c in candidats_pct:
    moy[c] = df[c].mean()

sorted_moy = sorted(moy.items(), key=lambda x: x[1], reverse=True)
top5 = [m[0] for m in sorted_moy[:5]]

# departments random pour heatmap
rdm_depts = df.sample(15, random_state=42)['nom_departement'].tolist()
heatmap_data = df[df['nom_departement'].isin(rdm_depts)][['nom_departement'] + top5].copy()
heatmap_data.set_index('nom_departement', inplace=True)
heatmap_data.columns = [c.replace('_pct', '') for c in heatmap_data.columns]

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Scores top 5 candidats')
plt.tight_layout()
plt.savefig('heatmap_top5.png')

# tentative de faire des cartes
try:
    # chargement des données géo
    geo = gpd.read_file('departements-version-simplifiee.geojson')
    
    # gerer les codes départements 
    geo['code'] = geo['code'].astype(str).str.zfill(2)
    
    # top 3 candidats
    top3 = [m[0] for m in sorted_moy[:3]]
    
    # graph
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
    for i, candidat in enumerate(top3):
        score_data = df[['code_dept', candidat]].copy()
        score_data.columns = ['code', 'score']
        
        merged = geo.merge(score_data, on='code', how='left')
        
        cmap = LinearSegmentedColormap.from_list('cmap', ['white', colors[candidat]])
        
        merged.plot(column='score', ax=ax[i], legend=True,
                  cmap=cmap,
                  legend_kwds={'label': f"{candidat.replace('_pct', '')} (%)",
                              'orientation': "horizontal"})
        
        ax[i].set_title(f'{candidat.replace("_pct", "")}')
        ax[i].set_axis_off()
    
    plt.tight_layout()
    plt.savefig('cartes_top3.png')
    
    # carte du gagnant par dept
    plt.figure(figsize=(14, 12))
    
    # trouver les gagnants
    df['gagnant'] = df[candidats_pct].idxmax(axis=1)
    df['score_max'] = df[candidats_pct].max(axis=1)
    
    gagnants_data = df[['code_dept', 'gagnant', 'score_max']]
    merged = geo.merge(gagnants_data, left_on='code', right_on='code_dept', how='left')
    
    ax = plt.gca()
    
    # colorier les départements selon le gagnant
    for candidat in candidats_pct:
        if candidat in merged['gagnant'].values:
            subset = merged[merged['gagnant'] == candidat]
            subset.plot(ax=ax, color=colors[candidat])
    
    # légende
    patches = []
    for candidat in candidats_pct:
        if candidat in merged['gagnant'].values:
            nom = candidat.replace('_pct', '')
            patch = mpatches.Patch(color=colors[candidat], label=nom)
            patches.append(patch)
    
    plt.legend(handles=patches, loc='lower left')
    plt.title('Candidat en tête par département')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('carte_gagnants.png')
    
except Exception as e:
    print(f"Problème avec les cartes: {e}")
    print("Besoin d'installer geopandas et d'avoir le fichier geojson")

# participation
plt.figure(figsize=(14, 8))
participation = df[['nom_departement', 'participation_pct', 'abstention_pct']].copy()
participation = participation.sort_values('participation_pct', ascending=False).head(20)
participation.set_index('nom_departement', inplace=True)
participation.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'])
plt.title('Participation - Top 20 départements')
plt.xlabel('Département')
plt.ylabel('%')
plt.xticks(rotation=45, ha='right')
plt.legend(['Participation', 'Abstention'])
plt.tight_layout()
plt.savefig('participation_top20.png')

# dispersion scores
plt.figure(figsize=(14, 8))
boxplot_data = df[candidats_pct].copy()
boxplot_data.columns = [c.replace('_pct', '') for c in boxplot_data.columns]
sns.boxplot(data=boxplot_data)
plt.title('Dispersion des scores')
plt.xlabel('Candidat')
plt.ylabel('Score (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('dispersion.png')

# écart macron/lepen
plt.figure(figsize=(14, 10))
df['ecart_ml'] = df['macron_pct'] - df['lepen_pct']
ecarts = df[['nom_departement', 'ecart_ml']].sort_values('ecart_ml')
extremes = pd.concat([ecarts.head(10), ecarts.tail(10)])
plt.barh(extremes['nom_departement'], extremes['ecart_ml'],
        color=['#F44336' if x < 0 else '#4CAF50' for x in extremes['ecart_ml']])
plt.title('Écarts Macron - Le Pen')
plt.xlabel('Écart (%)')
plt.axvline(x=0, color='black', linestyle='-')
plt.tight_layout()
plt.savefig('ecart_macron_lepen.png')

# blocs politiques

gauche = ['melenchon_pct', 'roussel_pct', 'jadot_pct', 'hidalgo_pct', 'poutou_pct', 'arthaud_pct']
centre = ['macron_pct', 'pecresse_pct']
droite = ['lepen_pct', 'zemmour_pct', 'dupont_aignan_pct', 'lassalle_pct']

# calcul des totaux
df['Gauche'] = df[gauche].sum(axis=1)
df['Centre'] = df[centre].sum(axis=1)
df['Droite'] = df[droite].sum(axis=1)

# top 20 départements
top20 = df.sort_values('exprimes', ascending=False).head(20)
blocs_data = top20[['nom_departement', 'Gauche', 'Centre', 'Droite']].copy()
blocs_data.set_index('nom_departement', inplace=True)
blocs_data.plot(kind='barh', stacked=True, color=['#F44336', '#FFEB3B', '#212121'], figsize=(14, 10))
plt.title('Blocs politiques - Top 20 départements')
plt.xlabel('%')
plt.ylabel('Département')
plt.legend()
plt.tight_layout()
plt.savefig('blocs_politiques.png')

# corrélations
plt.figure(figsize=(12, 10))
var_corr = ['participation_pct', 'macron_pct', 'lepen_pct', 'melenchon_pct', 
           'zemmour_pct', 'jadot_pct', 'pecresse_pct']
corr = df[var_corr].corr()
corr.columns = [c.replace('_pct', '') for c in var_corr]
corr.index = [c.replace('_pct', '') for c in var_corr]
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Corrélations')
plt.tight_layout()
plt.savefig('correlations.png')

print("Analyse terminée")