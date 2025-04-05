import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# liste des regions
regions = [
    "Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", 
    "Centre-Val de Loire", "Corse", "Grand Est", "Hauts-de-France", 
    "Île-de-France", "Normandie", "Nouvelle-Aquitaine", 
    "Occitanie", "Pays de la Loire", "Provence-Alpes-Côte d'Azur"
]

# donnees par region
data = {
    "Macron": [27.5, 25.9, 31.3, 26.7, 18.4, 25.1, 24.6, 30.8, 25.9, 26.4, 23.2, 29.6, 23.5],
    "Le Pen": [24.0, 27.1, 18.8, 26.7, 28.6, 28.5, 33.3, 18.0, 27.9, 23.4, 24.6, 21.6, 27.7],
    "Mélenchon": [19.6, 17.9, 19.5, 19.0, 13.4, 16.0, 17.2, 30.2, 15.8, 19.9, 22.4, 19.0, 16.9],
    "Zemmour": [7.5, 6.3, 4.4, 6.0, 12.2, 7.2, 6.6, 8.2, 6.2, 6.1, 6.7, 5.4, 9.9],
    "Pécresse": [5.3, 5.8, 5.2, 5.6, 4.0, 5.9, 5.1, 7.5, 5.4, 5.1, 4.8, 5.8, 5.9],
    "Jadot": [5.2, 4.7, 6.5, 4.6, 3.8, 4.4, 3.1, 6.0, 4.0, 5.0, 5.2, 6.0, 4.3],
    "Lassalle": [3.5, 3.6, 3.1, 3.2, 3.2, 3.1, 2.0, 1.4, 3.1, 5.0, 5.1, 3.7, 2.2],
    "Roussel": [2.4, 3.0, 2.5, 2.9, 1.9, 2.8, 4.0, 2.3, 3.3, 2.5, 2.5, 2.5, 1.5],
    "Dupont-Aignan": [2.3, 2.5, 1.8, 2.8, 2.2, 2.5, 2.3, 2.0, 2.9, 2.0, 1.9, 1.8, 2.4],
    "Hidalgo": [1.7, 1.6, 2.3, 1.5, 0.9, 1.5, 1.1, 2.2, 1.6, 1.8, 1.9, 2.0, 1.2],
    "Poutou": [0.7, 0.7, 0.9, 0.7, 0.5, 0.6, 0.6, 0.9, 0.7, 1.0, 0.9, 0.8, 0.7],
    "Arthaud": [0.3, 0.6, 0.5, 0.3, 0.1, 0.4, 0.5, 0.5, 0.4, 0.4, 0.3, 0.4, 0.2],
    "Participation": [73.5, 74.1, 77.9, 74.2, 67.9, 73.2, 70.6, 76.5, 73.0, 77.0, 77.2, 77.8, 75.0],
    "Abstention": [26.5, 25.9, 22.1, 25.8, 32.1, 26.8, 29.4, 23.5, 27.0, 23.0, 22.8, 22.2, 25.0]
}

df = pd.DataFrame(data, index=regions)

# couleurs des candidats
colors = {
    "Macron": '#FFEB3B',      # Jaune
    "Le Pen": '#212121',      # Noir
    "Mélenchon": '#F44336',   # Rouge
    "Zemmour": '#3F51B5',     # Bleu foncé
    "Pécresse": '#2196F3',    # Bleu
    "Jadot": '#4CAF50',       # Vert
    "Lassalle": '#FF9800',    # Orange
    "Roussel": '#B71C1C',     # Rouge foncé
    "Dupont-Aignan": '#795548',  # Marron
    "Hidalgo": '#E91E63',     # Rose
    "Poutou": '#9C27B0',      # Violet
    "Arthaud": '#880E4F'      # Bordeaux
}

# liste des candidats sans participation/abstention
candidats = [c for c in df.columns if c not in ["Participation", "Abstention"]]

# graphique votes par region
plt.figure(figsize=(20, 10))
couleurs_candidats = []
for c in candidats:
    couleurs_candidats.append(colors[c])

df[candidats].plot(kind='bar', stacked=True, color=couleurs_candidats, figsize=(20, 10))
plt.title('Résultats premier tour 2022 par région')
plt.xlabel('Région')
plt.ylabel('%')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('resultats_regions.png')

# treemap resultats nationaux
try:
    import squarify
    
    # moyenne nationale
    moy_nationale = df[candidats].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(16, 10))
    squarify.plot(sizes=moy_nationale.values, 
                 label=[f"{c}: {v:.1f}%" for c, v in moy_nationale.items()],
                 color=[colors[c] for c in moy_nationale.index], 
                 alpha=0.8)
    plt.axis('off')
    plt.title('Répartition des votes au niveau national')
    plt.tight_layout()
    plt.savefig('treemap_national.png')
except:
    print("Squarify non installé - pip install squarify")

# heatmap top 5 candidats
top5 = df[candidats].mean().sort_values(ascending=False).head(5).index.tolist()
plt.figure(figsize=(14, 10))
sns.heatmap(df[top5], annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Scores des 5 principaux candidats')
plt.tight_layout()
plt.savefig('heatmap_top5.png')

# participation par region
plt.figure(figsize=(14, 8))
part_df = df[['Participation', 'Abstention']].sort_values('Participation', ascending=False)
part_df.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'])
plt.title('Participation et abstention par région')
plt.xlabel('Région')
plt.ylabel('%')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('participation.png')

# boxplot des scores
plt.figure(figsize=(14, 8))
sns.boxplot(data=df[candidats])
plt.title('Dispersion des scores entre régions')
plt.xlabel('Candidat')
plt.ylabel('Score (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('dispersion.png')

# cartes pour les 3 principaux candidats
try:
    # candidats top 3
    top3 = ["Macron", "Le Pen", "Mélenchon"]
    
    # carte de france
    france = gpd.read_file('regions-version-simplifiee.geojson')
    
    # nettoyage des noms
    france['nom'] = france['nom'].str.replace('-', ' ')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    for i, candidat in enumerate(top3):
        # prep données
        scores = pd.DataFrame({
            'region': regions,
            'score': df[candidat]
        })
        
        # fusion avec carte
        france['region'] = france['nom']
        fusion = france.merge(scores, on='region', how='left')
        
        # création colormap
        cmap = LinearSegmentedColormap.from_list('cmap', ['white', colors[candidat]])
        
        # plot sur subplot
        fusion.plot(column='score', ax=axes[i], legend=True,
                   cmap=cmap, 
                   legend_kwds={'label': f"{candidat} (%)",
                               'orientation': "horizontal"})
        
        axes[i].set_title(candidat)
        axes[i].set_axis_off()
    
    plt.tight_layout()
    plt.savefig('cartes_top3.png')
    
except Exception as e:
    print(f"Problème avec les cartes: {e}")
    print("Besoin de geopandas et du fichier geojson")

# graphique radar pour comparer regions
# choix de quelques régions importantes
regions_select = ["Île-de-France", "Hauts-de-France", "Provence-Alpes-Côte d'Azur", "Bretagne"]
donnees_radar = df.loc[regions_select, top5]

# angles pour radar
cats = top5
N = len(cats)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # fermer le polygone

# figure
plt.figure(figsize=(12, 12))
ax = plt.subplot(111, polar=True)

# couleurs regions
colors_regions = ['#FF9800', '#4CAF50', '#9C27B0', '#2196F3']

# tracer pour chaque region
for i, region in enumerate(regions_select):
    valeurs = donnees_radar.loc[region].values.tolist()
    valeurs += valeurs[:1]  # fermer
    ax.plot(angles, valeurs, label=region, color=colors_regions[i])
    ax.fill(angles, valeurs, alpha=0.1, color=colors_regions[i])

# labels et graduations
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats)
plt.yticks([10, 20, 30], ["10%", "20%", "30%"])
plt.grid(True)

# titre et légende
plt.title('Profils électoraux par région')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('radar_regions.png')

# analyse par blocs politiques
# définition des blocs
gauche = ["Mélenchon", "Roussel", "Jadot", "Hidalgo", "Poutou", "Arthaud"]
centre = ["Macron", "Pécresse"]
droite = ["Le Pen", "Zemmour", "Dupont-Aignan", "Lassalle"]

# création df pour les blocs
blocs_df = pd.DataFrame(index=regions)

# calcul des totaux par bloc
blocs_df["Gauche"] = df[gauche].sum(axis=1)
blocs_df["Centre"] = df[centre].sum(axis=1)
blocs_df["Droite"] = df[droite].sum(axis=1)

# graphique
plt.figure(figsize=(14, 10))
blocs_df.sort_values("Gauche", ascending=False).plot(
    kind='barh', 
    stacked=True, 
    color=['#F44336', '#FFEB3B', '#212121']
)
plt.title('Blocs politiques par région')
plt.xlabel('%')
plt.ylabel('Région')
plt.legend()
plt.tight_layout()
plt.savefig('blocs.png')

print("Visualisations terminées!")