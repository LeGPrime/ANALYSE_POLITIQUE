import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# dossier pour les résultats
resultats_dir = "resultats"
if not os.path.exists(resultats_dir):
    os.makedirs(resultats_dir)

# chargement du CSV
try:
    df = pd.read_csv('data/deputes-active.csv', encoding='utf-8')
    print(f"Dataset chargé ! {len(df)} députés trouvés")
except:
    try:
        df = pd.read_csv('deputes-active.csv', encoding='utf-8')
        print(f"c'est chargé {len(df)} députés trouvés")
    except:
        print("Fichier introuvable")
        exit()

# scores en valeurs numériques
score_columns = ['scoreParticipation', 'scoreParticipationSpecialite', 'scoreLoyaute', 'scoreMajorite']
for col in score_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

# TOP 10 DES DÉPUTÉS LES PLUS ASSIDUS
print("\n" + "="*30)
print("TOP 10 DES DÉPUTÉS LES PLUS ASSIDUS")
print("="*30)

df_participation = df[df['scoreParticipation'].notna() & (df['scoreParticipation'] > 0)]
if not df_participation.empty:
    top_assidus = df_participation.nlargest(10, 'scoreParticipation')[['nom', 'prenom', 'groupe', 'scoreParticipation']]
    print(top_assidus)
    
    plt.figure(figsize=(12, 6))
    plt.barh(
        top_assidus.apply(lambda x: f"{x['prenom']} {x['nom']}", axis=1),
        top_assidus['scoreParticipation'],
        color='#4CAF50'
    )
    
    plt.xlabel('Score de participation (%)')
    plt.title('Top 10 des députés les plus assidus')
    plt.tight_layout()
    plt.savefig(f'{resultats_dir}/top_assidus.png', dpi=300)
else:
    print("Pas assez de données pour l'analyse...")

# TOP 10 DES DÉPUTÉS LES MOINS LOYAUX
print("\n" + "="*30)
print("TOP 10 DES DÉPUTÉS LES MOINS LOYAUX")
print("="*30)

df_loyaute = df[df['scoreLoyaute'].notna() & (df['scoreLoyaute'] > 0)]
if not df_loyaute.empty:
    bottom_loyaute = df_loyaute.nsmallest(10, 'scoreLoyaute')[['nom', 'prenom', 'groupe', 'scoreLoyaute']]
    print(bottom_loyaute)
    
    plt.figure(figsize=(12, 6))
    plt.barh(
        bottom_loyaute.apply(lambda x: f"{x['prenom']} {x['nom']}", axis=1),
        bottom_loyaute['scoreLoyaute'],
        color='#E74C3C'
    )
    
    plt.xlabel('Score de loyauté (%)')
    plt.title('Les 10 députés les moins loyaux à leur groupe')
    plt.tight_layout()
    plt.savefig(f'{resultats_dir}/bottom_loyaute.png', dpi=300)
else:
    print("données manquantes pour l'analyse de loyauté")

# PARTICIPATION MOYENNE PAR GROUPE POLITIQUE
print("\n" + "="*30)
print("PARTICIPATION MOYENNE PAR GROUPE POLITIQUE")
print("="*30)

participation_groupe = df.groupby('groupe')['scoreParticipation'].apply(lambda x: x[x.notna() & (x > 0)].mean()).dropna()
if not participation_groupe.empty:
    participation_groupe = participation_groupe.sort_values(ascending=False)
    print(participation_groupe)
    
    plt.figure(figsize=(14, 8))
    participation_groupe.plot(kind='bar', color='#2196F3')
    
    plt.ylabel('Score de participation moyen (%)')
    plt.title('Participation moyenne par groupe politique')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{resultats_dir}/participation_groupe.png', dpi=300)
else:
    print("pas assez de données pour calculer la participation par groupe")

# LOYAUTÉ MOYENNE PAR GROUPE POLITIQUE
print("\n" + "="*30)
print("LOYAUTÉ MOYENNE PAR GROUPE POLITIQUE")
print("="*30)

loyaute_groupe = df.groupby('groupe')['scoreLoyaute'].apply(lambda x: x[x.notna() & (x > 0)].mean()).dropna()
if not loyaute_groupe.empty:
    loyaute_groupe = loyaute_groupe.sort_values(ascending=False)
    print(loyaute_groupe)
    
    plt.figure(figsize=(14, 8))
    loyaute_groupe.plot(kind='bar', color='#FF9800')
    
    plt.ylabel('Score de loyauté moyen (%)')
    plt.title('Loyauté moyenne par groupe politique')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{resultats_dir}/loyaute_groupe.png', dpi=300)
else:
    print("pas assez de données pour calculer la loyauté par groupe")

# RÉPARTITION DES DÉPUTÉS PAR GROUPE
print("\n" + "="*30)
print("RÉPARTITION DES DÉPUTÉS PAR GROUPE")
print("="*30)

repartition_groupe = df['groupe'].value_counts()
print(repartition_groupe)

plt.figure(figsize=(12, 8))
repartition_groupe.plot(kind='pie', autopct='%1.1f%%')
plt.title('Répartition des députés par groupe politique')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{resultats_dir}/repartition_groupe.png', dpi=300)

# RÉPARTITION HOMMES/FEMMES PAR GROUPE
print("\n" + "="*30)
print("RÉPARTITION HOMMES/FEMMES PAR GROUPE")
print("="*30)

genre_groupe = pd.crosstab(df['groupe'], df['civ'])
if 'Mme' in genre_groupe.columns and 'M.' in genre_groupe.columns:
    genre_groupe['pourcentage_femmes'] = genre_groupe['Mme'] / (genre_groupe['M.'] + genre_groupe['Mme']) * 100
    print(genre_groupe)
    
    plt.figure(figsize=(14, 8))
    genre_groupe['pourcentage_femmes'].sort_values(ascending=False).plot(kind='bar', color='#E91E63')
    
    plt.ylabel('Pourcentage de femmes (%)')
    plt.title('Pourcentage de femmes par groupe politique')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{resultats_dir}/femmes_groupe.png', dpi=300)
else:
    print("données insuffisantes pour l'analyse")

# COMPARAISON NOUVEAUX VS EXPÉRIMENTÉS
print("\n" + "="*30)
print("NOUVEAUX VS EXPÉRIMENTÉS")
print("="*30)

nouveaux = df[df['nombreMandats'] == 1]
experimentes = df[df['nombreMandats'] >= 3]

print(f"Nombre de nouveaux députés: {len(nouveaux)}")
print(f"Nombre de députés expérimentés: {len(experimentes)}")

scores_comparaison = pd.DataFrame({
    'Nouveaux': [
        nouveaux['scoreParticipation'].mean(),
        nouveaux['scoreLoyaute'].mean()
    ],
    'Expérimentés': [
        experimentes['scoreParticipation'].mean(),
        experimentes['scoreLoyaute'].mean()
    ]
}, index=['Participation', 'Loyauté'])

print(scores_comparaison)

plt.figure(figsize=(10, 6))
scores_comparaison.plot(kind='bar', color=['#FFEB3B', '#673AB7'])

plt.xlabel('Type de score')
plt.ylabel('Score moyen (%)')
plt.title('Comparaison entre nouveaux députés et députés expérimentés')
plt.legend(title='Expérience')
plt.tight_layout()
plt.savefig(f'{resultats_dir}/nouveaux_vs_experimentes.png', dpi=300)

# âge moyen par groupe
if 'age' in df.columns and df['age'].notna().any():
    print("\n" + "="*30)
    print("ÂGE MOYEN PAR GROUPE POLITIQUE")
    print("="*30)
    
    age_groupe = df.groupby('groupe')['age'].mean().dropna()
    if not age_groupe.empty:
        age_groupe = age_groupe.sort_values(ascending=False)
        print(age_groupe)
        
        plt.figure(figsize=(14, 8))
        age_groupe.plot(kind='bar', color='#607D8B')
        
        plt.ylabel('Âge moyen')
        plt.title('Âge moyen par groupe politique')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{resultats_dir}/age_moyen_groupe.png', dpi=300)
    else:
        print("Données insuffisantes pour l'analyse")
else:
    print("La colonne 'age' est manquante")

print("Analyse terminée", resultats_dir)