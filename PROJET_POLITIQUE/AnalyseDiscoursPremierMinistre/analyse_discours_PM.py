import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys

# telecharge les nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# charge spacy
try:
    nlp = spacy.load('fr_core_news_sm')
except:
    print("Modèle spacy pas installé, je l'installe...")
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load('fr_core_news_sm')

class AnalyseDiscours:
    def __init__(self, dossier='DiscoursPM'):
        self.dossier = dossier
        self.discours = {}
        
        # mots à ignorer
        self.stop_words = set(stopwords.words('french'))
        self.stop_words.update([
            #  basiques
            'a', 'cet', 'cette', 'ces', 'j', 'ai', 'c', 'est', 'd', 'qu', 
            'n', 's', 'on', 'm', 't', 'l', 'y', 'voir', 'ça', 'etc',
            'monsieur', 'madame', 'mesdames', 'messieurs', 'chers', 'collègues',
            # termes politiques
            'assemblée', 'sénat', 'parlement', 'parlementaire', 'parlementaires',
            'nationale', 'national', 'république', 'républicain',
            'gouvernement', 'ministère', 'ministre', 'ministres',
            'président', 'députés', 'sénateur', 'majorité', 'opposition',
            'pays', 'nation', 'peuple', 'français', 'françaises', 'france',
            'état', 'politique', 'politiques', 'publique', 'publiques',
            'loi', 'lois', 'projet', 'projets', 'réforme', 'réformes',
            # verbes courants
            'souhaite', 'voudrais', 'voulons', 'veut', 'pense', 'crois', 'dire',
            'fait', 'faire', 'faut', 'prend', 'prendre', 'aujourd', 'hui',])
    
    def charger_discours(self):
        """charge les discours du dossier"""
        if not os.path.exists(self.dossier):
            print(f"Dossier {self.dossier} introuvable")
            return
        
        for fichier in os.listdir(self.dossier):
            if fichier.endswith('.txt'):
                chemin = os.path.join(self.dossier, fichier)
                try:
                    with open(chemin, 'r', encoding='utf-8') as f:
                        texte = f.read()
                except:
                    try:
                        with open(chemin, 'r', encoding='latin-1') as f:
                            texte = f.read()
                    except:
                        print(f"Impossible de lire {fichier}")
                        continue
                
                # enleve l'extension
                nom = fichier.replace('.txt', '')
                self.discours[nom] = texte
                print(f"Discours '{nom}' chargé ({len(texte)} caractères)")
        
        if not self.discours:
            print(f"Aucun fichier trouvé dans {self.dossier}")
    
    def nettoyer_texte(self, texte):
        # tout en minuscules
        texte = texte.lower()
        # enlever ponctuation
        texte = re.sub(r'[^\w\s]', ' ', texte)
        # enlever chiffres
        texte = re.sub(r'\d+', ' ', texte)
        # enlever espaces multiples
        texte = re.sub(r'\s+', ' ', texte).strip()
        return texte
    
    def tokeniser_et_filtrer(self, texte):
        doc = nlp(texte)
        # garde que les mots intéressants
        tokens = []
        for token in doc:
            if (token.lemma_ not in self.stop_words and 
                len(token.lemma_) > 2 and
                not token.is_punct and
                not token.is_space and
                not token.is_digit):
                tokens.append(token.lemma_)
        return tokens
    
    def analyser_frequence(self, nom_discours=None, n=20):
        """analyse la fréquence des mots"""
        resultats = {}
        
        if nom_discours and nom_discours in self.discours:
            # un seul discours
            texte = self.nettoyer_texte(self.discours[nom_discours])
            tokens = self.tokeniser_et_filtrer(texte)
            
            # compte les mots
            compteur = {}
            for mot in tokens:
                if mot in compteur:
                    compteur[mot] += 1
                else:
                    compteur[mot] = 1
            
            # tri par fréquence
            compteur_trie = sorted(compteur.items(), key=lambda x: x[1], reverse=True)
            resultats[nom_discours] = dict(compteur_trie[:n])
        else:
            # tous les discours
            for nom, texte in self.discours.items():
                texte_propre = self.nettoyer_texte(texte)
                tokens = self.tokeniser_et_filtrer(texte_propre)
                compteur = Counter(tokens)
                resultats[nom] = dict(compteur.most_common(n))
        
        # pour la visualisation
        return self._convertir_en_dataframe(resultats)
    
    def _convertir_en_dataframe(self, resultats_dict):
        dfs = []
        
        for nom, freq in resultats_dict.items():
            df_temp = pd.DataFrame(list(freq.items()), columns=['mot', 'fréquence'])
            df_temp['discours'] = nom
            dfs.append(df_temp)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['mot', 'fréquence', 'discours'])
    
    def creer_nuage_mots(self, nom_discours=None, sauvegarder=True, couleur="blue"):
        """créer nuage de mots pour un discours"""
        if not os.path.exists('resultats'):
            os.makedirs('resultats')
        
        # choisir pr la couleur
        if couleur == "blue":
            colormap = "Blues"
        elif couleur == "red":
            colormap = "Reds"
        elif couleur == "green":
            colormap = "Greens"
        else:
            colormap = "Blues"  # par défaut
            
        if nom_discours and nom_discours in self.discours:
            # nuage pour un discours
            texte = self.nettoyer_texte(self.discours[nom_discours])
            tokens = self.tokeniser_et_filtrer(texte)
            texte_filtre = ' '.join(tokens)
            
            wc = WordCloud(width=1200, height=800, background_color='white', 
                         max_words=150, colormap=colormap)
            wc.generate(texte_filtre)
            
            plt.figure(figsize=(16, 10))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Nuage de mots: {nom_discours}')
            
            if sauvegarder:
                plt.savefig(f'resultats/nuage_{nom_discours}.png')
            plt.show()
        else:
            # nuage pour tous
            tous_tokens = []
            for nom, texte in self.discours.items():
                texte_propre = self.nettoyer_texte(texte)
                tokens = self.tokeniser_et_filtrer(texte_propre)
                tous_tokens.extend(tokens)
            
            texte_tous = ' '.join(tous_tokens)
            
            wc = WordCloud(width=1200, height=800, background_color='white', 
                         max_words=150, colormap=colormap)
            wc.generate(texte_tous)
            
            plt.figure(figsize=(16, 10))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de mots: Tous les discours')
            
            if sauvegarder:
                plt.savefig('resultats/nuage_tous.png')
            plt.show()
    
    def visualiser_frequence(self, nom_discours=None, n=15, sauvegarder=True):
        """graphique des fréquences"""
        df = self.analyser_frequence(nom_discours, n)
        
        # crée le dossier si besoin
        if not os.path.exists('resultats'):
            os.makedirs('resultats')
        
        if nom_discours and nom_discours in self.discours:
            # graphique pour un discours
            plt.figure(figsize=(14, 8))
            
            # filtre et tri
            df_filtre = df[df['discours'] == nom_discours]
            df_filtre = df_filtre.sort_values('fréquence', ascending=True)
            
            # graphique à barres horizontales
            plt.barh(df_filtre['mot'], df_filtre['fréquence'], color='royalblue')
            
            plt.xlabel('Fréquence')
            plt.ylabel('Mots')
            plt.title(f'Fréquence des mots: {nom_discours}')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if sauvegarder:
                plt.savefig(f'resultats/freq_{nom_discours}.png')
            plt.show()
        else:
            # graphique comparatif
            plt.figure(figsize=(16, 10))
            
            # top mots tous discours confondus
            top_mots = df.groupby('mot')['fréquence'].sum().nlargest(n).index.tolist()
            df_filtre = df[df['mot'].isin(top_mots)]
            
            # graphique en forme de tableau
            pivot = df_filtre.pivot(index='mot', columns='discours', values='fréquence').fillna(0)
            pivot = pivot.reindex(index=pivot.sum(axis=1).sort_values(ascending=False).index)
            
            # style
            plt.style.use('seaborn-v0_8-white')
            pivot.plot(kind='barh', figsize=(16, 10))
            
            plt.xlabel('Fréquence')
            plt.ylabel('Mots')
            plt.title('Comparaison des mots entre discours')
            plt.legend(title='Discours', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if sauvegarder:
                plt.savefig('resultats/comparaison.png')
            plt.show()
    
    def analyser_par_periode(self):
        """analyse l'évolution par année"""
        # cherche des années dans les noms
        annees = {}
        for nom in self.discours.keys():
            match = re.search(r'(\d{4})', nom)
            if match:
                annee = match.group(1)
                if annee not in annees:
                    annees[annee] = []
                annees[annee].append(nom)
        
        if not annees:
            print("Pas d'années trouvées dans les noms")
            return
        
        # analyse par année
        resultats_periodes = {}
        for annee, discours_list in sorted(annees.items()):
            tous_tokens = []
            for nom in discours_list:
                texte_propre = self.nettoyer_texte(self.discours[nom])
                tokens = self.tokeniser_et_filtrer(texte_propre)
                tous_tokens.extend(tokens)
            
            compteur = Counter(tous_tokens)
            resultats_periodes[annee] = dict(compteur.most_common(20))
        
        # trouve les mots intéressants
        mots_communs = set()
        for freq in resultats_periodes.values():
            mots_communs.update(set(freq.keys()))
        
        # calcule les scores d'intérêt
        mots_analyses = []
        for mot in mots_communs:
            # compte dans combien de périodes on trouve ce mot
            count_periodes = 0
            for freq in resultats_periodes.values():
                if mot in freq:
                    count_periodes += 1
            
            if count_periodes >= 3:
                # calcul du score
                scores = []
                for freq in resultats_periodes.values():
                    scores.append(freq.get(mot, 0))
                
                # utilise variance et moyenne
                var = np.var(scores)
                moy = np.mean(scores)
                score = moy * (1 + var)
                mots_analyses.append((mot, score))
        
        # tri par score
        mots_analyses.sort(key=lambda x: x[1], reverse=True)
        top_mots = [m[0] for m in mots_analyses[:8]]
        
        # création du graphique
        plt.figure(figsize=(16, 10))
        
        # pour chaque mot intéressant
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_mots)))
        for i, mot in enumerate(top_mots):
            x = []  # années
            y = []  # fréquences
            
            for annee, freq in sorted(resultats_periodes.items()):
                x.append(annee)
                y.append(freq.get(mot, 0))
            
            plt.plot(x, y, marker='o', linewidth=2.5, markersize=8, 
                    label=mot, color=colors[i])
        
        plt.xlabel('Année')
        plt.ylabel('Fréquence')
        plt.title('Évolution des termes au fil du temps')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # zone grisée pour période intéressante
        plt.axvspan('2012', '2017', alpha=0.1, color='gray')
        
        plt.tight_layout()
        
        if not os.path.exists('resultats'):
            os.makedirs('resultats')
        plt.savefig('resultats/evolution.png')
        plt.show()
        
        # heatmap des périodes
        self._creer_heatmap_periodes(resultats_periodes)
    
    def _creer_heatmap_periodes(self, resultats):
        """crée une heatmap des termes par période"""
        # récupère les mots importants
        tous_mots = set()
        for freq in resultats.values():
            # top 15 de chaque période
            for mot in list(freq.keys())[:15]:
                tous_mots.add(mot)
        
        mots_list = list(tous_mots)
        periodes = sorted(resultats.keys())
        
        # limite à 20 mots max
        if len(mots_list) > 20:
            # prend les plus fréquents
            moyennes = {}
            for mot in mots_list:
                scores = []
                for periode in periodes:
                    scores.append(resultats[periode].get(mot, 0))
                moyennes[mot] = sum(scores) / len(scores)
            
            # tri par moyenne
            mots_tries = sorted(moyennes.items(), key=lambda x: x[1], reverse=True)
            mots_list = [m[0] for m in mots_tries[:20]]
        
        # création de la matrice
        data = np.zeros((len(mots_list), len(periodes)))
        for i, mot in enumerate(mots_list):
            for j, periode in enumerate(periodes):
                if mot in resultats[periode]:
                    data[i, j] = resultats[periode][mot]
        
        # création de la heatmap
        plt.figure(figsize=(14, 12))
        
        # affichage
        plt.imshow(data, cmap=plt.cm.YlOrRd)
        
        # labels
        plt.yticks(range(len(mots_list)), mots_list)
        plt.xticks(range(len(periodes)), periodes, rotation=45)
        
        plt.colorbar(label='Fréquence')
        plt.title('Fréquence des termes par période')
        
        # valeurs dans les cellules
        for i in range(len(mots_list)):
            for j in range(len(periodes)):
                if data[i, j] > 0:
                    plt.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', 
                           color='black' if data[i, j] < 15 else 'white')
        
        plt.tight_layout()
        plt.savefig('resultats/heatmap.png')
        plt.show()
    
    def analyser_par_locuteur(self):
        """analyse par locuteur (premier ministre)"""
        # extraction des noms
        locuteurs = {}
        for nom in self.discours.keys():
            # prend tout avant le premier underscore
            if '_' in nom:
                locuteur = nom.split('_')[0]
            else:
                locuteur = nom  # si pas d'underscore
                
            if locuteur not in locuteurs:
                locuteurs[locuteur] = []
            locuteurs[locuteur].append(nom)
        
        # analyse par locuteur
        resultats_locuteurs = {}
        for locuteur, discours_list in locuteurs.items():
            tokens = []
            for nom in discours_list:
                texte_propre = self.nettoyer_texte(self.discours[nom])
                mots = self.tokeniser_et_filtrer(texte_propre)
                tokens.extend(mots)
            
            compteur = Counter(tokens)
            resultats_locuteurs[locuteur] = dict(compteur.most_common(15))
        
        # création du graphique
        plt.figure(figsize=(16, 12))
        
        # quels mots comparer?
        top_mots = set()
        for freq in resultats_locuteurs.values():
            for mot in list(freq.keys())[:10]:  # top 10 de chaque locuteur
                top_mots.add(mot)
        
        # limite à 15 mots
        top_mots = list(top_mots)[:15]
        
        # préparation du graphique
        largeur = 0.8 / len(resultats_locuteurs)
        for i, (locuteur, freq) in enumerate(resultats_locuteurs.items()):
            # valeurs pour ce locuteur
            y = []
            for mot in top_mots:
                y.append(freq.get(mot, 0))
            
            # position des barres
            x = np.arange(len(top_mots))
            offset = i * largeur - (len(resultats_locuteurs) - 1) * largeur / 2
            
            # graphique
            plt.bar(x + offset, y, largeur, label=locuteur)
        
        plt.xticks(range(len(top_mots)), top_mots, rotation=45, ha='right')
        plt.ylabel('Fréquence')
        plt.title('Mots fréquents par Premier ministre')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # graphique radar aussi
        self._creer_radar_locuteurs(resultats_locuteurs)
        
        plt.tight_layout()
        plt.savefig('resultats/locuteurs.png')
        plt.show()
    
    def _creer_radar_locuteurs(self, resultats):
        """graphique radar des styles des locuteurs"""
        # cherche les mots les plus discriminants
        tous_mots = set()
        for freq in resultats.values():
            tous_mots.update(freq.keys())
        
        # calcul des variances
        variances = {}
        for mot in tous_mots:
            values = []
            for freq in resultats.values():
                values.append(freq.get(mot, 0))
            
            # ignore les mots trop rares
            if sum(values) > 10:
                variances[mot] = np.var(values)
        
        # mots les plus discriminants
        mots_top = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:8]
        mots_discriminants = [m[0] for m in mots_top]
        
        # config pour le radar
        angles = np.linspace(0, 2*np.pi, len(mots_discriminants), endpoint=False)
        angles = np.append(angles, angles[0])  # fermer le cercle
        
        # figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # couleurs
        colors = plt.cm.tab10(np.linspace(0, 1, len(resultats)))
        
        # trace pour chaque locuteur
        for i, (locuteur, freq) in enumerate(resultats.items()):
            values = [freq.get(mot, 0) for mot in mots_discriminants]
            values.append(values[0])  # fermer
            
            ax.plot(angles, values, linewidth=2, label=locuteur, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

# script principal
if __name__ == "__main__":
    analyseur = AnalyseDiscours()
    analyseur.charger_discours()
    
    if not analyseur.discours:
        print("Vérifiez le dossier 'DiscoursPM'.")
        exit()
    
    # liste des discours
    print("\nDiscours disponibles:")
    for i, nom in enumerate(sorted(analyseur.discours.keys()), 1):
        print(f"{i}. {nom}")
    
    # menu "interractif"
    while True:
        print("\n=== Menu d'analyse ===")
        print("1. Analyser la fréquence des mots")
        print("2. Créer un nuage de mots")
        print("3. Visualiser les fréquences")
        print("4. Analyser par période")
        print("5. Analyser par locuteur")
        print("6. Analyse complète")
        print("0. Quitter")
        
        choix = input("\nVotre choix: ")
        
        if choix == '0':
            break
        elif choix == '1':
            nom = input("Nom du discours (vide pour tous): ")
            n = int(input("Nombre de mots à afficher: "))
            resultat = analyseur.analyser_frequence(nom if nom else None, n)
            print(resultat)
        elif choix == '2':
            nom = input("Nom du discours (vide pour tous): ")
            couleur = input("Couleur (bleu, rouge, vert) [par défaut: bleu]: ") or "bleu"
            analyseur.creer_nuage_mots(nom if nom else None, couleur=couleur)
        elif choix == '3':
            nom = input("Nom du discours (vide pour comparer tous): ")
            n = int(input("Nombre de mots à afficher: "))
            analyseur.visualiser_frequence(nom if nom else None, n)
        elif choix == '4':
            analyseur.analyser_par_periode()
        elif choix == '5':
            analyseur.analyser_par_locuteur()
        
        elif choix == '6':
            print("Analyse complète...")
            analyseur.analyser_frequence(None, 20)
            analyseur.creer_nuage_mots(None)
            analyseur.visualiser_frequence(None)
            analyseur.analyser_par_periode()
            analyseur.analyser_par_locuteur()
            print("Analyse terminé. Résultats dans le dossier 'resultats'.")
        else:
            print("Choix invalide")
