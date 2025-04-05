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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# charger spacy
try:
    nlp = spacy.load('fr_core_news_sm')
except:
    print("modèle spacy non trouvé")
    import subprocess, sys
    subprocess.call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load('fr_core_news_sm')

class AnalyseVoeux:
    def __init__(self, dossier='VoeuxPrésident'):
        self.dossier = dossier
        self.discours = {}
        self.presidents = {} 
        self.annees = {}
        
        # stopwords de nltk
        self.stop_words = set(stopwords.words('french'))
        
        # rajout de mots interdit
        self.stop_words.update([
            # vœux et temps
            'année', 'an', 'ans', 'mois', 'jour', 'jours', 
            'vœux', 'voeux', 'souhaite', 'souhaits', 'meilleurs', 
            'bonne', 'bonnes', 'fin', 'début',
            
            # formules
            'mesdames', 'messieurs', 'chers', 'françaises', 'français',
            'compatriotes', 'monsieur', 'madame', 'mes', 'nos', 'votre',
            
            # termes politiques
            'république', 'france', 'nation', 'pays', 'état', 
            'président', 'présidence',
            
            # verbes courants
            'est', 'sont', 'être', 'avoir', 'fait', 'faire', 'dire',
            'peut', 'veux', 'veut', 'voudrais',
            
            # mots basiques
            'a', 'cet', 'cette', 'ces', 'j', 'ai', 'c', 'est', 'd', 'qu', 
            'n', 's', 'on', 'm', 't', 'l', 'y', 'voir', 'ça', 'etc',
            'aussi', 'plus', 'tout', 'tous'
        ])
        
        # liste des présidents pour identification
        self.liste_presidents = {
            "VGE": "Valéry Giscard d'Estaing",
            "Giscard": "Valéry Giscard d'Estaing",
            "Mitterrand": "François Mitterrand",
            "Chirac": "Jacques Chirac",
            "Sarkozy": "Nicolas Sarkozy",
            "Hollande": "François Hollande",
            "Macron": "Emmanuel Macron"
        }
        
        # couleurs pour les graphiques
        self.couleurs = {
            "Valéry Giscard d'Estaing": "#3366CC",  # Bleu
            "François Mitterrand": "#FF66B2",       # Rose
            "Jacques Chirac": "#3399FF",            # Bleu clair
            "Nicolas Sarkozy": "#0033CC",           # Bleu foncé
            "François Hollande": "#FF3366",         # Rouge rosé
            "Emmanuel Macron": "#6633FF"            # Violet
        }
    
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
                
                # nom du fichier sans extension
                nom = fichier.replace('.txt', '')
                self.discours[nom] = texte
                
                # trouver l'année
                for i in range(len(nom)):
                    if nom[i:i+4].isdigit():
                        annee = nom[i:i+4]
                        self.annees[nom] = annee
                        break
                else:
                    self.annees[nom] = "Inconnue"
                
                # trouver le président
                pres_trouve = None
                for nom_court, nom_complet in self.liste_presidents.items():
                    if nom_court in nom:
                        pres_trouve = nom_complet
                        break
                
                # si pas trouvé, essayer avec l'année
                if not pres_trouve:
                    if self.annees[nom] != "Inconnue":
                        annee_int = int(self.annees[nom])
                        if 1974 <= annee_int <= 1981:
                            pres_trouve = "Valéry Giscard d'Estaing"
                        elif 1981 <= annee_int <= 1995:
                            pres_trouve = "François Mitterrand"
                        elif 1995 <= annee_int <= 2007:
                            pres_trouve = "Jacques Chirac"
                        elif 2007 <= annee_int <= 2012:
                            pres_trouve = "Nicolas Sarkozy"
                        elif 2012 <= annee_int <= 2017:
                            pres_trouve = "François Hollande"
                        elif annee_int >= 2017:
                            pres_trouve = "Emmanuel Macron"
                        else:
                            pres_trouve = "Inconnu"
                
                self.presidents[nom] = pres_trouve
        
        print(f"{len(self.discours)} discours chargés")
    
    def nettoyer_texte(self, texte):
        """nettoie le texte"""
        texte = texte.lower()
        texte = re.sub(r'[^\w\s]', ' ', texte)  # enlève ponctuation
        texte = re.sub(r'\d+', ' ', texte)      # enlève chiffres
        texte = re.sub(r'\s+', ' ', texte).strip()  # espaces multiples
        return texte
    
    def tokeniser_et_filtrer(self, texte):
        """tokenise et filtre les mots"""
        doc = nlp(texte)
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
            compteur = Counter(tokens)
            resultats[nom_discours] = dict(compteur.most_common(n))
        else:
            # tous les discours
            for nom, texte in self.discours.items():
                texte_propre = self.nettoyer_texte(texte)
                tokens = self.tokeniser_et_filtrer(texte_propre)
                compteur = Counter(tokens)
                resultats[nom] = dict(compteur.most_common(n))
        
        return self._convertir_en_dataframe(resultats)
    
    def _convertir_en_dataframe(self, resultats_dict):
        """conversion en dataframe"""
        dfs = []
        
        for nom, freq in resultats_dict.items():
            df = pd.DataFrame(list(freq.items()), columns=['mot', 'fréquence'])
            df['discours'] = nom
            df['président'] = self.presidents.get(nom, "Inconnu")
            df['année'] = self.annees.get(nom, "Inconnue")
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['mot', 'fréquence', 'discours', 'président', 'année'])
    
    def creer_nuage_mots(self, nom_discours=None, couleur="tricolore", show=True, save=False):
        """crée un nuage de mots"""
        from matplotlib.colors import LinearSegmentedColormap
        
        # choix de couleur
        if couleur == "tricolore":
            # bleu blanc rouge
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
            colormap = LinearSegmentedColormap.from_list("drapeau", colors, N=100)
        elif couleur in ["blue", "bleu"]:
            colormap = "Blues"
        elif couleur in ["red", "rouge"]:
            colormap = "Reds"
        else:
            colormap = "Blues"  
        
        if nom_discours and nom_discours in self.discours:
            # pour un discours
            texte = self.nettoyer_texte(self.discours[nom_discours])
            tokens = self.tokeniser_et_filtrer(texte)
            texte_filtre = ' '.join(tokens)
            
            # titre avec président et année
            president = self.presidents.get(nom_discours, "")
            annee = self.annees.get(nom_discours, "")
            titre = f'Vœux de {president} - {annee}'
            
            wc = WordCloud(width=1000, height=600, background_color='white', 
                         max_words=100, colormap=colormap)
            wc.generate(texte_filtre)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(titre)
            
            if save:
                plt.savefig(f'nuage_{nom_discours}.png')
            if show:
                plt.show()
            
        else:
            # pour tous les discours
            tous_tokens = []
            for nom, texte in self.discours.items():
                texte_propre = self.nettoyer_texte(texte)
                tokens = self.tokeniser_et_filtrer(texte_propre)
                tous_tokens.extend(tokens)
            
            texte_tous = ' '.join(tous_tokens)
            
            wc = WordCloud(width=1000, height=600, background_color='white', 
                         max_words=100, colormap=colormap)
            wc.generate(texte_tous)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de mots: Tous les vœux présidentiels')
            
            if save:
                plt.savefig('nuage_tous.png')
            if show:
                plt.show()
    
    def comparer_presidents(self, n=10, show=True, save=False):
        """compare les mots caractéristiques de chaque président"""
        # analyser mots par président
        mots_president = {}
        
        for nom, texte in self.discours.items():
            president = self.presidents.get(nom, "Inconnu")
            if president not in mots_president:
                mots_president[president] = []
            
            texte_propre = self.nettoyer_texte(texte)
            tokens = self.tokeniser_et_filtrer(texte_propre)
            mots_president[president].extend(tokens)
        
        # fréquences par président
        freq_president = {}
        for president, mots in mots_president.items():
            freq_president[president] = Counter(mots)
        
        # mots distinctifs
        mots_distinctifs = {}
        for president, freq in freq_president.items():
            # compteur pour tous les autres présidents
            autres_freq = Counter()
            for autre, autre_freq in freq_president.items():
                if autre != president:
                    autres_freq.update(autre_freq)
            
            # calcul du ratio
            specificite = {}
            for mot, count in freq.items():
                # normalisation
                freq_norm = count / sum(freq.values())
                autres_norm = autres_freq.get(mot, 0) / max(1, sum(autres_freq.values()))
                
                # calcul du ratio
                if autres_norm > 0:
                    ratio = freq_norm / autres_norm
                else:
                    ratio = float('inf')  # mot unique
                
                # filtrer mots rares
                if count >= 3:
                    specificite[mot] = ratio
            
            # tri des mots
            mots_distinctifs[president] = sorted(specificite.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # graphique
        fig, axes = plt.subplots(len(mots_distinctifs), 1, figsize=(10, 3*len(mots_distinctifs)))
        
        if len(mots_distinctifs) == 1:
            axes = [axes]
        
        for i, (president, mots) in enumerate(mots_distinctifs.items()):
            mots_list = [mot for mot, _ in mots]
            scores = [min(score, 10) if score != float('inf') else 10 for _, score in mots]
            
            couleur = self.couleurs.get(president, "#333333")
            axes[i].barh(mots_list, scores, color=couleur)
            axes[i].set_title(f"Mots de {president}")
            axes[i].set_xlabel("Spécificité")
            
        plt.tight_layout()
        
        if save:
            plt.savefig('mots_presidents.png')
        if show:
            plt.show()
        
        return mots_distinctifs
    
    def analyser_evolution_temporelle(self, show=True, save=False):
        """analyse l'évolution des mots au fil du temps"""
        # regrouper par année
        textes_annee = {}
        for nom, texte in self.discours.items():
            annee = self.annees.get(nom, "")
            if annee and annee != "Inconnue":
                if annee not in textes_annee:
                    textes_annee[annee] = []
                textes_annee[annee].append(texte)
        
        # fréquences par année
        freq_annee = {}
        for annee, textes in textes_annee.items():
            tous_tokens = []
            for texte in textes:
                texte_propre = self.nettoyer_texte(texte)
                tokens = self.tokeniser_et_filtrer(texte_propre)
                tous_tokens.extend(tokens)
            
            freq_annee[annee] = Counter(tous_tokens)
        
        # mots les plus fréquents sur toute la période
        tous_mots = Counter()
        for freq in freq_annee.values():
            tous_mots.update(freq)
        
        mots_communs = [mot for mot, _ in tous_mots.most_common(50)]
        
        # mots qui varient dans le temps
        mots_interessants = []
        for mot in mots_communs[:20]:
            # calcul de la variance
            valeurs = [freq.get(mot, 0) / sum(freq.values()) * 1000 for freq in freq_annee.values()]
            variance = np.var(valeurs)
            if variance > 0:
                mots_interessants.append(mot)
        
        # limiter à 8 mots
        mots_interessants = mots_interessants[:8]
        
        # graphique d'évolution
        plt.figure(figsize=(12, 8))
        
        # tracer pour chaque mot
        for mot in mots_interessants:
            annees = sorted(freq_annee.keys())
            valeurs = []
            
            for annee in annees:
                freq = freq_annee[annee]
                valeur = freq.get(mot, 0) / sum(freq.values()) * 1000
                valeurs.append(valeur)
            
            plt.plot(annees, valeurs, marker='o', linewidth=2, label=mot)
        
        # lignes pour changements de président
        changements = {
            "1981": "Mitterrand",
            "1995": "Chirac",
            "2007": "Sarkozy",
            "2012": "Hollande",
            "2017": "Macron"
        }
        
        for annee, president in changements.items():
            if annee in annees:
                plt.axvline(x=annee, linestyle='--', alpha=0.7, color='gray')
                plt.text(annee, plt.ylim()[1]*0.95, president, rotation=90, alpha=0.7)
        
        plt.title("Évolution des termes dans les vœux présidentiels")
        plt.xlabel("Année")
        plt.ylabel("Fréquence (pour 1000 mots)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig('evolution.png')
        if show:
            plt.show()
        
        return mots_interessants
    
    def analyser_similitude(self, show=True, save=False):
        """analyse la similitude entre discours"""
        # préparer les textes
        textes_propres = {}
        for nom, texte in self.discours.items():
            texte_propre = self.nettoyer_texte(texte)
            tokens = self.tokeniser_et_filtrer(texte_propre)
            textes_propres[nom] = ' '.join(tokens)
        
        # vectoriser
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(textes_propres.values())
        
        # matrice de similitude
        sim_matrix = cosine_similarity(X)
        
        # similitude par président
        presidents = list(set(self.presidents.values()))
        n_presidents = len(presidents)
        sim_presidents = np.zeros((n_presidents, n_presidents))
        
        # pour chaque paire de présidents
        for i, pres1 in enumerate(presidents):
            for j, pres2 in enumerate(presidents):
                # discours de ces présidents
                discours_pres1 = [nom for nom, p in self.presidents.items() if p == pres1]
                discours_pres2 = [nom for nom, p in self.presidents.items() if p == pres2]
                
                # similitude moyenne
                similitudes = []
                for d1 in discours_pres1:
                    for d2 in discours_pres2:
                        idx1 = list(textes_propres.keys()).index(d1)
                        idx2 = list(textes_propres.keys()).index(d2)
                        similitudes.append(sim_matrix[idx1, idx2])
                
                sim_presidents[i, j] = np.mean(similitudes)
        
        # visualisation
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_presidents, cmap='viridis')
        
        # labels
        plt.xticks(np.arange(n_presidents), presidents)
        plt.yticks(np.arange(n_presidents), presidents)
        
        # rotation
        plt.xticks(rotation=45, ha="right")
        
        # valeurs dans les cellules
        for i in range(n_presidents):
            for j in range(n_presidents):
                plt.text(j, i, f"{sim_presidents[i, j]:.2f}", ha="center", va="center", 
                       color="white" if sim_presidents[i, j] < 0.7 else "black")
        
        plt.title("Similitude entre présidents")
        plt.colorbar(label="Similitude")
        plt.tight_layout()
        
        if save:
            plt.savefig('similitude.png')
        if show:
            plt.show()
        
        return sim_presidents, presidents

# fonction principale
def analyser_voeux(dossier='VoeuxPrésident', save=False):
    print("Analyse des vœux présidentiels...")
    
    # création et chargement
    analyse = AnalyseVoeux(dossier)
    analyse.charger_discours()
    
    if not analyse.discours:
        print("Aucun discours trouvé. Vérifiez le dossier.")
        return
    
    # nuage de mots
    print("\n1. Nuage de mots global...")
    analyse.creer_nuage_mots(None, couleur="tricolore", save=save)
    
    # mots par président
    print("\n2. Mots caractéristiques par président...")
    mots_distinctifs = analyse.comparer_presidents(n=8, save=save)
    
    # affichage
    print("\nMots caractéristiques:")
    for president, mots in mots_distinctifs.items():
        print(f"\n{president}:")
        for mot, score in mots:
            score_str = f"{score:.2f}" if score != float('inf') else "∞"
            print(f"  - {mot} ({score_str})")
    
    # évolution temporelle
    print("\n3. Analyse de l'évolution...")
    mots_interessants = analyse.analyser_evolution_temporelle(save=save)
    
    print("\nMots qui évoluent:")
    print(", ".join(mots_interessants))
    
    # similitude
    print("\n4. Similitude entre présidents...")
    sim_matrix, presidents = analyse.analyser_similitude(save=save)
    
    print("\nSimilitude entre présidents:")
    for i, pres1 in enumerate(presidents):
        for j, pres2 in enumerate(presidents):
            if i < j:
                print(f"{pres1} et {pres2}: {sim_matrix[i, j]:.2f}")
    
    # analyse par période
    print("\n5. Analyse par période historique...")
    
    # périodes
    periodes = {
        "1974-1980": "Chocs pétroliers",
        "1981-1989": "Guerre froide",
        "1990-2000": "Post-Guerre froide",
        "2001-2008": "Post-11 septembre",
        "2009-2014": "Crise financière",
        "2015-2019": "Terrorisme",
        "2020-2024": "COVID"
    }
    
    # discours par période
    discours_periode = {}
    for nom, annee in analyse.annees.items():
        if annee != "Inconnue":
            annee_int = int(annee)
            for periode, desc in periodes.items():
                debut, fin = map(int, periode.split('-'))
                if debut <= annee_int <= fin:
                    if periode not in discours_periode:
                        discours_periode[periode] = []
                    discours_periode[periode].append(nom)
                    break
    
    # mots-clés par période
    print("\nMots-clés par période:")
    for periode, discours in discours_periode.items():
        if len(discours) > 0:
            tokens = []
            for nom in discours:
                texte = analyse.nettoyer_texte(analyse.discours[nom])
                mots = analyse.tokeniser_et_filtrer(texte)
                tokens.extend(mots)
            
            compteur = Counter(tokens)
            top_mots = compteur.most_common(8)
            
            print(f"\n{periode} ({periodes[periode]}):")
            for mot, freq in top_mots:
                print(f"  - {mot}: {freq}")
    
    print("\nAnalyse terminée")

# exécution
if __name__ == "__main__":
    analyser_voeux(save=True)