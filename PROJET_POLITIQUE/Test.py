import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

class AnalyseurDiscours:
    def __init__(self):
        self.dossier_resultats = "resultats"
        os.makedirs(self.dossier_resultats, exist_ok=True)
        
        self.presidents = {
            "degaulle": "Charles de Gaulle", "pompidou": "Georges Pompidou",
            "giscard": "Valéry Giscard d'Estaing", "mitterrand": "François Mitterrand",
            "chirac": "Jacques Chirac", "chirac2": "Jacques Chirac (2e mandat)",
            "sarkozy": "Nicolas Sarkozy", "hollande": "François Hollande",
            "macron": "Emmanuel Macron", "macron2": "Emmanuel Macron (2e mandat)"
        }
        
        self.mots_vides = {
            'le', 'de', 'un', 'à', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'qui', 'ne', 'sur', 'se', 'pas', 'plus', 'par', 'avec',
            'tout', 'faire', 'son', 'mettre', 'autre', 'on', 'mais', 'comme', 'dire',
            'elle', 'si', 'leur', 'une', 'quand', 'où', 'prendre', 'france', 'français',
            'françaises', 'république', 'nation', 'pays', 'état', 'premier', 'ministre'
        }

    def nettoyer_texte(self, texte):
        """Nettoie et tokenise le texte"""
        texte = re.sub(r'[^a-z\sàáâäæçèéêëìíîïòóôöùúûüÿ]', ' ', texte.lower())
        mots = re.sub(r'\s+', ' ', texte).strip().split()
        return [m for m in mots if m not in self.mots_vides and len(m) >= 3]

    def extraire_metadata(self, ligne):
        """Extrait les métadonnées d'une ligne"""
        return dict(re.findall(r'\*(\w+)_([^\s*]+)', ligne))

    def distance_labbe(self, texte1, texte2):
        """Calcule la distance de Labbé entre deux textes"""
        c1, c2 = Counter(texte1), Counter(texte2)
        t1, t2 = sum(c1.values()), sum(c2.values())
        
        if t1 == 0 or t2 == 0:
            return 1.0
            
        tous_mots = set(c1.keys()) | set(c2.keys())
        return sum(abs(c1[m]/t1 - c2[m]/t2) for m in tous_mots) / 2

    def parser_fichier(self, chemin_fichier):
        """pour parser le fichier et organise les données"""
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            contenu = f.read()
        
        donnees = {
            'presidents': defaultdict(list),
            'types': defaultdict(list),
            'decennies': defaultdict(list)
        }
        
        # pr traiter les discours
        for discours in re.split(r'(?=\*\*\*\*\s+)', contenu):
            if not discours.strip():
                continue
                
            lignes = discours.strip().split('\n')
            if not lignes[0].startswith('****'):
                continue
                
            metadata = self.extraire_metadata(lignes[0])
            mots = self.nettoyer_texte('\n'.join(lignes[1:]))
            
            # Groupement par président
            if 'qui' in metadata:
                for cle, nom in self.presidents.items():
                    if cle in metadata['qui'].lower():
                        donnees['presidents'][nom].extend(mots)
                        break
            
            # Groupement par type
            if 'type' in metadata:
                donnees['types'][metadata['type']].extend(mots)
            
            # Groupement par décennie
            if 'annee' in metadata and len(metadata['annee']) >= 4:
                decennie = f"{metadata['annee'][:3]}0s"
                donnees['decennies'][decennie].extend(mots)
        
        return donnees

    def creer_heatmap(self, textes_dict, titre, fichier):
        """Crée une heatmap des distances"""
        categories = sorted(textes_dict.keys())
        n = len(categories)
        matrice = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrice[i, j] = self.distance_labbe(
                        textes_dict[categories[i]], 
                        textes_dict[categories[j]]
                    )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrice, xticklabels=categories, yticklabels=categories,
                   annot=True, fmt=".2f", cmap='RdYlBu_r')
        plt.title(f"Distance de Labbé - {titre}", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dossier_resultats, fichier), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques
        distances = [matrice[i, j] for i in range(n) for j in range(i+1, n)]
        if distances:
            idx_min = np.unravel_index(np.argmin(matrice + np.eye(n)), matrice.shape)
            idx_max = np.unravel_index(np.argmax(matrice), matrice.shape)
            print(f"\n{titre} - Plus proches: {categories[idx_min[0]]} ↔ {categories[idx_min[1]]} ({matrice[idx_min]:.3f})")
            print(f"{titre} - Plus éloignés: {categories[idx_max[0]]} ↔ {categories[idx_max[1]]} ({matrice[idx_max]:.3f})")

    def analyser_lexique(self, textes_dict, nb_mots=15):
        """Analyse le lexique de chaque catégorie"""
        for nom, mots in sorted(textes_dict.items()):
            if not mots:
                continue
                
            frequences = Counter(mots).most_common(nb_mots)
            
            plt.figure(figsize=(10, 6))
            mots_freq, vals = zip(*frequences)
            
            bars = plt.barh(mots_freq[::-1], vals[::-1], color='steelblue', alpha=0.7)
            for bar in bars:
                plt.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{int(bar.get_width())}', va='center', fontsize=9)
            
            plt.title(f"Vocabulaire caractéristique - {nom}", fontsize=12, pad=20)
            plt.xlabel("Fréquence")
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_resultats, f"lexique_{nom.replace(' ', '_').lower()}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def analyser_corpus(self, chemin_fichier):
        """Analyse complète du corpus"""
        print("Analyse du corpus de discours présidentiels")
        print("=" * 50)
        
        donnees = self.parser_fichier(chemin_fichier)
        
        # analyse la similarité
        analyses = [
            (donnees['presidents'], "Présidents", "distances_presidents.png"),
            (donnees['types'], "Types de discours", "distances_types.png"),
            (donnees['decennies'], "Décennies", "distances_decennies.png")
        ]
        
        for textes, titre, fichier in analyses:
            if len(textes) > 1:
                print(f"\nAnalyse: {titre}")
                self.creer_heatmap(textes, titre, fichier)
                self.analyser_lexique(textes)
        
        print(f"\nAnalyse terminée")

# Utilisation
if __name__ == "__main__":
    analyseur = AnalyseurDiscours()
    analyseur.analyser_corpus("elysee.txt")