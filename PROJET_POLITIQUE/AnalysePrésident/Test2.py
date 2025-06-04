import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap

class AnalyseurAvance:
    def __init__(self):
        self.dossier_resultats = "ResultatsTest2"
        os.makedirs(self.dossier_resultats, exist_ok=True)
        
        #  présidents
        self.presidents = {
            "degaulle": "Charles de Gaulle", "pompidou": "Georges Pompidou",
            "giscard": "Valéry Giscard d'Estaing", "mitterrand": "François Mitterrand",
            "chirac": "Jacques Chirac", "chirac2": "Jacques Chirac (2e mandat)",
            "sarkozy": "Nicolas Sarkozy", "hollande": "François Hollande",
            "macron": "Emmanuel Macron", "macron2": "Emmanuel Macron (2e mandat)"
        }
        
        # Couleurs président
        self.couleurs_presidents = {
            "Charles de Gaulle": "#003399", "Georges Pompidou": "#8b0000",
            "Valéry Giscard d'Estaing": "#4682b4", "François Mitterrand": "#ff69b4",
            "Jacques Chirac": "#1e90ff", "Jacques Chirac (2e mandat)": "#006400",
            "Nicolas Sarkozy": "#ff7f50", "François Hollande": "#ff4500",
            "Emmanuel Macron": "#9370db", "Emmanuel Macron (2e mandat)": "#800080"
        }
        
        # stop words
        self.mots_vides = {
            'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'à', 'il', 'ils', 
            'elle', 'elles', 'être', 'et', 'en', 'avoir', 'que', 'qui', 'quoi', 
            'pour', 'dans', 'ce', 'cette', 'ces', 'ne', 'sur', 'se', 'pas', 'plus', 
            'par', 'avec', 'tout', 'tous', 'toute', 'toutes', 'faire', 'son', 'sa', 
            'ses', 'leur', 'leurs', 'mettre', 'autre', 'autres', 'on', 'mais', 
            'comme', 'dire', 'si', 'quand', 'où', 'prendre', 'france', 'français', 
            'françaises', 'république', 'nation', 'pays', 'état', 'premier', 
            'ministre', 'monsieur', 'madame', 'président'
        }
        
        # mots par thème
        self.themes = {
            'Économie': [
                'économie', 'économique', 'croissance', 'crise', 'chômage', 'emploi',
                'budget', 'dette', 'impôt', 'entreprise', 'industrie', 'marché',
                'investissement', 'commerce', 'salaire', 'banque', 'financier'
            ],
            'Europe': [
                'europe', 'européen', 'européenne', 'union', 'communauté', 'euro',
                'traité', 'bruxelles', 'commission', 'parlement', 'intégration',
                'allemagne', 'italie', 'espagne', 'membre'
            ],
            'Social': [
                'social', 'solidarité', 'protection', 'santé', 'retraite', 'famille',
                'éducation', 'école', 'justice', 'hôpital', 'pauvreté', 'logement',
                'aide', 'assistance'
            ],
            'Sécurité': [
                'sécurité', 'défense', 'police', 'armée', 'militaire', 'terrorisme',
                'criminalité', 'violence', 'guerre', 'paix', 'ordre', 'protection'
            ],
            'Environnement': [
                'environnement', 'écologie', 'climat', 'développement', 'pollution',
                'énergie', 'nature', 'planète', 'vert', 'durable', 'carbone'
            ]
        }

    def nettoyer_texte(self, texte):
        """Nettoie et tokenise le texte"""
        texte = re.sub(r'[^a-zàáâäæçèéêëìíîïòóôöùúûüÿ\s]', ' ', texte.lower())
        mots = re.sub(r'\s+', ' ', texte).strip().split()
        return [m for m in mots if m not in self.mots_vides and len(m) >= 3]

    def extraire_metadata(self, ligne):
        """Extrait les métadonnées d'une ligne"""
        return dict(re.findall(r'\*(\w+)_([^\s*]+)', ligne))

    def charger_discours(self, fichier):
        """Charge les discours"""
        with open(fichier, 'r', encoding='utf-8') as f:
            contenu = f.read()
        
        discours_data = []
        for discours in re.split(r'(?=\*\*\*\*\s+)', contenu):
            if not discours.strip():
                continue
                
            lignes = discours.strip().split('\n')
            if not lignes[0].startswith('****'):
                continue
                
            metadata = self.extraire_metadata(lignes[0])
            contenu_discours = '\n'.join(lignes[1:])
            
            # pr identifier le président
            president_nom = None
            if 'qui' in metadata:
                for cle, nom in self.presidents.items():
                    if cle in metadata['qui'].lower():
                        president_nom = nom
                        break
            
            if president_nom:
                discours_data.append({
                    'president': president_nom,
                    'date': metadata.get('quand', ''),
                    'type': metadata.get('type', ''),
                    'annee': metadata.get('annee', ''),
                    'mots_nettoyes': self.nettoyer_texte(contenu_discours),
                    'texte_brut': contenu_discours
                })
        
        return discours_data

    def analyser_themes_temporels(self, discours_data):
        """Analyse l'évolution des thèmes dans le temps"""
        resultats = []
        
        for disc in discours_data:
            try:
                # pr extraire l'année
                if '-' in disc['date']:
                    annee = int(disc['date'].split('-')[0])
                else:
                    annee = int(disc['annee'])
            except:
                continue
            
            texte = ' '.join(disc['mots_nettoyes'])
            total_mots = len(disc['mots_nettoyes'])
            
            if total_mots == 0:
                continue
                
            # calcule les scores thématiques
            for theme, mots_cles in self.themes.items():
                score = sum(texte.count(mot) for mot in mots_cles)
                score_normalise = (score * 1000) / total_mots  # Pour 1000 mots
                
                resultats.append({
                    'annee': annee,
                    'president': disc['president'],
                    'theme': theme,
                    'score': score_normalise
                })
        
        df = pd.DataFrame(resultats)
        self._creer_graphiques_themes(df)
        return df

    def _creer_graphiques_themes(self, df):
        """Crée les visualisations thématiques"""
        # 1. Évolution temporelle par thème
        fig, axes = plt.subplots(len(self.themes), 1, figsize=(14, 3*len(self.themes)))
        
        for i, theme in enumerate(self.themes.keys()):
            theme_df = df[df['theme'] == theme]
            grouped = theme_df.groupby(['annee', 'president'])['score'].mean().reset_index()
            
            for president in df['president'].unique():
                pres_data = grouped[grouped['president'] == president]
                if not pres_data.empty:
                    axes[i].plot(pres_data['annee'], pres_data['score'],
                               marker='o', label=president,
                               color=self.couleurs_presidents.get(president, 'gray'))
            
            axes[i].set_title(f'Évolution du thème "{theme}"')
            axes[i].set_ylabel('Fréquence/1000 mots')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[-1].set_xlabel('Année')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dossier_resultats, 'evolution_themes.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap thèmes vs présidents
        pivot_df = df.pivot_table(values='score', index='president', 
                                 columns='theme', aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Importance des thèmes par président')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dossier_resultats, 'heatmap_themes.png'), dpi=300)
        plt.close()

    def creer_nuages_mots(self, discours_data):
        """Génère des nuages de mots par président"""
        textes_par_president = defaultdict(list)
        for disc in discours_data:
            textes_par_president[disc['president']].extend(disc['mots_nettoyes'])
        
        n_presidents = len(textes_par_president)
        n_cols = 2
        n_rows = (n_presidents + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        colormap = LinearSegmentedColormap.from_list(
            'custom', ['#003366', '#336699', '#66ccff', '#003399'])
        
        for i, (president, mots) in enumerate(sorted(textes_par_president.items())):
            if i < len(axes):
                compteur = Counter(mots)
                texte_wc = ' '.join([f"{mot} " * freq for mot, freq in compteur.items()])
                
                wc = WordCloud(width=800, height=400, background_color='white',
                              max_words=80, colormap=colormap)
                wc.generate(texte_wc)
                
                axes[i].imshow(wc, interpolation='bilinear')
                axes[i].set_title(f"{president}", fontsize=12)
                axes[i].axis('off')
        
        # Masquer les axes non utilisés
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Vocabulaire caractéristique par président', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dossier_resultats, 'nuages_mots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def analyser_corpus_complet(self, fichier):
        """Exécute l'analyse complète du corpus"""
    
        discours_data = self.charger_discours(fichier)
        print(f"{len(discours_data)} discours analysés")
        self.analyser_themes_temporels(discours_data)
        self.creer_nuages_mots(discours_data)
        
        print(f"Analyse terminée")
        
        # Statistiques finales
        presidents_count = len(set(d['president'] for d in discours_data))
        print(f"{presidents_count} présidents analysés")

# Utilisation
if __name__ == "__main__":
    analyseur = AnalyseurAvance()
    analyseur.analyser_corpus_complet("elysee.txt")