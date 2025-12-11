import streamlit as st
import re
import math
from collections import Counter

class NaiveBayesLanguageIdentifier:
    def __init__(self, ngram_range=(1, 3)):
        self.ngram_range = ngram_range
        self.language_models = {}  # {lang: {ngram: log_probability}}
        self.language_priors = {}  # {lang: log_prior}
        self.vocabulary = set()

    def _normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_ngrams(self, text):
        text = self._normalize_text(text)
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
        return ngrams

    def train(self, language_data):
        import math
        
        total_docs = len(language_data)
        all_ngram_counts = {}
        
        for lang, corpus in language_data.items():
            ngrams = self._extract_ngrams(corpus)
            ngram_counts = Counter(ngrams)
            all_ngram_counts[lang] = ngram_counts
            self.vocabulary.update(ngram_counts.keys())
        
        vocab_size = len(self.vocabulary)
        
        for lang, ngram_counts in all_ngram_counts.items():
            total_ngrams = sum(ngram_counts.values())
            self.language_models[lang] = {}
            
            self.language_priors[lang] = math.log(1 / total_docs)
            
            for ngram in self.vocabulary:
                count = ngram_counts.get(ngram, 0)
                prob = (count + 1) / (total_ngrams + vocab_size)
                self.language_models[lang][ngram] = math.log(prob)
            
            self.language_models[lang]['__unknown__'] = math.log(1 / (total_ngrams + vocab_size))

    def predict(self, document_text):
        import math
        
        if not self.language_models:
            return "Model not trained.", {}
        
        ngrams = self._extract_ngrams(document_text)
        
        if not ngrams:
            return "Error: No n-grams extracted.", {}
        
        scores = {}
        
        for lang in self.language_models:
            log_prob = self.language_priors[lang]
            
            for ngram in ngrams:
                if ngram in self.language_models[lang]:
                    log_prob += self.language_models[lang][ngram]
                else:
                    log_prob += self.language_models[lang]['__unknown__']
            
            scores[lang] = log_prob
        
        predicted_lang = max(scores, key=scores.get)
        
        max_score = max(scores.values())
        exp_scores = {lang: math.exp(score - max_score) for lang, score in scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {lang: (exp_score / total_exp) * 100 for lang, exp_score in exp_scores.items()}
        
        return predicted_lang, probabilities


CORPUS_ENGLISH = """
The quick brown fox jumps over the lazy dog. To be or not to be, that is the question. 
The Recursive Neural Tensor Network outperforms all previous methods on several metrics. 
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Natural language processing deals with the interaction between computers and humans using natural language.
The weather today is quite pleasant with sunny skies and mild temperatures throughout the afternoon.
Scientists have discovered a new species of marine life in the depths of the Pacific Ocean.
The government announced new policies to address climate change and promote renewable energy sources.
Education is the most powerful weapon which you can use to change the world, as Nelson Mandela once said.
The stock market experienced significant volatility following the release of quarterly earnings reports.
Technology continues to reshape how we communicate, work, and interact with one another daily.
"""

CORPUS_GERMAN = """
Der schnelle braune Fuchs springt über den faulen Hund. Sein oder nicht sein, das ist hier die Frage.
Das Recursive Neural Tensor Netzwerk übertrifft alle bisherigen Methoden in mehreren Metriken.
Maschinelles Lernen ist ein Teilgebiet der künstlichen Intelligenz, das Systemen ermöglicht, aus Daten zu lernen.
Die Verarbeitung natürlicher Sprache befasst sich mit der Interaktion zwischen Computern und Menschen.
Das Wetter heute ist ziemlich angenehm mit sonnigem Himmel und milden Temperaturen am Nachmittag.
Wissenschaftler haben eine neue Spezies von Meereslebewesen in den Tiefen des Pazifischen Ozeans entdeckt.
Die Regierung hat neue Maßnahmen angekündigt, um den Klimawandel zu bekämpfen und erneuerbare Energien zu fördern.
Bildung ist die mächtigste Waffe, die man benutzen kann, um die Welt zu verändern, wie Nelson Mandela sagte.
Der Aktienmarkt erlebte erhebliche Volatilität nach der Veröffentlichung der Quartalsergebnisse.
Technologie verändert weiterhin, wie wir kommunizieren, arbeiten und miteinander interagieren.
"""

CORPUS_FRENCH = """
Le rapide renard brun saute par-dessus le chien paresseux. Être ou ne pas être, telle est la question.
Le Réseau Tensoriel Neuronal Récurrent surpasse toutes les méthodes précédentes sur plusieurs mesures.
L'apprentissage automatique est un sous-domaine de l'intelligence artificielle qui permet aux systèmes d'apprendre.
Le traitement du langage naturel traite de l'interaction entre les ordinateurs et les humains.
Le temps aujourd'hui est assez agréable avec un ciel ensoleillé et des températures douces tout l'après-midi.
Les scientifiques ont découvert une nouvelle espèce de vie marine dans les profondeurs de l'océan Pacifique.
Le gouvernement a annoncé de nouvelles politiques pour lutter contre le changement climatique.
L'éducation est l'arme la plus puissante que l'on puisse utiliser pour changer le monde, comme l'a dit Mandela.
Le marché boursier a connu une volatilité significative après la publication des résultats trimestriels.
La technologie continue de transformer notre façon de communiquer, de travailler et d'interagir.
"""

CORPUS_ITALIAN = """
La veloce volpe marrone salta sopra il cane pigro. Essere o non essere, questo è il problema.
La Rete Neurale Tensoriale Ricorsiva supera tutti i metodi precedenti su diverse metriche.
L'apprendimento automatico è un sottoinsieme dell'intelligenza artificiale che consente ai sistemi di imparare.
L'elaborazione del linguaggio naturale si occupa dell'interazione tra computer e esseri umani.
Il tempo oggi è abbastanza piacevole con cielo soleggiato e temperature miti per tutto il pomeriggio.
Gli scienziati hanno scoperto una nuova specie di vita marina nelle profondità dell'Oceano Pacifico.
Il governo ha annunciato nuove politiche per affrontare il cambiamento climatico e promuovere le energie rinnovabili.
L'istruzione è l'arma più potente che si può usare per cambiare il mondo, come disse Nelson Mandela.
Il mercato azionario ha registrato una significativa volatilità dopo la pubblicazione dei risultati trimestrali.
La tecnologia continua a trasformare il modo in cui comunichiamo, lavoriamo e interagiamo tra di noi.
"""

CORPUS_SPANISH = """
El rápido zorro marrón salta sobre el perro perezoso. Ser o no ser, esa es la cuestión.
La Red Neuronal Tensorial Recursiva supera todos los métodos anteriores en varias métricas.
El aprendizaje automático es un subconjunto de la inteligencia artificial que permite a los sistemas aprender.
El procesamiento del lenguaje natural se ocupa de la interacción entre computadoras y humanos.
El clima hoy es bastante agradable con cielos soleados y temperaturas suaves durante toda la tarde.
Los científicos han descubierto una nueva especie de vida marina en las profundidades del Océano Pacífico.
El gobierno anunció nuevas políticas para abordar el cambio climático y promover las energías renovables.
La educación es el arma más poderosa que puedes usar para cambiar el mundo, como dijo Nelson Mandela.
El mercado de valores experimentó una volatilidad significativa tras la publicación de los resultados trimestrales.
La tecnología sigue transformando la forma en que nos comunicamos, trabajamos e interactuamos entre nosotros.
"""

CORPUS_PORTUGUESE = """
A rápida raposa marrom salta sobre o cão preguiçoso. Ser ou não ser, eis a questão.
A Rede Neural Tensorial Recursiva supera todos os métodos anteriores em várias métricas.
O aprendizado de máquina é um subconjunto da inteligência artificial que permite aos sistemas aprender.
O processamento de linguagem natural trata da interação entre computadores e humanos usando linguagem natural.
O tempo hoje está bastante agradável com céu ensolarado e temperaturas amenas durante toda a tarde.
Os cientistas descobriram uma nova espécie de vida marinha nas profundezas do Oceano Pacífico.
O governo anunciou novas políticas para enfrentar as mudanças climáticas e promover fontes de energia renovável.
A educação é a arma mais poderosa que você pode usar para mudar o mundo, como disse Nelson Mandela.
O mercado de ações experimentou volatilidade significativa após a divulgação dos resultados trimestrais.
A tecnologia continua a transformar a forma como nos comunicamos, trabalhamos e interagimos uns com os outros.
"""

CORPUS_DUTCH = """
De snelle bruine vos springt over de luie hond. Zijn of niet zijn, dat is de vraag.
Het Recursieve Neurale Tensor Netwerk overtreft alle vorige methoden op verschillende metrieken.
Machine learning is een deelgebied van kunstmatige intelligentie waarmee systemen kunnen leren van gegevens.
Natuurlijke taalverwerking houdt zich bezig met de interactie tussen computers en mensen met natuurlijke taal.
Het weer vandaag is vrij aangenaam met zonnige luchten en milde temperaturen gedurende de hele middag.
Wetenschappers hebben een nieuwe soort zeeleven ontdekt in de diepten van de Stille Oceaan.
De regering heeft nieuw beleid aangekondigd om klimaatverandering aan te pakken en hernieuwbare energie te bevorderen.
Onderwijs is het krachtigste wapen dat je kunt gebruiken om de wereld te veranderen, zoals Nelson Mandela zei.
De aandelenmarkt kende aanzienlijke volatiliteit na de publicatie van de kwartaalresultaten.
Technologie blijft de manier veranderen waarop we communiceren, werken en met elkaar omgaan.
"""

CORPUS_SWEDISH = """
Den snabba bruna räven hoppar över den lata hunden. Att vara eller inte vara, det är frågan.
Det Rekursiva Neurala Tensor Nätverket överträffar alla tidigare metoder på flera mätvärden.
Maskininlärning är en del av artificiell intelligens som gör det möjligt för system att lära sig från data.
Naturlig språkbehandling handlar om interaktionen mellan datorer och människor med naturligt språk.
Vädret idag är ganska behagligt med soligt väder och milda temperaturer under hela eftermiddagen.
Forskare har upptäckt en ny art av marint liv i djupet av Stilla havet.
Regeringen har tillkännagivit nya riktlinjer för att hantera klimatförändringarna och främja förnybar energi.
Utbildning är det kraftfullaste vapnet du kan använda för att förändra världen, som Nelson Mandela sa.
Aktiemarknaden upplevde betydande volatilitet efter publiceringen av kvartalsresultaten.
Tekniken fortsätter att omforma hur vi kommunicerar, arbetar och interagerar med varandra dagligen.
"""

CORPUS_POLISH = """
Szybki brązowy lis skacze nad leniwym psem. Być albo nie być, oto jest pytanie.
Rekurencyjna Sieć Neuronowa Tensorowa przewyższa wszystkie poprzednie metody pod względem wielu metryk.
Uczenie maszynowe jest podzbiorem sztucznej inteligencji, który umożliwia systemom uczenie się z danych.
Przetwarzanie języka naturalnego zajmuje się interakcją między komputerami a ludźmi przy użyciu języka naturalnego.
Pogoda dzisiaj jest dość przyjemna ze słonecznym niebem i łagodnymi temperaturami przez całe popołudnie.
Naukowcy odkryli nowy gatunek życia morskiego w głębinach Oceanu Spokojnego.
Rząd ogłosił nowe polityki mające na celu przeciwdziałanie zmianom klimatycznym i promowanie odnawialnych źródeł energii.
Edukacja jest najpotężniejszą bronią, której możesz użyć, aby zmienić świat, jak powiedział Nelson Mandela.
Rynek akcji doświadczył znacznej zmienności po publikacji wyników kwartalnych.
Technologia nadal zmienia sposób, w jaki się komunikujemy, pracujemy i wchodzimy w interakcje ze sobą.
"""

CORPUS_ROMANIAN = """
Vulpea maronie rapidă sare peste câinele leneș. A fi sau a nu fi, aceasta este întrebarea.
Rețeaua Neuronală Tensorială Recursivă depășește toate metodele anterioare pe mai multe metrici.
Învățarea automată este un subset al inteligenței artificiale care permite sistemelor să învețe din date.
Procesarea limbajului natural se ocupă de interacțiunea dintre computere și oameni folosind limbajul natural.
Vremea de astăzi este destul de plăcută cu cer însorit și temperaturi blânde pe tot parcursul după-amiezii.
Oamenii de știință au descoperit o nouă specie de viață marină în adâncurile Oceanului Pacific.
Guvernul a anunțat noi politici pentru a aborda schimbările climatice și a promova sursele de energie regenerabilă.
Educația este cea mai puternică armă pe care o poți folosi pentru a schimba lumea, așa cum a spus Nelson Mandela.
Piața bursieră a cunoscut o volatilitate semnificativă după publicarea rezultatelor trimestriale.
Tehnologia continuă să transforme modul în care comunicăm, lucrăm și interacționăm unii cu alții zilnic.
"""

CORPUS_DANISH = """
Den hurtige brune ræv springer over den dovne hund. At være eller ikke være, det er spørgsmålet.
Det Rekursive Neurale Tensor Netværk overgår alle tidligere metoder på flere målepunkter.
Maskinlæring er en del af kunstig intelligens, der gør det muligt for systemer at lære af data.
Naturlig sprogbehandling beskæftiger sig med interaktionen mellem computere og mennesker ved brug af naturligt sprog.
Vejret i dag er ret behageligt med solskin og milde temperaturer hele eftermiddagen.
Forskere har opdaget en ny art af havliv i dybderne af Stillehavet.
Regeringen har annonceret nye politikker for at tackle klimaforandringer og fremme vedvarende energikilder.
Uddannelse er det mest kraftfulde våben, du kan bruge til at ændre verden, som Nelson Mandela sagde.
Aktiemarkedet oplevede betydelig volatilitet efter offentliggørelsen af kvartalsresultaterne.
Teknologien fortsætter med at omforme, hvordan vi kommunikerer, arbejder og interagerer med hinanden dagligt.
"""

CORPUS_NORWEGIAN = """
Den raske brune reven hopper over den late hunden. Å være eller ikke være, det er spørsmålet.
Det Rekursive Nevrale Tensor Nettverket overgår alle tidligere metoder på flere måleparametre.
Maskinlæring er en del av kunstig intelligens som gjør det mulig for systemer å lære fra data.
Naturlig språkbehandling handler om interaksjonen mellom datamaskiner og mennesker ved bruk av naturlig språk.
Været i dag er ganske behagelig med solskinn og milde temperaturer gjennom hele ettermiddagen.
Forskere har oppdaget en ny art av marint liv i dypet av Stillehavet.
Regjeringen har kunngjort nye retningslinjer for å takle klimaendringer og fremme fornybare energikilder.
Utdanning er det kraftigste våpenet du kan bruke for å forandre verden, som Nelson Mandela sa.
Aksjemarkedet opplevde betydelig volatilitet etter publiseringen av kvartalsresultatene.
Teknologien fortsetter å omforme hvordan vi kommuniserer, arbeider og samhandler med hverandre daglig.
"""

LANGUAGE_DATA = {
    "English": CORPUS_ENGLISH,
    "German": CORPUS_GERMAN,
    "French": CORPUS_FRENCH,
    "Italian": CORPUS_ITALIAN,
    "Spanish": CORPUS_SPANISH,
    "Portuguese": CORPUS_PORTUGUESE,
    "Dutch": CORPUS_DUTCH,
    "Swedish": CORPUS_SWEDISH,
    "Polish": CORPUS_POLISH,
    "Romanian": CORPUS_ROMANIAN,
    "Danish": CORPUS_DANISH,
    "Norwegian": CORPUS_NORWEGIAN,
}

@st.cache_data
def load_and_train_model():
    identifier = NaiveBayesLanguageIdentifier(ngram_range=(1, 3))
    identifier.train(LANGUAGE_DATA)
    return identifier

st.title("Naive Bayes Language Identifier")
st.markdown("---")

st.markdown("""
**Naive Bayes classification** for language identification is based on the approach by 
**Dunning (1994)** - *"Statistical Identification of Language"*. 
Unlike distance-based methods, Naive Bayes is based on **confidence scores** for each language.

Bayes' theorem:
   
   **P(Language | Text) ∝ P(Text | Language) × P(Language)**

With **Laplace smoothing**, the model can handle n-grams it hasn't seen during training without causing errors or zero probabilities.

Using character sequences (1-3 characters) captures language-specific patterns like 'th' in English, 'sch' in German, 'zione' in Italian, without needing word boundaries.

""")

st.markdown("---")

model = load_and_train_model()

input_text = st.text_area("Write the text you want to identify:")

if st.button("Identify Language"):
    if input_text:
        predicted_lang, scores = model.predict(input_text)
        
        st.markdown("---")
        
        st.metric(label="Predicted Language", value=predicted_lang)

        st.subheader("Probability Scores")
        st.info(
            "The model chooses the language with the **highest probability**, "
            "calculated using Bayes' theorem with character n-gram likelihoods."
        )
        
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        scores_table = {
            "Language": list(sorted_scores.keys()),
            "Probability (%)": [f"{v:.2f}%" for v in sorted_scores.values()]
        }
        
        st.dataframe(scores_table, hide_index=True)
    
    else:
        st.warning("Please enter some text to identify the language.")
