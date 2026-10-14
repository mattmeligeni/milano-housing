[English](README.md) | **Italiano**

# Milano Housing — stima dei prezzi immobiliari con machine learning

Modello di valutazione automatica (AVM) per il mercato residenziale di Milano.
A partire da ~19.000 annunci raccolti dall'API mobile di Immobiliare.it, il
progetto costruisce un modello che stima il prezzo di un immobile dalle sue
caratteristiche, e un rilevatore di annunci prezzati in modo anomalo.

Il valore del progetto non sta nel singolo numero di accuratezza, ma nel metodo:
ogni feature è stata inclusa solo dopo aver misurato il suo contributo, ogni
riga scartata ha una ragione documentata, e la validazione è costruita per non
mentire a sé stessa.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/modello-CatBoost-yellow)
![R2](https://img.shields.io/badge/R%C2%B2%20(GroupKFold)-0.916-brightgreen)
![Licenza](https://img.shields.io/badge/licenza-PolyForm%20Strict%201.0.0-lightgrey)
[![Demo dal vivo](https://img.shields.io/badge/demo%20dal%20vivo-mattiameligeni.com-1D3F7A)](https://www.mattiameligeni.com/#/progetti/milano-housing)

**[Prova la demo dal vivo →](https://www.mattiameligeni.com/#/progetti/milano-housing)** Scegli un punto sulla mappa di Milano, descrivi la casa e ottieni la
stima. Il modello addestrato gira tutto nel browser e dà la stessa stima di CatBoost, a meno di un miliardesimo.
Mostra anche cosa cambia con una sola differenza (ascensore, ristrutturazione, piano…) e la mappa del prezzo della
stessa casa in tutta la città.

## Risultato

- **R² = 0.916** in validazione onesta (GroupKFold per microzona)
- **errore mediano ~13%** su immobili in quartieri mai visti in addestramento
- modello finale: **CatBoost**, 16 feature, addestrato su **17.086 annunci** puliti

Il modello spiega la formazione dei prezzi milanesi con un nucleo ristretto di
fattori: dimensione, centralità, lusso, stato di conservazione, piano. Oltre a
questi, la maggior parte delle feature geografiche testate (parchi, scuole,
stazioni) si è rivelata ridondante con la distanza dal centro.

<p align="center">
  <img src="docs/img/shap-importanza.png" width="48%" alt="Importanza delle feature (SHAP)">
  <img src="docs/img/shap-dipendenze.png" width="48%" alt="Grafici di dipendenza SHAP">
</p>

## Pipeline

### 1. Raccolta dati

I dati provengono dall'API mobile privata di Immobiliare.it, ricostruita via
reverse engineering del traffico dell'app Android. A differenza dello scraping
HTML, questo approccio restituisce JSON già tipizzato e strutturato, immune agli
errori di posizione che affliggono lo scraping del DOM. Il crawler gestisce la
paginazione a offset, l'anti-bot DataDome, i checkpoint per la ripresa, e una
passata di riconciliazione per chiudere i buchi lasciati dall'indice che si
muove durante la raccolta. *Crawler e documentazione dell'API non sono pubblicati:
servirebbero solo a ripetere la raccolta fuori dai termini di servizio del portale.*

### 2. Costruzione del dataset (`src/build_dataset.py`)

Il grezzo (un dettaglio JSON per riga in `annunci_milano.checkpoint.jsonl`) viene
appiattito in tabella estraendo ~40 colonne **per chiave, mai per posizione**.
Le decisioni di selezione (esclusione dei campi con leakage, dotazioni ridotte a
presente/assente, gestione dei valori mancanti) vivono qui, a valle della
raccolta, così da poter essere riviste senza ricrawlare.

Una guardia esplicita impedisce a colonne derivate dal prezzo (prezzo al m²,
fascia di prezzo, dati del mutuo) di rientrare tra le feature.

<p align="center"><img src="docs/img/distribuzioni.png" width="80%" alt="Distribuzioni di prezzo e superficie"></p>

### 3. Pulizia

Il dataset grezzo passa da ~19.200 a 17.086 righe. Ogni filtro nasce
dall'ispezione dei dati reali, non da regole a priori:

- annunci senza prezzo (nessun target)
- aste e nuove costruzioni (mercati con logiche di prezzo estranee)
- **nuda proprietà** (prezzo scontato per l'usufrutto altrui — individuata nel
  testo delle descrizioni)
- immobili cielo-terra e terreni (superfici oltre 1000 m²)
- prezzi al m² fuori dalla fascia [1500, 20000] €, che includono sia errori di
  battitura sia **annunci geolocalizzati falsamente a Milano** (ville in
  provincia spacciate per milanesi)
- normalizzazione del piano (seminterrato, terra, rialzato → scala ordinata)

### 4. Arricchimento geografico

Dalle coordinate di ogni annuncio vengono calcolate:

- distanza dal Duomo (centro città)
- distanza dalla fermata di metro più vicina (KDTree)
- distanza dalla scuola di base più vicina (KDTree su asili, infanzia, primarie)

Le distanze da centro e metro sono le feature esterne più potenti: a differenza
del nome della microzona, sono **trasferibili** a quartieri mai visti in
addestramento, e hanno alzato l'R² in validazione a gruppi da 0.64 a 0.80.

### 5. Modellazione

Confronto tra modelli in gradazione, ciascuno sullo stesso terreno di
validazione a gruppi:

| Modello             | R² (GroupKFold) |
| ------------------- | --------------- |
| Dummy (media)       | 0.00            |
| Regressione lineare | 0.81            |
| XGBoost             | 0.79            |
| **CatBoost**        | **0.92**        |

CatBoost vince nettamente per la gestione nativa della microzona, categorica ad
alta cardinalità (145 valori), tramite ordered target encoding che evita il
leakage. Gli iperparametri sono stati cercati con validazione a gruppi.

**La validazione è per microzona, non casuale.** Questo evita che immobili
gemelli (lo stesso appartamento ripubblicato, o case dello stesso palazzo)
finiscano a cavallo tra addestramento e test gonfiando il punteggio. La
differenza tra validazione casuale (0.92) e a gruppi (0.79 senza distanze) è la
misura di quanto quel trucco falserebbe il risultato.

### 6. Interpretazione (SHAP)

Il modello è interrogabile: superficie e distanza dal centro sono i due motori
principali, seguiti da lusso, stato, piano. La posizione fine (microzona) pesa
poco proprio perché le distanze continue ne catturano già l'informazione.

### 7. Applicazione: rilevatore di occasioni

Il modello, girato verso i propri residui, individua immobili prezzati sotto i
loro simili di zona. Non è un distributore automatico di affari: è un filtro che
riduce 17.000 immobili a una manciata di candidati da **verificare a mano**.

L'analisi dei residui ha rivelato che le anomalie più forti quasi sempre non
sono affari, ma difetti invisibili al modello (vicinanza a stazioni rumorose,
esposizione, piano) o errori nei dati. Questo ha portato a scoprire e catalogare
tutte le categorie che fanno _sembrare_ un immobile un'occasione senza esserlo,
usate poi per raffinare il filtro. Conclusione onesta: il mercato è
ragionevolmente efficiente, e i veri mispricing sono rari.

## Struttura del repository

```
milano-housing/
├── src/build_dataset.py             # grezzo JSONL -> tabella piatta
├── notebooks/analisi_modello.ipynb  # EDA, pulizia, modelli, SHAP, rilevatore
├── src/predict_example.py           # carica il modello e stima una casa d'esempio
├── src/esporta_demo_web.py          # esporta il modello per la demo nel browser, verificata contro CatBoost
├── tests/test_build_dataset.py      # test del costruttore del dataset (senza dati)
├── docs/img/                        # grafici usati in questo README
└── requirements.txt
```

**Non inclusi:** gli annunci raccolti, i dataset derivati e il modello addestrato. Derivano dalla banca dati del
portale e non si possono ridistribuire. Il notebook conserva risultati aggregati e grafici; le uscite che elencano
singoli annunci sono state svuotate. Il modello addestrato si può provare nella [demo dal vivo](https://www.mattiameligeni.com/#/progetti/milano-housing),
esportato da `src/esporta_demo_web.py`.

## Uso del modello

```python
import numpy as np, pandas as pd
from catboost import CatBoostRegressor

modello = CatBoostRegressor()
modello.load_model("data/processed/modello_prezzi_milano.cbm")  # modello addestrato in locale, non incluso

casa = pd.DataFrame([{
    "surface_m2": 80, "rooms": 3, "bathrooms": 2, "bedrooms": 2, "lift": 1,
    "is_luxury": 0, "has_cellar": 1, "has_balcony": 1, "has_terrace": 0,
    "dist_centro_km": 2.5, "dist_metro_km": 0.3, "year_built": 1970,
    "dist_scuola_km": 0.2, "floor_num": 3,
    "microzone_name": "Isola", "condition": "Ottimo / Ristrutturato",
}])

# il modello predice log1p(prezzo): si riporta in euro con expm1
prezzo = np.expm1(modello.predict(casa))[0]
print(f"{prezzo:,.0f} €")
```

## Limiti

Onestà sui confini del lavoro, che ne definiscono la maturità:

- **Fonte dei dati**: raccolti da un'API non pubblica, fuori dai termini di
  servizio del portale. Uso personale e didattico, volumi contenuti. Un impiego
  commerciale richiederebbe una fonte lecita.
- **Prezzo richiesto ≠ prezzo di vendita**: il modello stima il prezzo di
  listino, non quello di rogito. Un AVM professionale si allena sui prezzi di
  vendita effettivi.
- **Fotografia statica**: il modello riflette il mercato al momento della
  raccolta e va riaddestrato periodicamente.
- **Feature non osservate**: esposizione, vista, piano nobile, qualità del
  palazzo non sono nei dati; il modello sbaglia sistematicamente dove queste
  contano (super-lusso, immobili atipici).

## Stack

Python · pandas · CatBoost · XGBoost · scikit-learn · SHAP · geopy · scipy

## Licenza

Codice consultabile con la [PolyForm Strict License 1.0.0](LICENSE): si può leggere ed eseguire per scopi non
commerciali; non si può modificare, ridistribuire né usare commercialmente senza permesso scritto.

---

<sub>© 2026 [Mattia Meligeni](https://mattiameligeni.com). Progetto non affiliato a Immobiliare.it.</sub>
