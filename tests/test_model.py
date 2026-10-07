import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# ricarica il modello salvato
modello = CatBoostRegressor()
modello.load_model("data/processed/modello_prezzi_milano.cbm")

# descrivi una casa: tutte le feature che il modello si aspetta, nell'ordine giusto
casa = pd.DataFrame([{
    "surface_m2": 45,
    "rooms": 1,
    "bathrooms": 2,
    "bedrooms": 2,
    "lift": 1,
    "is_luxury": 0,
    "has_cellar": 1,
    "has_balcony": 1,
    "has_terrace": 0,
    "dist_centro_km": 5.5,
    "dist_metro_km": 1.3,
    "year_built": 1970,
    "dist_scuola_km": 0.2,
    "floor_num": -1,
    "microzone_name": "Isola",
    "condition": "Ottimo / Ristrutturato",
}])

# il modello predice il LOGARITMO del prezzo, va riportato in euro con expm1
prezzo_log = modello.predict(casa)
prezzo_euro = np.expm1(prezzo_log)
print(f"prezzo stimato: {prezzo_euro[0]:,.0f} euro")