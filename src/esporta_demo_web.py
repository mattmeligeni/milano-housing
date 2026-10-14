"""
esporta_demo_web.py

Esporta il modello addestrato in un pacchetto JSON che un browser puo' valutare da solo, senza Python ne' server:
la demo interattiva sul sito (mattiameligeni.com, pagina di Milano Housing).

Il modello e' CatBoost con 854 alberi "oblivious" di profondita' 4. Gli split sulle feature numeriche sono soglie
semplici; quelli sulla microzona e sullo stato sono CTR (target encoding ordinato, anche combinato con soglie
numeriche), che CatBoost calcola con hash interni. Invece di reimplementare quegli hash, lo script chiede al modello
stesso (calc_leaf_indexes) l'esito di ogni split CTR per ogni combinazione di microzona, stato e soglie coinvolte, e lo
salva in tabelle di bit. Il browser fa poi esattamente la stessa somma sulle foglie: la stima e' quella del modello,
non un'approssimazione. Alla fine lo script lo verifica su tutti gli annunci puliti.

Escono due file:
- modello.json: alberi, foglie, tabelle dei CTR, elenco di microzone e stati;
- mappa.json:   baricentro di ogni microzona (media delle coordinate degli annunci, solo aggregati), fermate della
                metro e scuole di base (open data del Comune di Milano), per calcolare le distanze come nel notebook.

Uso:
    python src/esporta_demo_web.py ../../Web/mattiameligeni.com/housing
"""

import base64
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

RADICE = Path(__file__).resolve().parents[1]
MODELLO = RADICE / "data/processed/modello_prezzi_milano.cbm"
DATI = RADICE / "data/processed/annunci_clean_full.parquet"
GREZZI = RADICE / "data/raw"

STATI = ["Ottimo / Ristrutturato", "Buono / Abitabile", "Nuovo / In costruzione", "Da ristrutturare", "Sconosciuto"]

# Casa "neutra" usata per riempire le feature che uno split CTR non guarda.
BASE = {
    "surface_m2": 80, "rooms": 3, "bathrooms": 1, "bedrooms": 2, "lift": 1, "is_luxury": 0, "has_cellar": 0,
    "has_balcony": 1, "has_terrace": 0, "dist_centro_km": 3.0, "dist_metro_km": 0.5, "year_built": 1960,
    "dist_scuola_km": 0.3, "floor_num": 2,
}


def carica():
    modello = CatBoostRegressor()
    modello.load_model(str(MODELLO))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        modello.save_model(str(p), format="json")
        j = json.loads(p.read_text())
    return modello, j


def split_globali(j):
    """Ordine globale degli split: prima le soglie numeriche, poi i bordi dei CTR (verificato sugli alberi)."""
    g = []
    for f in j["features_info"]["float_features"]:
        g += [("F", f["feature_index"], b) for b in f["borders"]]
    for i, c in enumerate(j["features_info"]["ctrs"]):
        g += [("C", i, b) for b in c["borders"]]
    return g


def valore_per(vincoli):
    """Un valore che soddisfa tutte le condizioni (soglia, sopra?) su una stessa feature; None se impossibile."""
    lo, hi = -1e9, 1e9
    for b, sopra in vincoli:
        if sopra:
            lo = max(lo, b)
        else:
            hi = min(hi, b)
    if lo >= hi:
        return None
    # Le soglie a -3.4e38 sono quelle dei valori mancanti: "+1" li' sparirebbe nell'arrotondamento.
    if lo == -1e9:
        return hi - 1.0
    if hi == 1e9:
        return lo + 1.0 if lo > -1e30 else 0.0
    return (lo + hi) / 2 if lo > -1e30 else hi - 1.0


def main(uscita: Path):
    modello, j = carica()
    nomi = modello.feature_names_
    nf = 14
    df = pd.read_parquet(DATI)
    microzone = sorted(df["microzone_name"].dropna().unique().tolist()) + ["Sconosciuta"]
    alberi = j["oblivious_trees"]
    glob = split_globali(j)
    ctrs = j["features_info"]["ctrs"]

    # Per ogni CTR: da quali categorie e soglie dipende.
    info_ctr = []
    for c in ctrs:
        cats = sorted({e["cat_feature_index"] for e in c["elements"] if e["combination_element"] == "cat_feature_value"})
        fl = [(e["float_feature_index"], e["border"]) for e in c["elements"] if e["combination_element"] == "float_feature"]
        info_ctr.append({"m": 0 in cats, "c": 1 in cats, "f": fl})

    tabelle = {}
    controlli = 0
    for ci, ic in enumerate(info_ctr):
        usi = [(ti, k, s) for ti, t in enumerate(alberi) for k, s in enumerate(t["splits"])
               if s["split_type"] == "OnlineCtr" and glob[s["split_index"]][1] == ci]
        if not usi:
            continue
        nm = len(microzone) if ic["m"] else 1
        nc = len(STATI) if ic["c"] else 1
        kf = len(ic["f"])
        righe, valide = [], []
        for m in range(nm):
            for c in range(nc):
                for bits in range(2 ** kf):
                    r = dict(BASE)
                    vincoli = {}
                    for i, (fi, b) in enumerate(ic["f"]):
                        vincoli.setdefault(fi, []).append((b, bool(bits >> i & 1)))
                    ok = True
                    for fi, v in vincoli.items():
                        x = valore_per(v)
                        if x is None:
                            ok = False
                            break
                        r[nomi[fi]] = x
                    r["microzone_name"] = microzone[m] if ic["m"] else microzone[0]
                    r["condition"] = STATI[c] if ic["c"] else STATI[0]
                    righe.append(r)
                    valide.append(ok)
        X = pd.DataFrame(righe)[nomi]
        foglie = modello.calc_leaf_indexes(Pool(X, cat_features=[14, 15]))
        for ti, k, s in usi:
            bit = (foglie[:, ti] >> k) & 1
            bit[~np.array(valide)] = 0
            tabelle.setdefault(s["split_index"], np.packbits(bit.astype(np.uint8), bitorder="little"))
        # Controllo dell'ordine dei bit sugli split numerici degli stessi alberi.
        for ti in {u[0] for u in usi}:
            for k, s in enumerate(alberi[ti]["splits"]):
                if s["split_type"] == "FloatFeature":
                    atteso = (X.iloc[:, s["float_feature_index"]].to_numpy() > s["border"]).astype(int)
                    assert np.array_equal((foglie[:, ti] >> k) & 1, atteso), "ordine dei bit delle foglie inatteso"
                    controlli += 1

    # Alberi compatti: [indice feature, soglia] per gli split numerici, [-1, indice split] per i CTR.
    out_alberi = []
    for t in alberi:
        sp = [[s["float_feature_index"], s["border"]] if s["split_type"] == "FloatFeature" else [-1, s["split_index"]]
              for s in t["splits"]]
        out_alberi.append([sp, [float(np.float32(v)) for v in t["leaf_values"]]])
    scala, bias = j["scale_and_bias"]
    pacchetto = {
        "feature": nomi[:nf],
        "microzone": microzone,
        "stati": STATI,
        "ctr": [{"m": int(i["m"]), "c": int(i["c"]), "f": i["f"]} for i in info_ctr],
        "split_ctr": {str(si): glob[si][1] for si in tabelle},
        "tabelle": {str(si): base64.b64encode(v.tobytes()).decode() for si, v in tabelle.items()},
        "alberi": out_alberi,
        "scala": scala,
        "bias": bias[0] if isinstance(bias, list) else bias,
        "metriche": {"r2": 0.916, "errore_mediano": 0.13, "annunci": len(df), "alberi": len(alberi)},
    }

    # Verifica: la stessa valutazione del browser, rifatta qui, contro CatBoost su tutti gli annunci.
    tab = {int(k): np.unpackbits(np.frombuffer(base64.b64decode(v), np.uint8), bitorder="little")
           for k, v in pacchetto["tabelle"].items()}

    def stima(r):
        mi = microzone.index(r["microzone_name"]) if r["microzone_name"] in microzone else len(microzone) - 1
        st = STATI.index(r["condition"]) if r["condition"] in STATI else len(STATI) - 1
        tot = 0.0
        for sp, foglie in out_alberi:
            idx = 0
            for k, (a, b) in enumerate(sp):
                if a >= 0:
                    x = r[nomi[a]]
                    bit = 0 if pd.isna(x) else int(x > b)
                else:
                    ic = info_ctr[glob[b][1]]
                    kf = len(ic["f"])
                    fb = 0
                    for i, (fi, bb) in enumerate(ic["f"]):
                        x = r[nomi[fi]]
                        fb |= (0 if pd.isna(x) else int(x > bb)) << i
                    nc = len(STATI) if ic["c"] else 1
                    pos = (((mi if ic["m"] else 0) * nc + (st if ic["c"] else 0)) << kf) | fb
                    bit = int(tab[b][pos])
                idx |= bit << k
            tot += foglie[idx]
        return scala * tot + pacchetto["bias"]

    X = df[nomi].copy()
    X["microzone_name"] = X["microzone_name"].fillna("Sconosciuta").astype(str)
    X["condition"] = X["condition"].fillna("Sconosciuto").astype(str)
    # Solo annunci completi: la demo passa sempre tutti i valori (CatBoost tratta i mancanti nei CTR combinati in un
    # modo che qui non serve riprodurre).
    completi = X.dropna()
    campione = completi.sample(min(5000, len(completi)), random_state=1)
    # ...piu' case inventate su tutto l'intervallo che la demo permette, con ogni microzona e ogni stato.
    rng = np.random.default_rng(7)
    n = 3000
    inventate = pd.DataFrame({
        "surface_m2": rng.integers(20, 400, n), "rooms": rng.integers(1, 6, n), "bathrooms": rng.integers(1, 4, n),
        "bedrooms": rng.integers(0, 6, n), "lift": rng.integers(0, 2, n), "is_luxury": rng.integers(0, 2, n),
        "has_cellar": rng.integers(0, 2, n), "has_balcony": rng.integers(0, 2, n), "has_terrace": rng.integers(0, 2, n),
        "dist_centro_km": rng.uniform(0, 11, n), "dist_metro_km": rng.uniform(0, 3, n),
        "year_built": rng.integers(1850, 2027, n), "dist_scuola_km": rng.uniform(0, 1.5, n),
        "floor_num": rng.choice([-1, 0, 0.5] + list(range(1, 16)), n),
        "microzone_name": rng.choice(microzone, n), "condition": rng.choice(STATI, n),
    })[nomi]
    campione = pd.concat([campione, inventate], ignore_index=True)
    rif = modello.predict(campione)
    nostra = np.array([stima(r) for _, r in campione.iterrows()])
    diff = np.abs(np.expm1(nostra) / np.expm1(rif) - 1)
    print(f"verifica su {len(campione)} annunci: scarto massimo {diff.max():.2e}, medio {diff.mean():.2e} "
          f"(controlli sull'ordine dei bit: {controlli})")
    assert diff.max() < 1e-4, "la valutazione esportata non coincide con CatBoost"

    # Mappa: baricentri delle microzone (solo aggregati), metro e scuole (open data).
    g = df.dropna(subset=["microzone_name", "lat", "lon"]).groupby("microzone_name")
    zone = [{"n": z, "lat": round(float(d["lat"].mean()), 5), "lon": round(float(d["lon"].mean()), 5),
             "macro": str(d["macrozone_name"].mode().iloc[0]) if d["macrozone_name"].notna().any() else ""}
            for z, d in g if len(d) >= 3]
    metro = json.loads((GREZZI / "fermateMetro.geojson").read_text())
    fermate = [{"n": f["properties"].get("nome", ""), "l": f["properties"].get("linee", ""),
                "lat": round(f["geometry"]["coordinates"][1], 5), "lon": round(f["geometry"]["coordinates"][0], 5)}
               for f in metro["features"]]
    scuole = set()
    for nome, la, lo in [("asiliNido", "LAT_WGS84", "LONG_WGS84"), ("scuoleInfanzia", "LAT_Y_4326", "LONG_X_4326"),
                         ("scuolePrimarie", "LAT_Y_4326", "LONG_X_4326")]:
        for f in json.loads((GREZZI / f"{nome}.geojson").read_text())["features"]:
            p = f["properties"]
            if p.get(la) is not None and p.get(lo) is not None:
                scuole.add((round(float(p[la]), 5), round(float(p[lo]), 5)))
    mappa = {"zone": zone, "metro": fermate, "scuole": sorted(scuole), "duomo": [45.4637, 9.18813]}

    uscita.mkdir(parents=True, exist_ok=True)
    (uscita / "modello.json").write_text(json.dumps(pacchetto, separators=(",", ":")))
    (uscita / "mappa.json").write_text(json.dumps(mappa, separators=(",", ":"), ensure_ascii=False))
    for f in ("modello.json", "mappa.json"):
        print(f, f"{(uscita / f).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RADICE / "privato/demo-web")
