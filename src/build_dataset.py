"""
build_dataset.py

Trasforma il checkpoint grezzo (annunci_milano.checkpoint.jsonl, un dettaglio
completo per riga) nella tabella di analisi (CSV + JSON), applicando le
decisioni tenere/scartare. NON tocca la rete: lavora solo su disco, quindi puoi
rigenerare la tabella quante volte vuoi senza ricrawlare.

Principio: il grezzo resta intatto. Qui si SELEZIONA, non si distrugge.
I campi con leakage (prezzo al m², priceRange, mortgage, ...) sono esclusi di
proposito e non devono rientrare.

Uso:
    python src/build_dataset.py annunci_milano.checkpoint.jsonl \
        --out-csv data/processed/annunci_flat.csv \
        --out-parquet data/processed/annunci_flat.parquet
"""

import json
import os
import re
import csv
import time
import argparse


# ---------- getter sicuri: un percorso assente torna None, mai un'eccezione ----------

def dig(obj, *path, default=None):
    for key in path:
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return default
        if obj is None:
            return default
    return obj


def as_bool_int(v):
    """None -> None (dato ignoto); True/'si'/1 -> 1; resto -> 0."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"si", "sì", "true", "yes", "1"} else 0
    return 0


def flag_present(v):
    """Per dotazioni: presente (qualsiasi valore non nullo) -> 1, assente -> 0.
    Qui assente NON e' dato mancante, e' 'non ce l'ha': default 0."""
    return 0 if v is None else 1


def parse_int(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+", str(v).replace(".", "").replace("\u00a0", ""))
    return int(m.group()) if m else None


def parse_money_month(v):
    """'€ 108/mese' -> 108 ; '€ 1.250/mese' -> 1250 ; None -> None."""
    if not v:
        return None
    m = re.search(r"(\d[\d.]*)", str(v).replace(".", ""))
    return int(m.group(1)) if m else None


# ---------- helper per le sezioni a label variabile ----------

def maindata_rows(rec):
    """Appiattisce tutte le righe di mainData in {label_lower: value}."""
    out = {}
    for section in rec.get("mainData") or []:
        for row in section.get("rows") or []:
            label = (row.get("label") or "").strip().lower()
            if label:
                out[label] = row.get("value")
    return out


def find_by_label(rows, *needles):
    """Prima riga la cui label contiene uno dei termini cercati."""
    for label, value in rows.items():
        if any(n in label for n in needles):
            return value
    return None


def energy_features(rec):
    """{label_lower: value} da energyClass.features, gestendo type=seasons."""
    out = {}
    for f in dig(rec, "energyClass", "features", default=[]) or []:
        label = (f.get("label") or "").strip().lower()
        val = f.get("value")
        if val is None and isinstance(f.get("opt"), dict):
            opt = f["opt"]
            val = opt.get("winter") or opt.get("summer")
        if label:
            out[label] = val
    return out


def condo_fees(rec):
    for c in rec.get("costs") or []:
        if "condominio" in (c.get("label") or "").lower():
            return parse_money_month(c.get("value"))
    return None


def feature_flags(rec):
    """Flag da featuresTags[] (lista di stringhe o di dict con 'label')."""
    tags = []
    for t in rec.get("featuresTags") or []:
        tags.append((t if isinstance(t, str) else t.get("label", "")).lower())
    blob = " | ".join(tags)
    # Nota: il posto auto/box NON e' esposto dall'API (ne' in featuresTags ne'
    # in topology), quindi non e' ricavabile e non compare tra le feature.
    return {
        "armored_door": int("blindat" in blob),
        "fiber": int("fibra" in blob),
        "alarm": int("allarme" in blob),
        "double_exposure": int("esposizione doppia" in blob),
    }


def advertiser_type(rec):
    adv = (dig(rec, "analytics", "advertiser") or dig(rec, "author", "type") or "")
    adv = str(adv).lower()
    if "priv" in adv:
        return "private"
    if adv:
        return "agency"
    return None


def is_project(rec):
    return bool(dig(rec, "subProperties", "count") or rec.get("newConstructionDetails"))


def is_nuda_proprieta(rec):
    """La nuda proprieta' e' un mercato a parte: prezzo scontato per l'usufrutto
    in capo ad altri. Compare nel testo libero della descrizione."""
    desc = (dig(rec, "description", "content") or "").lower()
    return int("nuda propriet" in desc)


# ---------- la funzione centrale ----------

def flatten(rec, crawled_at=None):
    md = maindata_rows(rec)
    ef = energy_features(rec)

    condition = find_by_label(md, "stato") or ef.get("stato")
    year_built = parse_int(find_by_label(md, "anno di costruzione", "anno")
                           or ef.get("anno di costruzione"))
    heating = find_by_label(md, "riscaldamento") or dig(rec, "analytics", "heating")

    row = {
        # target
        "price_raw": parse_int(dig(rec, "price", "raw")),

        # struttura
        "surface_m2": parse_int(dig(rec, "topology", "surface", "size")),
        "rooms": parse_int(dig(rec, "topology", "rooms")),
        "bathrooms": parse_int(dig(rec, "topology", "bathrooms")),
        "bedrooms": parse_int(dig(rec, "analytics", "numBedrooms")),
        "floor_raw": dig(rec, "topology", "floor"),
        "lift": as_bool_int(dig(rec, "topology", "lift")),
        "is_luxury": as_bool_int(dig(rec, "topology", "isLuxury")),

        # dotazioni: presente/assente
        "has_cellar": flag_present(dig(rec, "topology", "cellar")),
        "has_balcony": flag_present(dig(rec, "topology", "balcony")),
        "has_terrace": flag_present(dig(rec, "topology", "terrace")),
        "furnished": flag_present(dig(rec, "topology", "furnished")),

        # posizione
        "macrozone_id": dig(rec, "geography", "macrozone", "id"),
        "macrozone_name": dig(rec, "geography", "macrozone", "name"),
        "microzone_id": dig(rec, "geography", "microzone", "id"),
        "microzone_name": dig(rec, "geography", "microzone", "name"),
        "zone_id": dig(rec, "geography", "zone", "id"),
        "zone_name": dig(rec, "geography", "zone", "name"),
        "zipcode": dig(rec, "geography", "zipcode"),
        "lat": dig(rec, "geography", "geolocation", "latitude"),
        "lon": dig(rec, "geography", "geolocation", "longitude"),
        "geo_visibility": dig(rec, "geography", "geolocation", "visibilityType"),

        # solo dal dettaglio
        "condition": condition,
        "typology": dig(rec, "topology", "typology", "name"),
        "year_built": year_built,
        "energy_class": dig(rec, "energyClass", "consumptions", 0, "value"),
        "condo_fees_month": condo_fees(rec),
        "heating_type": heating,

        # meta (fuori dalle feature di prezzo)
        "id": rec.get("id"),
        "advertiser_type": advertiser_type(rec),
        "creation_date": rec.get("creationDate"),
        "last_modified": rec.get("lastModified"),
        "is_project": int(is_project(rec)),
        "is_auction": int(bool(rec.get("auctionsDetails"))),
        "is_nuda_proprieta": is_nuda_proprieta(rec),
        "crawled_at": crawled_at or int(time.time()),
    }
    row.update(feature_flags(rec))
    return row


# colonne che NON devono MAI comparire qui (leakage). Guardia esplicita.
LEAKAGE = {"price_mq", "price_value", "price_range", "pricerange",
           "start_price", "discount", "mortgage", "amount", "rate"}


def _assert_no_leakage(row):
    bad = LEAKAGE & {k.lower() for k in row}
    if bad:
        raise ValueError(f"Colonna con leakage nella tabella feature: {bad}")


def build(checkpoint_path, out_csv, out_parquet=None):
    # Streaming: una riga alla volta, non carichiamo mai l'intero file in RAM.
    # Il grezzo non lo accumuliamo: il monolitico lo produce gia' il crawler.
    rows_all = []
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = flatten(rec)
            _assert_no_leakage(row)
            rows_all.append(row)

    # dedup per id, ultima occorrenza vince (dato piu' fresco); senza id non si
    # puo' deduplicare ne' ricollegare al grezzo: si scarta, contandolo a parte.
    dedup = {}
    n_no_id = 0
    for r in rows_all:
        if r["id"] is None:
            n_no_id += 1
            continue
        dedup[r["id"]] = r
    rows = list(dedup.values())

    # scrive la tabella piatta in CSV e, se pandas c'e', anche in parquet
    wrote_parquet = False
    if rows:
        cols = list(rows[0].keys())
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

        if out_parquet:
            try:
                import pandas as pd
                pd.DataFrame(rows).to_parquet(out_parquet, index=False)
                wrote_parquet = True
            except ImportError:
                print("pandas/pyarrow assenti: salto il parquet, resta il CSV.")

    n_dup = len(rows_all) - n_no_id - len(rows)
    print(f"Annunci letti dal checkpoint: {len(rows_all)}")
    print(f"Righe uniche in tabella: {len(rows)}  (duplicati per id scartati: {n_dup}, senza id: {n_no_id})")
    print(f"Colonne: {len(rows[0]) if rows else 0}")
    print(f"Scritto: {out_csv}" + (f" e {out_parquet}" if wrote_parquet else ""))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Legge il checkpoint .jsonl (fonte di verita') e ne ricava "
                    "la tabella piatta di analisi. NON genera il monolitico: "
                    "quello lo produce gia' il crawler.")
    ap.add_argument("checkpoint", help="percorso di annunci_milano.checkpoint.jsonl")
    ap.add_argument("--out-csv", default="data/processed/annunci_flat.csv")
    ap.add_argument("--out-parquet", default="data/processed/annunci_flat.parquet",
                    help="passa stringa vuota per non generarlo")
    a = ap.parse_args()
    build(a.checkpoint, a.out_csv, a.out_parquet or None)