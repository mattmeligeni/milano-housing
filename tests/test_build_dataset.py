"""Test del costruttore del dataset: funzioni pure, nessun dato reale necessario.

    python -m pytest -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_dataset as bd  # noqa: E402


def test_dig_percorsi_assenti_tornano_default():
    rec = {"a": {"b": [10, 20]}}
    assert bd.dig(rec, "a", "b", 1) == 20
    assert bd.dig(rec, "a", "x") is None
    assert bd.dig(rec, "a", "b", 5, default=0) == 0


@pytest.mark.parametrize("valore, atteso", [
    (None, None), (True, 1), (False, 0), (2, 1), (0, 0), ("Sì", 1), ("no", 0),
])
def test_as_bool_int(valore, atteso):
    assert bd.as_bool_int(valore) == atteso


@pytest.mark.parametrize("valore, atteso", [
    (None, None), (85, 85), ("1.250", 1250), ("piano -1", -1), ("nessuno", None),
])
def test_parse_int(valore, atteso):
    assert bd.parse_int(valore) == atteso


@pytest.mark.parametrize("valore, atteso", [
    ("€ 108/mese", 108), ("€ 1.250/mese", 1250), (None, None), ("", None),
])
def test_parse_money_month(valore, atteso):
    assert bd.parse_money_month(valore) == atteso


def annuncio(**extra):
    rec = {
        "id": 1,
        "price": {"raw": 350000},
        "topology": {"surface": {"size": 80}, "rooms": 3, "lift": True, "balcony": "sì"},
        "mainData": [{"rows": [{"label": "Stato", "value": "Buono / Abitabile"},
                                {"label": "Anno di costruzione", "value": "1965"}]}],
        "costs": [{"label": "Spese condominio", "value": "€ 150/mese"}],
        "featuresTags": ["Porta blindata", {"label": "Fibra ottica"}],
        "description": {"content": "Trilocale luminoso."},
    }
    rec.update(extra)
    return rec


def test_flatten_campi_principali():
    row = bd.flatten(annuncio(), crawled_at=0)
    assert row["price_raw"] == 350000
    assert row["surface_m2"] == 80
    assert row["lift"] == 1
    assert row["has_balcony"] == 1
    assert row["has_terrace"] == 0
    assert row["condition"] == "Buono / Abitabile"
    assert row["year_built"] == 1965
    assert row["condo_fees_month"] == 150
    assert row["armored_door"] == 1 and row["fiber"] == 1 and row["alarm"] == 0
    assert row["is_nuda_proprieta"] == 0


def test_nuda_proprieta_dalla_descrizione():
    row = bd.flatten(annuncio(description={"content": "Vendesi NUDA PROPRIETÀ, usufruttuaria 80 anni"}), crawled_at=0)
    assert row["is_nuda_proprieta"] == 1


def test_nessuna_colonna_con_leakage():
    bd._assert_no_leakage(bd.flatten(annuncio(), crawled_at=0))
    with pytest.raises(ValueError):
        bd._assert_no_leakage({"price_mq": 5000})


def test_build_deduplica_e_scarta_senza_id(tmp_path):
    righe = [annuncio(id=1), annuncio(id=1, price={"raw": 360000}), annuncio(id=None), annuncio(id=2)]
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text("\n".join(json.dumps(r) for r in righe) + "\nnon json\n", encoding="utf-8")
    out = tmp_path / "out" / "flat.csv"
    rows = bd.build(checkpoint, str(out))
    assert {r["id"] for r in rows} == {1, 2}
    assert next(r for r in rows if r["id"] == 1)["price_raw"] == 360000  # vince l'ultima occorrenza
    assert out.exists()
