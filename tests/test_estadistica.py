"""Pruebas de la estadistica del analisis entre grupos (OE5)."""

import numpy as np
import pandas as pd
import pytest

from psim.estadistica import comparar_grupos, delta_cliff, holm, intervalo_delta


def test_delta_cliff_extremos() -> None:
    """Separacion total da 1 o -1; muestras identicas dan 0."""
    assert delta_cliff([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert delta_cliff([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    assert delta_cliff([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_delta_cliff_valor_calculado_a_mano() -> None:
    """x = (3, 4), y = (1, 2, 3): 5 pares a favor, 0 en contra, 1 empate -> 5/6."""
    assert delta_cliff([3, 4], [1, 2, 3]) == pytest.approx(5 / 6)


def test_delta_cliff_coincide_con_conteo_de_pares() -> None:
    """La version por rangos coincide con contar todos los pares, con empates."""
    rng = np.random.default_rng(7)
    x = rng.integers(0, 10, size=40).astype(float)
    y = rng.integers(0, 10, size=35).astype(float)
    pares = np.sign(x[:, None] - y[None, :]).mean()
    assert delta_cliff(x, y) == pytest.approx(pares)


def test_delta_cliff_ignora_no_finitos() -> None:
    """Los NaN e infinitos no participan en el calculo."""
    assert delta_cliff([3, 4, np.nan], [1, 2, 3, np.inf]) == pytest.approx(5 / 6)


def test_intervalo_contiene_estimacion_y_es_reproducible() -> None:
    """El intervalo contiene el delta observado y se repite con la misma semilla."""
    rng = np.random.default_rng(1)
    x = rng.normal(1, 1, 80)
    y = rng.normal(0, 1, 80)
    inferior, superior = intervalo_delta(x, y, replicas=500, semilla=3)
    assert inferior <= delta_cliff(x, y) <= superior
    assert (inferior, superior) == intervalo_delta(x, y, replicas=500, semilla=3)


def test_holm_valores_calculados_a_mano() -> None:
    """p = (0.01, 0.04, 0.03, 0.005) -> (0.03, 0.06, 0.06, 0.02)."""
    ajustado = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(ajustado, [0.03, 0.06, 0.06, 0.02])


def test_holm_conserva_nan_y_no_los_cuenta() -> None:
    """Un NaN queda como NaN y m se cuenta solo con los valores validos."""
    ajustado = holm([0.02, np.nan, 0.01])
    assert np.isnan(ajustado[1])
    np.testing.assert_allclose(ajustado[[0, 2]], [0.02, 0.02])


def test_holm_no_supera_uno() -> None:
    """El valor ajustado se limita a 1."""
    assert holm([0.6, 0.7]).max() == pytest.approx(1.0)


def test_comparar_grupos_detecta_diferencia_y_separa_estratos() -> None:
    """Un descriptor desplazado en el grupo 1 da delta positivo en cada estrato."""
    rng = np.random.default_rng(0)
    filas = []
    for sitio in ("A", "B"):
        for target, centro in ((0, 0.0), (1, 1.0)):
            for _ in range(60):
                filas.append(
                    {
                        "dataset_id": sitio,
                        "target": target,
                        "desplazado": rng.normal(centro, 1),
                        "igual": rng.normal(0, 1),
                    }
                )
    datos = pd.DataFrame(filas)
    r = comparar_grupos(datos, ["desplazado", "igual"], replicas=200)
    assert len(r) == len(["A", "B"]) * len(["desplazado", "igual"])
    desplazado = r[r["descriptor"] == "desplazado"]
    assert (desplazado["delta"] > 0.3).all()
    assert desplazado["significativo"].all()
    assert (r.loc[r["descriptor"] == "igual", "delta"].abs() < 0.3).all()
    assert (r["n_con"] == 60).all()
