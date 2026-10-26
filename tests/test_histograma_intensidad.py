"""Pruebas de histogramas y transformaciones de intensidad."""

import numpy as np
import pytest

from psim.histograma import cdf, distancia, especificar, histograma
from psim.intensidad import (
    diferencia,
    gamma,
    logaritmica,
    normalizar_rango,
    requantizar,
)


def test_histograma_integra_uno_y_cdf_termina_en_uno() -> None:
    """La densidad integra 1 y la acumulada es creciente y termina en 1."""
    rng = np.random.default_rng(0)
    valores = rng.random(5000)
    centros, densidad = histograma(valores, bins=50)
    assert np.sum(densidad) * (centros[1] - centros[0]) == pytest.approx(1.0)
    _, acumulada = cdf(valores, bins=50)
    assert np.all(np.diff(acumulada) >= 0)
    assert acumulada[-1] == pytest.approx(1.0)


def test_especificar_iguala_la_distribucion_y_respeta_el_exterior() -> None:
    """La region queda con la distribucion de referencia; fuera no cambia."""
    rng = np.random.default_rng(1)
    imagen = rng.uniform(0.0, 0.3, (64, 64))
    region = np.zeros((64, 64), dtype=bool)
    region[16:48, 16:48] = True
    referencia = rng.uniform(0.6, 0.9, 20000)
    salida = especificar(imagen, region, referencia)
    assert distancia(salida[region], referencia) < 0.01
    np.testing.assert_array_equal(salida[~region], imagen[~region])


def test_especificar_conserva_el_orden() -> None:
    """La transformacion es monotona: el pixel mas oscuro sigue siendo el mas oscuro."""
    rng = np.random.default_rng(2)
    imagen = rng.random((32, 32))
    region = np.ones((32, 32), dtype=bool)
    salida = especificar(imagen, region, rng.normal(0.5, 0.1, 5000))
    assert np.argmin(salida) == np.argmin(imagen)
    assert np.argmax(salida) == np.argmax(imagen)


def test_gamma_y_log_fijan_los_extremos() -> None:
    """0 va a 0 y 1 va a 1; gamma 1 es la identidad."""
    extremos = np.array([0.0, 1.0])
    for salida in (gamma(extremos, 0.5), gamma(extremos, 2.0), logaritmica(extremos)):
        np.testing.assert_allclose(salida, extremos)
    rng = np.random.default_rng(3)
    imagen = rng.random((8, 8))
    np.testing.assert_allclose(gamma(imagen, 1.0), imagen)


def test_normalizar_rango_lleva_percentiles_a_cero_y_uno() -> None:
    """Con percentiles 0 y 100, el minimo va a 0 y el maximo a 1."""
    imagen = np.linspace(0.2, 0.6, 100).reshape(10, 10)
    region = np.ones((10, 10), dtype=bool)
    salida = normalizar_rango(imagen, region, 0.0, 100.0)
    assert salida.min() == pytest.approx(0.0)
    assert salida.max() == pytest.approx(1.0)


def test_requantizar_deja_los_niveles_pedidos() -> None:
    """Con 1 bit quedan 2 niveles; con 2 bits, a lo sumo 4."""
    imagen = np.linspace(0.0, 1.0, 1000)
    assert set(np.unique(requantizar(imagen, 1))) == {0.0, 1.0}
    assert len(np.unique(requantizar(imagen, 2))) == 4


def test_diferencia_es_cero_entre_imagenes_iguales() -> None:
    """El mapa de diferencia de una imagen consigo misma es cero."""
    rng = np.random.default_rng(4)
    imagen = rng.random((16, 16))
    assert diferencia(imagen, imagen).max() == 0.0
