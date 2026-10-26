"""Pruebas de wavelets y multirresolucion."""

import numpy as np
import pytest

from psim.wavelets import (
    energia_por_nivel,
    energia_subbandas,
    escalograma,
    nombres_subbandas,
    perfil_horizontal,
    reconstruir_sin_detalles,
)


def _tablero(n: int = 64) -> np.ndarray:
    """Tablero de ajedrez de un pixel: todo es detalle de escala fina."""
    yy, xx = np.indices((n, n))
    return ((yy + xx) % 2).astype(float)


def _suave(n: int = 64) -> np.ndarray:
    """Mancha gaussiana ancha: casi todo es aproximacion."""
    yy, xx = np.indices((n, n))
    return np.exp(-((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (2 * 15.0**2))


def test_nombres_y_energias_suman_uno() -> None:
    """Hay 1 + 3 x niveles subbandas y sus energias relativas suman 1."""
    rng = np.random.default_rng(0)
    energias = energia_subbandas(rng.random((64, 64)))
    assert list(energias) == nombres_subbandas(3)
    assert len(energias) == 10
    assert sum(energias.values()) == pytest.approx(1.0)


def test_tablero_concentra_energia_en_el_nivel_1() -> None:
    """Un patron de un pixel tiene su energia en la escala mas fina."""
    niveles = energia_por_nivel(energia_subbandas(_tablero()))
    assert niveles["nivel_1"] > 0.9


def test_mancha_suave_concentra_energia_en_la_aproximacion() -> None:
    """Una estructura ancha queda casi toda en A3."""
    niveles = energia_por_nivel(energia_subbandas(_suave()))
    assert niveles["A3"] > 0.9


def test_reconstruccion_sin_quitar_niveles_es_exacta() -> None:
    """Sin anular detalles, la reconstruccion devuelve la imagen original."""
    rng = np.random.default_rng(1)
    imagen = rng.random((64, 64))
    np.testing.assert_allclose(reconstruir_sin_detalles(imagen, 0), imagen, atol=1e-10)


def test_quitar_el_nivel_1_elimina_el_tablero() -> None:
    """Anular el detalle fino borra casi por completo el patron de un pixel."""
    tablero = _tablero()
    suavizado = reconstruir_sin_detalles(tablero, 1)
    assert suavizado.std() < 0.2 * tablero.std()


def test_perfil_horizontal_cruza_el_campo() -> None:
    """El perfil sale de la fila del centroide y cubre el ancho del campo."""
    imagen = np.arange(100.0).reshape(10, 10)
    campo = np.zeros((10, 10), dtype=bool)
    campo[4:7, 2:8] = True
    fila, perfil = perfil_horizontal(imagen, campo)
    assert fila == 5
    np.testing.assert_allclose(perfil, imagen[5, 2:8])


def test_escalograma_responde_en_la_posicion_del_pulso() -> None:
    """Un pulso en el centro da la respuesta maxima en esa posicion."""
    perfil = np.zeros(128)
    perfil[60:68] = 1.0
    magnitud = escalograma(perfil, np.arange(1, 17))
    assert magnitud.shape == (16, 128)
    columna = np.unravel_index(np.argmax(magnitud), magnitud.shape)[1]
    assert 56 <= columna <= 72
