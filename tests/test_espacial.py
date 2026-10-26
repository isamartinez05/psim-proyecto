"""Pruebas de las operaciones de vecindad y las metricas de OE2."""

import numpy as np
import pytest

from psim.espacial import (
    a_flotante,
    bordes,
    conservacion_bordes,
    contraste_rms,
    emboss,
    filtro_gaussiano,
    filtro_media,
    filtro_mediana,
    ruido_estimado,
    unsharp,
)


def _escalon() -> np.ndarray:
    """Imagen 64x64 con un borde vertical en la mitad."""
    imagen = np.zeros((64, 64))
    imagen[:, 32:] = 1.0
    return imagen


def test_a_flotante_lleva_uint8_a_cero_uno() -> None:
    """255 en uint8 equivale a 1.0."""
    resultado = a_flotante(np.array([[0, 255]], dtype=np.uint8))
    np.testing.assert_allclose(resultado, [[0.0, 1.0]])


def test_filtros_conservan_imagen_constante() -> None:
    """Una imagen constante no cambia con media, mediana ni gaussiano."""
    constante = np.full((32, 32), 0.4)
    for filtro in (filtro_media, filtro_mediana, filtro_gaussiano):
        np.testing.assert_allclose(filtro(constante), constante, atol=1e-12)


def test_mediana_elimina_ruido_impulsivo_aislado() -> None:
    """Un pixel blanco aislado desaparece con la mediana 3x3."""
    imagen = np.zeros((16, 16))
    imagen[8, 8] = 1.0
    assert filtro_mediana(imagen, tamano=3)[8, 8] == 0.0


def test_bordes_responden_en_el_escalon() -> None:
    """Cada operador da respuesta maxima junto al borde y nula lejos de el."""
    for nombre, magnitud in bordes(_escalon()).items():
        assert magnitud[:, 30:34].max() > 0, nombre
        assert magnitud[:, 5:10].max() == pytest.approx(0.0, abs=1e-12), nombre


def test_unsharp_y_emboss_quedan_en_cero_uno() -> None:
    """Las salidas realzadas se mantienen en [0, 1]."""
    rng = np.random.default_rng(0)
    imagen = rng.random((32, 32))
    for salida in (unsharp(imagen), emboss(imagen)):
        assert salida.min() >= 0.0
        assert salida.max() <= 1.0


def test_contraste_rms_valor_conocido() -> None:
    """Mitad 0 y mitad 1: la desviacion estandar es 0.5."""
    region = np.ones((64, 64), dtype=bool)
    assert contraste_rms(_escalon(), region) == pytest.approx(0.5)


def test_ruido_estimado_crece_con_el_ruido() -> None:
    """Mas ruido gaussiano agregado da mayor ruido estimado."""
    rng = np.random.default_rng(1)
    base = np.full((128, 128), 0.5)
    bajo = ruido_estimado(base + rng.normal(0, 0.01, base.shape))
    alto = ruido_estimado(base + rng.normal(0, 0.05, base.shape))
    assert alto > bajo


def test_conservacion_bordes_es_uno_sin_cambios() -> None:
    """Una imagen comparada consigo misma conserva sus bordes por completo."""
    region = np.ones((64, 64), dtype=bool)
    imagen = filtro_gaussiano(_escalon(), sigma=1.0)
    assert conservacion_bordes(imagen, imagen, region) == pytest.approx(1.0)
