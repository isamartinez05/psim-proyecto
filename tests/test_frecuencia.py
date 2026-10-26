"""Pruebas del dominio de frecuencia."""

import numpy as np
import pytest

from psim.frecuencia import (
    filtrar,
    fraccion_alta,
    pasaaltas_butterworth,
    pasabajas_butterworth,
    pasabajas_gaussiano,
    pasabanda_butterworth,
    perfil_radial,
    radio,
    recorte_central,
)

FORMA = (64, 64)


def _senoidal(frecuencia: float, n: int = 64) -> np.ndarray:
    """Franjas verticales de una sola frecuencia, en ciclos por pixel."""
    x = np.arange(n)
    return np.tile(np.sin(2 * np.pi * frecuencia * x), (n, 1))


def test_radio_cero_en_la_componente_continua() -> None:
    """La frecuencia del coeficiente (0, 0) es cero y la maxima no pasa de Nyquist."""
    r = radio(FORMA)
    assert r[0, 0] == 0.0
    assert r.max() <= np.sqrt(2) * 0.5 + 1e-12


def test_filtrar_con_unos_devuelve_la_imagen() -> None:
    """Una funcion de transferencia unitaria no cambia la imagen."""
    rng = np.random.default_rng(0)
    imagen = rng.random(FORMA)
    np.testing.assert_allclose(filtrar(imagen, np.ones(FORMA)), imagen, atol=1e-12)


def test_pasabajas_valen_uno_en_continua_y_decrecen() -> None:
    """Ganancia 1 en r = 0 y menor a frecuencias altas."""
    for h in (pasabajas_gaussiano(FORMA, 0.15), pasabajas_butterworth(FORMA, 0.15)):
        assert h[0, 0] == pytest.approx(1.0)
        assert h[0, 32] < h[0, 4]


def test_butterworth_vale_medio_en_el_corte() -> None:
    """En r = corte la ganancia del Butterworth es 0.5."""
    h = pasabajas_butterworth(FORMA, corte=0.25)
    assert h[0, 16] == pytest.approx(0.5)


def test_pasaaltas_y_pasabanda_anulan_la_continua() -> None:
    """Ambos eliminan la componente media de la imagen."""
    constante = np.full(FORMA, 0.7)
    for h in (
        pasaaltas_butterworth(FORMA, 0.05),
        pasabanda_butterworth(FORMA, 0.03, 0.15),
    ):
        assert h[0, 0] == pytest.approx(0.0)
        np.testing.assert_allclose(filtrar(constante, h), 0.0, atol=1e-12)


def test_pasabajas_atenua_una_frecuencia_alta() -> None:
    """Franjas de 0.4 ciclos/pixel casi desaparecen con un corte en 0.1."""
    franjas = _senoidal(0.4)
    filtrada = filtrar(franjas, pasabajas_butterworth(FORMA, 0.1, orden=4))
    assert filtrada.std() < 0.05 * franjas.std()


def test_perfil_radial_tiene_su_pico_en_la_frecuencia_de_la_senal() -> None:
    """El maximo del perfil cae en el anillo de 0.125 ciclos/pixel."""
    frecuencias, potencia = perfil_radial(_senoidal(0.125), bins=32)
    assert frecuencias[np.argmax(potencia)] == pytest.approx(0.125, abs=0.01)


def test_fraccion_alta_distingue_franjas_finas_de_gruesas() -> None:
    """Franjas finas tienen casi toda su potencia sobre 0.25; las gruesas, casi nada."""
    assert fraccion_alta(_senoidal(0.375)) > 0.9
    assert fraccion_alta(_senoidal(0.0625)) < 0.1


def test_recorte_central_tiene_el_lado_pedido_y_cabe_en_la_imagen() -> None:
    """El recorte mide lado x lado aunque el centroide este cerca del borde."""
    imagen = np.zeros((100, 100))
    campo = np.zeros((100, 100), dtype=bool)
    campo[80:, 80:] = True
    assert recorte_central(imagen, campo, lado=40).shape == (40, 40)
