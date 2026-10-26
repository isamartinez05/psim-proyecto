"""Pruebas de geometria de regiones y morfologia."""

import numpy as np
import pytest

from psim.geometria import (
    apertura,
    black_hat,
    borde,
    cierre,
    dilatacion,
    erosion,
    etiquetar,
    top_hat,
    transformada_distancia,
)


def _cuadrado(n: int = 9, lado: int = 5) -> np.ndarray:
    """Cuadrado lleno centrado en una imagen vacia."""
    mascara = np.zeros((n, n), dtype=bool)
    inicio = (n - lado) // 2
    mascara[inicio : inicio + lado, inicio : inicio + lado] = True
    return mascara


def test_conectividad_cambia_el_numero_de_regiones() -> None:
    """Dos pixeles en diagonal: 2 regiones con vecindad 4 y 1 con vecindad 8."""
    mascara = np.zeros((4, 4), dtype=bool)
    mascara[1, 1] = mascara[2, 2] = True
    assert etiquetar(mascara, 4)[1] == 2
    assert etiquetar(mascara, 8)[1] == 1


def test_conectividad_invalida_falla() -> None:
    """Solo se aceptan vecindades 4 u 8."""
    with pytest.raises(ValueError, match="4 u 8"):
        etiquetar(np.ones((3, 3), dtype=bool), 6)


def test_transformada_distancia_en_el_centro() -> None:
    """El centro de un cuadrado 5x5 esta a 3 pixeles del fondo; el fondo vale 0."""
    distancia = transformada_distancia(_cuadrado())
    assert distancia[4, 4] == pytest.approx(3.0)
    assert distancia[0, 0] == 0.0


def test_borde_de_un_cuadrado() -> None:
    """El borde de un cuadrado 5x5 tiene 16 pixeles."""
    assert borde(_cuadrado()).sum() == 16


def test_erosion_y_dilatacion_encierran_la_region() -> None:
    """Erosion contenida en la region, y la region contenida en la dilatacion."""
    mascara = _cuadrado(21, 11)
    assert np.all(erosion(mascara, 2) <= mascara)
    assert np.all(mascara <= dilatacion(mascara, 2))


def test_apertura_elimina_ruido_y_cierre_rellena_huecos() -> None:
    """La apertura borra un punto aislado y el cierre tapa un hueco pequeno."""
    mascara = _cuadrado(31, 15)
    con_punto = mascara.copy()
    con_punto[2, 2] = True
    assert not apertura(con_punto, 2)[2, 2]
    con_hueco = mascara.copy()
    con_hueco[15, 15] = False
    assert cierre(con_hueco, 2)[15, 15]


def test_top_hat_y_black_hat_detectan_puntos_pequenos() -> None:
    """Un punto claro aparece en el top-hat y uno oscuro en el black-hat."""
    claro = np.full((21, 21), 0.5)
    claro[10, 10] = 1.0
    assert top_hat(claro, 3)[10, 10] == pytest.approx(0.5)
    oscuro = np.full((21, 21), 0.5)
    oscuro[10, 10] = 0.0
    assert black_hat(oscuro, 3)[10, 10] == pytest.approx(0.5)
    constante = np.full((21, 21), 0.5)
    assert np.abs(top_hat(constante, 3)).max() == pytest.approx(0.0)
