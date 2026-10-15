"""Pruebas del modelado supervisado del lab005."""

import numpy as np
import pandas as pd
import pytest

from psim.modeling import (
    coeficientes,
    metricas_regresion,
    pipeline_continuo,
    preparar,
    validar_predictores,
    verificar_grupos,
)

CONTRATO = {
    "estrategia": "base_adap",
    "objetivo": "ref_fraccion_area",
    "predictores": ["fraccion_area", "energia_wavelet_d1"],
}


def _datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla mínima: G1 tiene dos imágenes, D no es elegible."""
    tabla = pd.DataFrame(
        {
            "image_id": ["A", "B", "C", "D"],
            "config_id": ["base_adap"] * 4,
            "eligible_for_model": [1, 1, 1, 0],
            "split": ["train"] * 4,
            "dataset_id": ["X"] * 4,
            "fraccion_area": [0.2, 0.4, 0.3, 0.9],
            "energia_wavelet_d1": [1.0, 3.0, 2.0, 9.0],
            "ref_fraccion_area": [0.25, 0.35, 0.30, 0.99],
        }
    )
    splits = pd.DataFrame(
        {
            "image_id": ["A", "B", "C", "D"],
            "grupo_id": ["G1", "G1", "G2", "G3"],
            "split": ["train"] * 4,
        }
    )
    return tabla, splits


def test_grupo_que_cruza_particiones_falla() -> None:
    """Un mismo paciente no puede estar en train y en prueba."""
    splits = pd.DataFrame({"grupo_id": ["G1", "G1"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="cruzan"):
        verificar_grupos(splits)


@pytest.mark.parametrize(
    "prohibido",
    ["ref_fraccion_area", "err_compacidad", "dice", "target", "image_id", "grupo_id"],
)
def test_predictor_prohibido_falla(prohibido: str) -> None:
    """Objetivo, identificadores y columnas de la máscara manual no son predictores."""
    with pytest.raises(ValueError, match="no permitidos"):
        validar_predictores(["fraccion_area", prohibido], "ref_fraccion_area")


def test_preparar_agrega_por_grupo_y_excluye_no_elegibles() -> None:
    """Las imágenes de un paciente se promedian y las no elegibles salen."""
    tabla, splits = _datos()
    datos, excluidos = preparar(tabla, splits, CONTRATO, "train")
    assert list(datos["grupo_id"]) == ["G1", "G2"]
    g1 = datos.set_index("grupo_id").loc["G1"]
    assert g1["n_imagenes"] == len(["A", "B"])
    assert g1["fraccion_area"] == pytest.approx(0.3)
    assert excluidos.empty


def test_objetivo_faltante_se_excluye_sin_imputar() -> None:
    """Un objetivo faltante se registra como exclusión, no se rellena."""
    tabla, splits = _datos()
    tabla.loc[tabla["image_id"] == "C", "ref_fraccion_area"] = np.nan
    datos, excluidos = preparar(tabla, splits, CONTRATO, "train")
    assert list(excluidos["grupo_id"]) == ["G2"]
    assert "G2" not in set(datos["grupo_id"])


def test_particion_inconsistente_falla() -> None:
    """La tabla y splits.csv deben coincidir en la partición de cada imagen."""
    tabla, splits = _datos()
    splits.loc[splits["image_id"] == "A", "split"] = "test"
    with pytest.raises(ValueError, match="no coinciden"):
        preparar(tabla, splits, CONTRATO, "train")


def test_imputacion_aprende_solo_de_entrenamiento() -> None:
    """Las medianas del imputador son las de train y no cambian al predecir."""
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 10.0], "b": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    modelo = pipeline_continuo(1).fit(train, y)
    np.testing.assert_allclose(modelo.named_steps["impute"].statistics_, train.median())
    modelo.predict(pd.DataFrame({"a": [100.0, np.nan], "b": [50.0, 60.0]}))
    np.testing.assert_allclose(modelo.named_steps["impute"].statistics_, train.median())


def test_metricas_con_valores_conocidos() -> None:
    """y = (1, 2, 3), predicción = (1, 2, 4): MAE = 1/3 y RMSE = raíz de 1/3."""
    m = metricas_regresion(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 4.0]))
    assert m["MAE"] == pytest.approx(1 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(1 / 3))


def test_r2_indefinido_con_referencia_constante() -> None:
    """Con referencia constante, R2 no tiene su interpretación usual."""
    m = metricas_regresion(pd.Series([2.0, 2.0]), pd.Series([1.0, 3.0]))
    assert np.isnan(m["R2"])


def test_ajuste_reproducible_y_terminos_de_grado_2() -> None:
    """Dos ajustes iguales dan los mismos coeficientes, con el intercepto primero."""
    rng = np.random.default_rng(2026)
    x = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = 1 + 2 * x["a"] - x["b"] + 0.5 * x["a"] ** 2
    c1 = coeficientes(pipeline_continuo(2).fit(x, y), ["a", "b"])
    c2 = coeficientes(pipeline_continuo(2).fit(x, y), ["a", "b"])
    np.testing.assert_allclose(c1["coeficiente"], c2["coeficiente"])
    assert list(c1["termino"]) == ["intercepto", "a", "b", "a^2", "a b", "b^2"]
