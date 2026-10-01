"""Registra las decisiones tomadas tras ver los datos, con su evidencia.

Cada fila documenta un hallazgo del analisis exploratorio, la evidencia
numerica que lo sustenta, la decision que se tomo, su efecto esperado y la
limitacion que subsiste. Todas las decisiones se tomaron sobre la particion
de entrenamiento.
"""

from __future__ import annotations

import pandas as pd

from psim import resultado

DECISIONES = [
    {
        "ambito": "calidad",
        "hallazgo": (
            "Montgomery entrega siempre 4020x4892 pixeles con independencia de "
            "cuanto colimo el tecnico. El area sin senal llega al 67 por ciento "
            "en el caso mas colimado, frente a una mediana del 1 por ciento en "
            "Shenzhen."
        ),
        "evidencia": (
            "fraccion de pixeles por debajo de 8: mediana 19.4 por ciento en "
            "Montgomery y 1.1 por ciento en Shenzhen sobre 562 imagenes de "
            "entrenamiento; 31 imagenes superan el 25 por ciento, todas de "
            "Montgomery"
        ),
        "decision": (
            "Delimitar el campo irradiado como region binaria, sin forzarlo a "
            "rectangulo y sin rellenar huecos, y usarlo como region de interes "
            "para histogramas, umbralizacion y denominador de la fraccion de area"
        ),
        "efecto_esperado": (
            "Las medidas dejan de depender de la apertura del colimador. El "
            "fondo dentro de la region pasa del 20-32 por ciento que dejaba el "
            "recorte rectangular al 0 por ciento"
        ),
        "limitacion": (
            "Las axilas de un paciente delgado se registran al nivel del fondo y "
            "quedan fuera de la region. Es coherente con medir el cuerpo del "
            "paciente, pero significa que la region no es el torax completo"
        ),
    },
    {
        "ambito": "escala y ROI",
        "hallazgo": (
            "Las mascaras de referencia de Shenzhen estan a 512x512 mientras que "
            "sus imagenes rondan 2940x3000, y contienen valores intermedios "
            "propios del antialiasing del borde"
        ),
        "evidencia": (
            "388 de 486 mascaras requieren reescalado; los valores intermedios "
            "son el 0.14 por ciento de los pixeles y binarizar en cualquier "
            "umbral entre 1 y 255 cambia el area menos del 0.3 por ciento"
        ),
        "decision": (
            "Reescalar la mascara a la imagen por vecino mas cercano y binarizar "
            "en 128. Reducir despues imagen y mascara al mismo lado de trabajo "
            "de 512 pixeles antes de segmentar"
        ),
        "efecto_esperado": (
            "El Dice se calcula sobre la misma rejilla para ambos origenes, y el "
            "tiempo de proceso baja de horas a 26 minutos para las 486 imagenes"
        ),
        "limitacion": (
            "El reescalado no crea detalle que la anotacion no tenga: la mascara "
            "de Shenzhen conserva la precision de 512x512 aunque se represente "
            "sobre una rejilla mayor"
        ),
    },
    {
        "ambito": "caracteristicas",
        "hallazgo": (
            "La simetria izquierda-derecha y la compacidad son los descriptores "
            "mas inestables frente a la estrategia de segmentacion"
        ),
        "evidencia": (
            "error relativo mediano de la simetria entre 0.647 y 0.839 segun la "
            "estrategia, y de la compacidad entre 0.410 y 0.593; frente a 0.066 "
            "a 0.158 de la relacion de aspecto"
        ),
        "decision": (
            "Conservar los cinco descriptores y reportar su error relativo por "
            "separado, en lugar de promediarlos o descartar los inestables"
        ),
        "efecto_esperado": (
            "La respuesta a la pregunta distingue que descriptores son "
            "reproducibles y cuales no, que es lo que el objetivo cuatro pide"
        ),
        "limitacion": (
            "La inestabilidad de la simetria proviene en parte de que una "
            "segmentacion que pierde un campo pulmonar produce asimetria maxima, "
            "de modo que el descriptor mezcla anatomia con fallo de metodo"
        ),
    },
    {
        "ambito": "preprocesamiento",
        "hallazgo": (
            "Ninguna estrategia de realce mejora la linea base sin realce de "
            "forma consistente en los dos origenes"
        ),
        "evidencia": (
            "de las seis combinaciones con realce, la mejor alcanza 4 de 5 "
            "descriptores mejorados en Shenzhen pero solo 2 de 5 en Montgomery; "
            "ninguna llega a 3 de 5 en ambos. El Dice mediano mas alto es de "
            "base_adap con 0.810, sin realce"
        ),
        "decision": (
            "Seleccionar base_adap como combinacion del flujo final y reportar "
            "el resultado negativo del realce como hallazgo, no como fallo"
        ),
        "efecto_esperado": (
            "El flujo final es mas simple y mas reproducible entre sitios: "
            "base_adap da 0.8077 en Montgomery y 0.8102 en Shenzhen"
        ),
        "limitacion": (
            "El resultado es valido para los parametros declarados de cada "
            "realce. Un CLAHE con otro limite de recorte o rejilla podria "
            "comportarse de otro modo; no se exploro ese espacio"
        ),
    },
    {
        "ambito": "evaluacion futura",
        "hallazgo": (
            "El efecto del sitio de adquisicion supera al de la clase: en "
            "intensidad los dos origenes forman nubes casi disjuntas, mientras "
            "que dentro de cada origen las clases apenas difieren"
        ),
        "evidencia": (
            "mediana de intensidad 111.0 en Montgomery y 173.5 en Shenzhen, con "
            "IQR de 151.5 frente a 90.0; dentro de Montgomery la mediana es "
            "110.0 para TB negativo y 111.0 para TB positivo"
        ),
        "decision": (
            "Reportar toda metrica estratificada por origen y exigir "
            "consistencia en ambos como criterio de exito, nunca un promedio "
            "agregado"
        ),
        "efecto_esperado": (
            "Una estrategia que funcione en un solo sitio no puede presentarse "
            "como solucion. El criterio ya descarto a clahe_fijo, que agregado "
            "habria parecido una mejora"
        ),
        "limitacion": (
            "Con dos sitios no se puede distinguir si la diferencia proviene del "
            "detector, del protocolo o de la poblacion. Un tercer origen seria "
            "necesario para separarlos"
        ),
    },
    {
        "ambito": "unidad de analisis",
        "hallazgo": (
            "Las lecturas clinicas declaran dos grupos de imagenes del mismo "
            "paciente en Montgomery, y 90 imagenes comparten sexo y edad con "
            "alguna otra sin que eso pruebe identidad"
        ),
        "evidencia": (
            "dos referencias cruzadas explicitas en el texto de las lecturas; "
            "34 combinaciones de sexo y edad con mas de una imagen, que afectan "
            "a 90 de las 138 imagenes de Montgomery"
        ),
        "decision": (
            "Agrupar solo los dos casos demostrados y declarar el resto como "
            "riesgo residual, en lugar de agrupar por sexo y edad"
        ),
        "efecto_esperado": (
            "No se inventa independencia donde no se puede demostrar, ni se "
            "confunden pacientes distintos reduciendo artificialmente el numero "
            "de unidades"
        ),
        "limitacion": (
            "No se puede descartar que existan mas imagenes del mismo paciente "
            "sin declarar. La agrupacion por sexo y edad tampoco habria servido: "
            "uno de los dos pares demostrados tiene edades distintas porque el "
            "paciente cumplio anos entre estudios"
        ),
    },
]


def main() -> None:
    t = pd.DataFrame(DECISIONES)
    destino = resultado(3, "decisiones.csv")
    t.to_csv(destino, index=False)
    print(f"{len(t)} decisiones escritas en {destino}")
    print()
    print(t[["ambito", "decision"]].to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()
