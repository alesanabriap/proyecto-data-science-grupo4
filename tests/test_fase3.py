import numpy as np
import pandas as pd
from src.fase3.comun import ponderada, preprocesador
from src.fase3.nbi import normalizar

def test_pesos():
    assert ponderada([0, 1], [1, 3]) == .75

def test_nbi_normalizacion():
    assert normalizar("Ñeembucú") == "neembucu"

def test_preprocesador():
    f=pd.DataFrame({
        "edad":[12.,31.,np.nan],
        "quintil_ingreso":[1.,4.,2.],
        "anios_estudio":[np.nan,12.,8.],
        "sexo":["Mujer","Hombre","Mujer"],
        "area":["Rural","Urbana","Rural"],
        "tiene_seguro":["0","1","0"],
        "estado_civil":["Soltero","Casado","Soltero"],
        "departamento":["Central","Asunción","Central"],
    })
    X=preprocesador().fit_transform(f)
    assert X.shape[0]==3
    assert np.isfinite(X).all()
