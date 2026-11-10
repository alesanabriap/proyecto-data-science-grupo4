import numpy as np
import pandas as pd
from src.fase3.ejecutar import weighted_mean, normalize, preprocessing, metrics, fit_model, segment

def test_media_ponderada():
    assert weighted_mean([0, 1], [1, 3]) == 0.75

def test_normalizar_departamento():
    assert normalize("Ñeembucú") == "neembucu"

def tabla_sintetica(n=120):
    rng=np.random.default_rng(42)
    return pd.DataFrame({
       "edad":rng.integers(18,80,n), "anios_estudio":rng.integers(0,18,n),
       "log_ingreso":rng.normal(13,1,n),
       "sexo":np.where(rng.random(n)>.5,"Mujer","Hombre"),
       "area":np.where(rng.random(n)>.5,"Urbana","Rural"),
       "departamento":np.where(rng.random(n)>.5,"Asunción","Central"),
       "sin_seguro":rng.integers(0,2,n), "peso":rng.uniform(1,4,n),
    })

def test_clustering_no_usa_objetivo():
    df=tabla_sintetica()
    a,b,modelo=segment(df.iloc[:90],df.iloc[90:],k=3,pick_k=False)
    assert len(a)==90 and len(b)==30
    assert "sin_seguro" not in modelo["xform"].feature_names_in_
    assert a.perfil.nunique() >= 2

def test_tres_modelos_y_baseline():
    df=tabla_sintetica()
    for nombre in ("Baseline","Regresion_logistica","Random_Forest","Hist_Gradient_Boosting"):
        m,cols=fit_model(nombre,df.iloc[:90],False)
        r=metrics(m,cols,df.iloc[90:])
        assert 0 <= r["auc_roc"] <= 1
        assert 0 <= r["accuracy"] <= 1
