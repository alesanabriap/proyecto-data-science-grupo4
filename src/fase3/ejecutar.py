"""Fase 3 oficial: perfiles socioeconómicos y clasificación de cobertura médica."""
from __future__ import annotations
import json
from pathlib import Path
import unicodedata
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score, accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    silhouette_score, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ/"data/raw/REG02_EPHC_ANUAL_2024.csv"
TEMP = RAIZ/"data/temporal"
NBI = RAIZ/"data/externos/nbi_departamental_2022.csv"
OUT = RAIZ/"output/fase3"
SEED = 42
KEY = ["UPM", "NVIVI", "NHOGA", "L02"]
NUM = ["edad", "anios_estudio", "log_ingreso"]
CAT = ["sexo", "area", "departamento"]
COLUMNS = NUM+CAT
SEG_CAT = ["sexo", "area"]
SEG_NUM = NUM
SEGM = SEG_NUM+SEG_CAT
LABEL = "sin_seguro"
DPTO = {
    0:"Asunción",1:"Concepción",2:"San Pedro",3:"Cordillera",4:"Guairá",
    5:"Caaguazú",6:"Caazapá",7:"Itapúa",8:"Misiones",9:"Paraguarí",
    10:"Alto Paraná",11:"Central",12:"Ñeembucú",13:"Amambay",
    14:"Canindeyú",15:"Presidente Hayes",
}

def weighted_mean(v,w):
    return float(np.average(np.asarray(v,dtype=float),weights=np.asarray(w,dtype=float)))

def normalize(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(x for x in s if not unicodedata.combining(x)).strip().lower()

def load():
    use = KEY+["P02","P06","AREA","DPTO","S01A","añoest","ipcm","FACTOR"]
    df = pd.read_csv(RAW, sep=";", encoding="utf-8-sig", usecols=use,
                     low_memory=False, dtype={"ipcm":str,"añoest":str})
    n_original=len(df)
    chunks=[]
    for t in range(1,5):
        x=pd.read_csv(TEMP/f"REG02_EPHC_2024_T{t}.csv",sep=";",
                      encoding="utf-8-sig",usecols=KEY,low_memory=False)
        x["trimestre"]=t
        chunks.append(x)
    quarters=pd.concat(chunks,ignore_index=True)
    if quarters.duplicated(KEY).any():
        raise ValueError("La clave combinada UPM/vivienda/hogar/persona se repite entre trimestres.")
    df=df.merge(quarters,on=KEY,how="left",validate="one_to_one")
    if df.trimestre.isna().any():raise ValueError("Faltan trimestres para personas de la base anual")
    for x in ["P02","P06","AREA","DPTO","S01A","FACTOR"]:
        df[x]=pd.to_numeric(df[x],errors="coerce")
    df["edad"]=df.P02
    df["anios_estudio"]=pd.to_numeric(df["añoest"].str.replace(",","."), errors="coerce")
    df.loc[df.anios_estudio==99,"anios_estudio"]=np.nan
    df["ingreso_per_capita"]=pd.to_numeric(df["ipcm"].str.replace(",","."),errors="coerce")
    df.loc[df.ingreso_per_capita<0,"ingreso_per_capita"]=np.nan
    df["log_ingreso"]=np.log1p(df.ingreso_per_capita)
    df["peso"]=df.FACTOR
    df["departamento"]=df.DPTO.map(DPTO)
    df["sexo"]=df.P06.map({1:"Hombre",6:"Mujer"})
    df["area"]=df.AREA.map({1:"Urbana",6:"Rural"})
    df["sin_seguro"]=df.S01A.eq(7).astype(int)
    df=df.loc[(df.edad>=18)&df.S01A.isin([1,2,3,4,5,6,7])&
              df.peso.gt(0)&df.departamento.notna()&df.area.notna()&df.sexo.notna()].copy()
    df["upm_id"]=df.UPM.astype(str)
    if set(df.trimestre.unique())!={1,2,3,4}:
        raise AssertionError("No se identificaron los cuatro trimestres.")
    train=set(df.loc[df.trimestre<4,"upm_id"])
    test=set(df.loc[df.trimestre==4,"upm_id"])
    if train & test: raise AssertionError("UPM compartidas entre desarrollo y prueba.")
    df["trimestre"]=df.trimestre.astype(int)
    return df, n_original

def preprocessing(num=NUM,cat=CAT):
    return ColumnTransformer([
        ("numericas",Pipeline([
            ("imputar",SimpleImputer(strategy="median",add_indicator=True)),
            ("escalar",StandardScaler()),
        ]),num),
        ("categoricas",Pipeline([
            ("imputar",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]),cat),
    ],sparse_threshold=0)

def segment(train, other, k=None, pick_k=True):
    xform=preprocessing(SEG_NUM,SEG_CAT)
    xt=xform.fit_transform(train[SEGM])
    scores={}
    if pick_k:
        sample=np.random.default_rng(SEED).choice(len(xt),size=min(2400,len(xt)),replace=False)
        for candidate in [3,4,5,6]:
            model=MiniBatchKMeans(n_clusters=candidate,n_init=5,random_state=SEED,
                                  batch_size=2048,max_iter=160)
            model.fit(xt)
            selected=model.predict(xt[sample])
            scores[candidate]=float(silhouette_score(xt[sample],selected)) if len(np.unique(selected))>1 else -1
        k=max(scores,key=scores.get)
    m=MiniBatchKMeans(n_clusters=k,n_init=7,random_state=SEED,batch_size=2048,max_iter=200)
    m.fit(xt)
    a=train.copy()
    b=other.copy()
    a["perfil"]=m.predict(xt).astype(str)
    b["perfil"]=m.predict(xform.transform(other[SEGM])).astype(str)
    return a,b,{"k":int(k),"scores":scores,"xform":xform,"model":m}

def stability(train,k,base):
    rng=np.random.default_rng(29)
    ix=rng.choice(len(train),size=min(5500,len(train)),replace=False)
    x=base["xform"].transform(train.iloc[ix][SEGM])
    original=base["model"].predict(x)
    others=[]
    for seed in (101,202,303):
        alt=MiniBatchKMeans(n_clusters=k,n_init=5,random_state=seed,batch_size=1500,max_iter=180)
        alt.fit(base["xform"].transform(train[SEGM]))
        others.append(float(adjusted_rand_score(original,alt.predict(x))))
    return others

def fit_model(name,data,with_cluster):
    features=COLUMNS+(["perfil"] if with_cluster else [])
    cat=CAT+(["perfil"] if with_cluster else [])
    models={
        "Baseline":DummyClassifier(strategy="most_frequent"),
        "Regresion_logistica":LogisticRegression(C=1,max_iter=700),
        "Random_Forest":RandomForestClassifier(n_estimators=120,max_depth=13,
                                                min_samples_leaf=35,random_state=SEED,n_jobs=-1),
        "Hist_Gradient_Boosting":HistGradientBoostingClassifier(
            learning_rate=.06,max_iter=150,max_leaf_nodes=15,l2_regularization=4,
            random_state=SEED),
    }
    pipe=Pipeline([("procesar",preprocessing(NUM,cat)),
                   ("modelo",models[name])])
    pipe.fit(data[features],data[LABEL],modelo__sample_weight=data.peso)
    return pipe,features

def metrics(model,features,data):
    p=model.predict_proba(data[features])[:,1]
    y=data[LABEL].to_numpy()
    w=data.peso.to_numpy()
    yp=(p>=.5).astype(int)
    return {
        "n":int(len(data)),"prevalencia_sin_seguro_pct":100*weighted_mean(y,w),
        "accuracy":float(accuracy_score(y,yp,sample_weight=w)),
        "balanced_accuracy":float(balanced_accuracy_score(y,yp,sample_weight=w)),
        "precision":float(precision_score(y,yp,sample_weight=w,zero_division=0)),
        "recall":float(recall_score(y,yp,sample_weight=w,zero_division=0)),
        "f1":float(f1_score(y,yp,sample_weight=w,zero_division=0)),
        "auc_roc":float(roc_auc_score(y,p,sample_weight=w)),
        "brier":float(brier_score_loss(y,p,sample_weight=w)),
    }

def weighted_by_group(df,feature):
    rows=[]
    for value,g in df.groupby(feature,dropna=False):
        rows.append({"variable":feature,"categoria":str(value),"n":len(g),
                     "cobertura_pct":100*(1-weighted_mean(g[LABEL],g.peso)),
                     "sin_cobertura_pct":100*weighted_mean(g[LABEL],g.peso)})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df,original=load()
    dev=df.loc[df.trimestre<4].copy()
    test=df.loc[df.trimestre==4].copy()
    descriptions=[]
    for c in ["area","sexo","departamento"]:
        descriptions.extend(weighted_by_group(df,c))
    df["grupo_edad"]=pd.cut(df.edad,bins=[17,24,34,44,59,130],
                             labels=["18-24","25-34","35-44","45-59","60+"])
    descriptions+=weighted_by_group(df,"grupo_edad")
    df["grupo_ingreso"]=pd.qcut(df.ingreso_per_capita.rank(method="first"),q=5,
                                labels=["Q1","Q2","Q3","Q4","Q5"])
    descriptions+=weighted_by_group(df,"grupo_ingreso")
    pd.DataFrame(descriptions).to_csv(OUT/"brechas_descriptivas.csv",index=False)

    tr,te,cluster=segment(dev,test)
    ari=stability(dev,cluster["k"],cluster)
    cluster_scores=pd.DataFrame({"k":list(cluster["scores"].keys()),
                                  "silhouette":list(cluster["scores"].values())})
    cluster_scores.to_csv(OUT/"seleccion_k.csv",index=False)
    desc=[]
    for part,part_name in [(tr,"desarrollo_T1_T3"),(te,"prueba_T4")]:
        for profile,g in part.groupby("perfil"):
            desc.append({"conjunto":part_name,"perfil":profile,"n":len(g),
                         "cobertura_pct":100*(1-weighted_mean(g.sin_seguro,g.peso)),
                         "sin_cobertura_pct":100*weighted_mean(g.sin_seguro,g.peso),
                         "edad_media":weighted_mean(g.edad,g.peso),
                         "ingreso_mediano":float(g.ingreso_per_capita.median()),
                         "estudios_media":weighted_mean(g.anios_estudio.fillna(g.anios_estudio.median()),g.peso),
                         "rural_pct":100*weighted_mean(g.area.eq("Rural"),g.peso)})
    pd.DataFrame(desc).to_csv(OUT/"resumen_perfiles.csv",index=False)
    details={"k":cluster["k"],"silhouette":cluster["scores"][cluster["k"]],
             "estabilidad_ari":ari}
    with (OUT/"detalle_clusters.json").open("w") as f:json.dump(details,f,indent=2)
    joblib.dump({"preprocessor":cluster["xform"],"model":cluster["model"],"columns":SEGM},
                OUT/"cluster_model.joblib",compress=3)
    # T1 -> T2 and T1+T2 -> T3. Clusters fitted ONLY on corresponding training fold.
    val_rows=[]
    for stop,next_q in [(1,2),(2,3)]:
        train=df.loc[df.trimestre<=stop]
        valid=df.loc[df.trimestre==next_q]
        fold_train,fold_valid,_=segment(train,valid,pick_k=True)
        for with_c in [False,True]:
            aa,bb=(fold_train,fold_valid) if with_c else (train,valid)
            for model_name in ["Baseline","Regresion_logistica","Random_Forest","Hist_Gradient_Boosting"]:
                model,features=fit_model(model_name,aa,with_c)
                val_rows.append({"modelo":model_name,"con_cluster":with_c,"val_trimestre":next_q,
                                 **metrics(model,features,bb)})
    val=pd.DataFrame(val_rows)
    val.to_csv(OUT/"validacion_temporal.csv",index=False)
    ranked=val.groupby(["modelo","con_cluster"],as_index=False).agg(
        auc_validacion=("auc_roc","mean"),brier_validacion=("brier","mean"))
    ranked.to_csv(OUT/"ranking_validacion.csv",index=False)
    # All fixed configurations are reported on final T4 once, for transparent comparison.
    final=[]
    best_models={}
    for with_c in [False,True]:
        train,testset=(tr,te) if with_c else (dev,test)
        for name in ["Baseline","Regresion_logistica","Random_Forest","Hist_Gradient_Boosting"]:
            model,cols=fit_model(name,train,with_c)
            final.append({"modelo":name,"con_cluster":with_c,
                          "conjunto":"entrenamiento","n":len(train),
                          **metrics(model,cols,train)})
            final.append({"modelo":name,"con_cluster":with_c,
                          "conjunto":"prueba_T4","n":len(testset),
                          **metrics(model,cols,testset)})
            best_models[(name,with_c)]=(model,cols)
    final_df=pd.DataFrame(final)
    final_df.to_csv(OUT/"comparacion_test.csv",index=False)
    winner=ranked.loc[ranked.modelo.ne("Baseline")].sort_values(
        ["auc_validacion","brier_validacion"],ascending=[False,True]).iloc[0]
    winning_model,winning_cols=best_models[(winner.modelo,bool(winner.con_cluster))]
    win_test=te if bool(winner.con_cluster) else test
    prob=winning_model.predict_proba(win_test[winning_cols])[:,1]
    predict=(prob>=.5).astype(int)
    cm=confusion_matrix(win_test.sin_seguro,predict,
                        labels=[0,1],sample_weight=win_test.peso)
    fig,ax=plt.subplots(figsize=(7,5))
    ConfusionMatrixDisplay(cm,display_labels=["Con seguro","Sin seguro"]).plot(
        ax=ax,colorbar=False,values_format=".0f")
    ax.set_title("Matriz de confusión - ganador validación temporal (T4)")
    fig.tight_layout()
    fig.savefig(OUT/"matriz_confusion.png",dpi=180)
    plt.close(fig)
    joblib.dump({"model":winning_model,"features":winning_cols,
                 "name":winner.modelo,"cluster":bool(winner.con_cluster)},
                OUT/"mejor_clasificador.joblib",compress=3)
    # NBI: aggregate department-level only. EPHC covers 16 units.
    nbi=pd.read_csv(NBI)
    if nbi.departamento.duplicated().any():raise AssertionError("NBI departamental duplicado")
    rows=[]
    for name,g in df.groupby("departamento"):
        rows.append({"departamento":name,"n":len(g),
                     "sin_seguro_pct":100*weighted_mean(g.sin_seguro,g.peso),
                     "con_seguro_pct":100*(1-weighted_mean(g.sin_seguro,g.peso))})
    by_dep=pd.DataFrame(rows)
    by_dep["key"]=by_dep.departamento.map(normalize)
    nbi["key"]=nbi.departamento.map(normalize)
    cross=by_dep.merge(nbi,on="key",how="left",validate="one_to_one",indicator=True)
    if (cross["_merge"]!="both").any():raise AssertionError("Sin coincidencia NBI")
    cross.drop(columns=["key","_merge"]).to_csv(OUT/"cruce_nbi_cobertura.csv",index=False)
    corr=[]
    for col in ["nbi_una_o_mas_pct","nbi_vivienda_pct","nbi_sanitaria_pct",
                "nbi_educacion_pct","nbi_subsistencia_pct"]:
        rho,p=spearmanr(cross[col],cross.sin_seguro_pct)
        corr.append({"variable_nbi":col,"rho":float(rho),"p_exploratorio":float(p),
                     "n_territorios":len(cross)})
    pd.DataFrame(corr).to_csv(OUT/"correlaciones_nbi.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.scatter(cross.nbi_una_o_mas_pct,cross.sin_seguro_pct)
    for _,r in cross.iterrows():
        ax.annotate(str(r.departamento_x),(r.nbi_una_o_mas_pct,r.sin_seguro_pct),
                    xytext=(3,3),textcoords="offset points",fontsize=7)
    ax.set(xlabel="Hogares con al menos una NBI (2022, %)",
           ylabel="Adultos sin seguro médico (2024, %)",
           title="NBI vs falta de seguro - correlación ecológica departamental")
    fig.tight_layout()
    fig.savefig(OUT/"nbi_cobertura.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    display=pd.DataFrame(desc)
    sel=display[display.conjunto=="prueba_T4"].sort_values("sin_cobertura_pct")
    ax.barh(["Perfil "+str(p) for p in sel.perfil],sel.sin_cobertura_pct)
    ax.set(xlabel="Adultos sin seguro médico (% ponderado)",title="Perfil socioeconómico y cobertura - T4")
    fig.tight_layout()
    fig.savefig(OUT/"perfiles_cobertura.png",dpi=180)
    plt.close(fig)
    best_without=ranked.loc[(ranked.modelo.ne("Baseline")) & (~ranked.con_cluster)].sort_values(
        "auc_validacion",ascending=False).iloc[0]
    best_with=ranked.loc[(ranked.modelo.ne("Baseline")) & (ranked.con_cluster)].sort_values(
        "auc_validacion",ascending=False).iloc[0]
    summary={
        "base":"EPHC 2024, REG02 anual original y trimestres originales",
        "n_original":original,"n_adultos":len(df),"n_desarrollo":len(dev),"n_prueba":len(test),
        "prevalencia_falta_seguro_pct":100*weighted_mean(df.sin_seguro,df.peso),
        "k_clusters":cluster["k"],"silhouette":cluster["scores"][cluster["k"]],
        "estabilidad_ari":ari,
        "mejor_modelo":winner.modelo,"ganador_con_cluster":bool(winner.con_cluster),
        "ganador_auc_validacion":float(winner.auc_validacion),
        "mejor_auc_validacion_sin_cluster":float(best_without.auc_validacion),
        "mejor_auc_validacion_con_cluster":float(best_with.auc_validacion),
        "auc_mejora_cluster":float(best_with.auc_validacion-best_without.auc_validacion),
        "ganador_en_prueba":metrics(winning_model,winning_cols,win_test),
        "n_departamentos":len(cross),
        "nbi_correlaciones":corr,
        "advertencias":[
            "Piloto exploratorio: ninguna relación implica causalidad.",
            "NBI de hogares 2022 se cruza con EPHC personas adultas 2024 solo a nivel departamental.",
            "Las métricas ponderadas no reemplazan un análisis completo del diseño complejo de la EPHC.",
            "La comparación entre modelos se selecciona en T2 y T3; T4 reservado para evaluación final.",
            "Se trabaja sobre adultos para evitar que diferencias por dependencia infantil dominen los perfiles.",
        ],
    }
    with (OUT/"resultados.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
