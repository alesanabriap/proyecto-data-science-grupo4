REG02 trimestrales oficiales EPHC 2024 incluidos para la serie temporal contextual.

Los archivos no contienen S01A ni S03. La serie temporal de la Fase 1 utiliza AREA y el factor trimestral para estimar la composición urbana/rural.

Corrección de la Fase 2: estos archivos contienen las mismas 57.744 personas que la base anual. Al enlazarlos con ella por la clave UPM + NVIVI + NHOGA + L02 se recupera el trimestre de entrevista de cada persona, lo que permite medir enfermedad y seguro por trimestre (ver src/fase2/datos.py, función cargar_trimestres).
