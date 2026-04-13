# IDEAS - Roadmap de mejora para VFM

## 1) Contexto

El enfoque VFM actual (backbone foundation congelado + cabeza lineal) funciona sorprendentemente bien por su simplicidad, estabilidad y bajo riesgo de overfitting.

Principio rector para este roadmap:

- Mantener la simplicidad como ventaja competitiva.
- Solo añadir complejidad si mejora robustez OOD de forma medible.
- Evitar cambios que rompan comparabilidad con el baseline actual.

## 2) Baseline de referencia (congelar antes de mejorar)

Antes de tocar arquitectura o entrenamiento, toda variante debe compararse contra este baseline fijo:

- Backbone: DINOv3 (config actual de scripts/vfm).
- Entrenamiento: head-only, backbone frozen.
- Preprocesado: resize + center-crop (sin augmentations, como baseline VFM).
- Optimización: AdamW, hiperparámetros base actuales.
- Evaluación: mismas particiones CSV y mismas métricas globales + por benchmark.

Checklist mínimo de comparabilidad:

- Misma seed (o reporte promedio y desviación en 3 seeds).
- Mismo train/val/test split.
- Mismo protocolo de inferencia.
- Reportar Accuracy, Balanced Accuracy, AUC, F1, Recall, Precision.

## 3) Objetivo principal del roadmap

Subir robustez OOD sin empeorar el rendimiento ID.

Criterio sugerido de éxito:

- +2 a +4 puntos en métricas OOD (Accuracy o AUC),
- con caída <= 0.5 puntos en ID,
- y coste de inferencia razonable para despliegue.

## 4) Backlog priorizado

## P0 - Quick wins (alto impacto, bajo riesgo)

### P0.1 LoRA ligera sobre backbone (sin full fine-tuning)

Hipótesis:

- Un ajuste ligero del backbone mejora adaptación a generadores no vistos sin perder la estabilidad del enfoque simple.

Acción:

- Añadir modo opcional LoRA sobre bloques seleccionados del backbone.
- Comparar contra frozen total.
- Probar grid pequeña: rank {4, 8, 16}, alpha {16, 32}.

Éxito:

- Mejora OOD >= 2 puntos sin degradación ID relevante.

Riesgo:

- Mayor consumo VRAM/tiempo.

---

### P0.2 Augmentaciones forenses controladas en train

Hipótesis:

- Robustez OOD mejora si exponemos el modelo a degradaciones realistas, sin destruir señales forenses.

Acción:

- Pipeline suave (solo train):
  - JPEG compression quality en rango moderado.
  - Resize up/down moderado.
  - Blur suave ocasional.
- Mantener val/test sin cambios.
- Mantener baseline sin aug como control.

Éxito:

- Menor brecha ID vs OOD.

Riesgo:

- Augmentaciones agresivas pueden borrar artefactos útiles.

---

### P0.3 Calibración y umbral operativo

Hipótesis:

- El ranking puede ser bueno pero la probabilidad mal calibrada; calibrar mejora decisiones de producto.

Acción:

- Temperature scaling en validación.
- Evaluar umbral fijo vs umbral por escenario (alta precisión vs alto recall).

Éxito:

- Mejor trade-off operativo sin reentrenar backbone.

Riesgo:

- Ganancia limitada si el problema principal es representación.

## P1 - Mejoras estructurales moderadas

### P1.1 Ir más allá de CLS token (agregación espacial ligera)

Hipótesis:

- Artefactos sintéticos locales se benefician de pooling/aggregación sobre tokens espaciales.

Acción:

- Implementar agregador ligero sobre tokens (mean/attention pooling simple).
- Comparar contra CLS-only.

Éxito:

- Incremento consistente en OOD y casos frontera.

Riesgo:

- Complejidad adicional en forward/inferencia.

---

### P1.2 Ensemble compacto opcional

Hipótesis:

- Backbones foundation distintos capturan errores complementarios.

Acción:

- Ensamble de 2 modelos (ejemplo: DINOv3 + DINOv2-L), por promedio de probabilidades.
- Mantener modo single-model como default.

Éxito:

- Mejora medible en OOD con coste asumible.

Riesgo:

- Latencia y coste de inferencia aumentan.

---

### P1.3 Endurecer fallback de imágenes corruptas

Hipótesis:

- El fallback negro puede introducir sesgos no deseados.

Acción:

- Instrumentar contador de fallbacks por split y benchmark.
- Añadir política configurable: black / skip / raise (según entorno).
- Reportar impacto en métricas.

Éxito:

- Menos riesgo de sesgo silencioso.

Riesgo:

- Cambios en dataset efectivo si se usa skip.

## P2 - Ingeniería y consolidación

### P2.1 Unificación con utilidades reutilizables del repo

Acción:

- Reducir duplicación entre scripts/vfm y src en:
  - carga robusta de imágenes,
  - utilidades de evaluación,
  - componentes de agregación reutilizables.

Beneficio:

- Menor deuda técnica y menos bugs por divergencia.

---

### P2.2 Estandarizar trazabilidad de experimentos

Acción:

- Cada corrida debe guardar:
  - config completa,
  - seed,
  - commit hash,
  - métricas globales,
  - métricas por benchmark,
  - artefacto de predicciones.

Beneficio:

- Comparación objetiva y reproducible entre variantes.

## 5) Track de I+D (no bloqueante)

La línea multi-scale/tube de new_idea.md se mantiene como investigación paralela.

Regla de entrada:

- Solo priorizarla sobre P0/P1 si quick wins no alcanzan el objetivo OOD.

## 6) Plan de ejecución por fases

Fase A (1-2 semanas):

- P0.1 + P0.2 + P0.3 en paralelo.
- Entrega: tabla comparativa baseline vs variantes.

Fase B (1-2 semanas):

- P1.1 + P1.2 según resultados de Fase A.
- Entrega: recomendación single-model vs ensemble.

Fase C (1 semana):

- P1.3 + P2.1 + P2.2.
- Entrega: pipeline más robusto y mantenible.

## 7) Matriz de decisiones (go/no-go)

Promover una mejora a "default":

- Mejora OOD clara y estable,
- sin regresión material en ID,
- coste adicional aceptable para entrenamiento/inferencia,
- implementación mantenible.

Descartar o dejar opcional:

- Ganancias marginales con complejidad alta,
- resultados inestables entre seeds,
- dependencia fuerte de tuning frágil.

## 8) Riesgos clave y mitigaciones

Riesgo 1: sobreoptimizar para un benchmark concreto.

- Mitigación: evaluar siempre por grupos de benchmark + métricas globales.

Riesgo 2: pérdida de simplicidad del baseline.

- Mitigación: conservar siempre un path "simple baseline" como referencia viva.

Riesgo 3: mejoras no reproducibles.

- Mitigación: estandarizar seeds, logging y artefactos por corrida.

## 9) Template de reporte por experimento

Usar este formato para cada prueba:

- Nombre experimento:
- Cambio aplicado:
- Config clave:
- Seed(s):
- Resultados ID:
- Resultados OOD:
- Costo (tiempo, VRAM, latencia):
- Decisión (promover / iterar / descartar):
- Notas:

## 10) Resumen ejecutivo

VFM ya demostró ser fuerte con un enfoque mínimo. El plan correcto no es complicarlo de golpe, sino ejecutar mejoras incrementales y medibles que ataquen el gap OOD y mejoren confiabilidad operativa. Este documento prioriza exactamente eso.
