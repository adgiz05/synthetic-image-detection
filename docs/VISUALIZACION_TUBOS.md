# Visualización de Tubos Multi-Escala

Este documento describe cómo visualizar y verificar que el pipeline de carga de datos funciona correctamente.

## 📋 Scripts Disponibles

### 1. **Quick Test** - Verificación rápida
```bash
python scripts/quick_test_data.py
```

**Qué hace:**
- Carga un batch pequeño
- Imprime formas (shapes) de todos los tensores
- Muestra información de labels y centros
- Verifica memoria utilizada

**Úsalo para:** Verificar rápidamente que todo funciona antes de generar visualizaciones.

---

### 2. **Visualización Completa**
```bash
./experiments/visualize_batch.sh
```

O manualmente:
```bash
python scripts/visualize_batch.py \
    --train_csv data/train.csv \
    --root_dir data/ \
    --num_tubes 8 \
    --scales 64 128 256 \
    --target_size 128 \
    --num_views 2 \
    --batch_size 2 \
    --num_samples 2 \
    --num_tubes_per_sample 4 \
    --output_dir visualizations
```

**Qué genera:**
- `image_X_tube_Y.png`: Visualización completa de un tubo (todas las escalas y vistas)
- `scale_comparison_*.png`: Comparación de todas las escalas para un mismo tubo
- `view_comparison_*.png`: Comparación de todas las vistas (augmentaciones) para una escala

**Úsalo para:** 
- Verificar que los tubos multi-escala se extraen correctamente
- Verificar que las augmentaciones se aplican bien
- Ver visualmente la progresión de escalas
- Comprobar que el mismo centro espacial se mantiene entre escalas

---

## 🎯 Qué Verificar en las Visualizaciones

### 1. **Tubos Multi-Escala** (`image_X_tube_Y.png`)

Deberías ver:
- **Misma región, diferentes escalas**: Cada fila muestra la misma ubicación espacial pero con diferente campo de visión
- **Escala 0 (pequeña)**: Detalles muy locales, microtextura
- **Escala 1 (media)**: Contexto local
- **Escala 2 (grande)**: Contexto más amplio
- **Coherencia espacial**: Las escalas deberían mostrar la misma región (ajustado por el zoom)

### 2. **Comparación de Escalas** (`scale_comparison_*.png`)

Verifica:
- ✅ Las tres escalas muestran progresivamente más contexto
- ✅ El centro permanece aproximadamente en el mismo punto
- ✅ La calidad visual es buena (no hay artefactos de resize extremos)

### 3. **Comparación de Vistas** (`view_comparison_*.png`)

Verifica:
- ✅ **View 0 (original)**: Debería ser la menos modificada
- ✅ **View 1+ (augmented)**: Debería tener degradaciones sutiles:
  - Ligera compresión JPEG/WebP
  - Posible blur/sharpen suave
  - Pequeñas variaciones en calidad
- ❌ **NO deberían tener**: Cambios semánticos drásticos, rotaciones, color shifts extremos

---

## 📊 Estructura de un Batch

```
batch = {
    'tubes': [B, N_tubes, K_scales, V_views, C, H, W]
             [2,    8,        3,        2,    3, 128, 128]
             
    'tube_centers': [B, N_tubes, 2]
                    [2,    8,    2]  # (cy, cx) normalized
                    
    'labels': [B]  # 0=real, 1=synthetic
              [2]
              
    'model_labels': [B]  # optional: generator class
                    [2]
}
```

### Ejemplo de indexación:

```python
# Primer tubo de la primera imagen, escala media, vista original
patch = batch['tubes'][0, 0, 1, 0]  # [C, H, W]

# Todos los tubos de una imagen
image_tubes = batch['tubes'][0]  # [N, K, V, C, H, W]

# Todas las escalas de un tubo
tube_scales = batch['tubes'][0, 0]  # [K, V, C, H, W]

# Todas las vistas de un patch
patch_views = batch['tubes'][0, 0, 1]  # [V, C, H, W]
```

---

## 🔧 Configuración de Tubos

### Parámetros Clave:

- **`num_tubes`**: Número de ubicaciones espaciales muestreadas por imagen
  - Más tubos = más cobertura de la imagen
  - Default: 8

- **`scales`**: Tamaños de los crops antes de reescalar
  - `[64, 128, 256]` = pequeño, medio, grande
  - Todos se reescalan a `target_size`
  - Default: `[64, 128, 256]`

- **`target_size`**: Tamaño final de todos los patches
  - Default: 128

- **`num_views`**: Número de vistas por patch (1 original + N augmented)
  - Default: 2

### Augmentaciones Controladas:

Las degradaciones están diseñadas para preservar huellas forenses mientras añaden robustez:

- **JPEG/WebP compression**: `jpeg_prob=0.5`, quality 70-95
- **Resize variation**: `resize_prob=0.3`, ratio 0.8-1.2x
- **Gaussian blur**: `blur_prob=0.2`, sigma 0.5-2.0
- **Sharpen**: `sharpen_prob=0.2`, strength 0.5-2.0
- **Additive noise**: `noise_prob=0.3`, std 0.01-0.05

---

## 🐛 Troubleshooting

### ⚠️ Imágenes aparecen en negro

Este es un problema común relacionado con la normalización/desnormalización de ImageNet.

**Solución 1: Usar modo debug sin normalización**
```bash
./experiments/visualize_batch.sh --no-norm
```

Esto desactiva la normalización de ImageNet y muestra los tensores directamente en el rango [0, 1].

**Solución 2: Probar con script simplificado**
```bash
python scripts/test_no_norm.py
```

Este script crea visualizaciones sin normalización para verificar que el pipeline base funciona.

**Solución 3: Diagnosticar valores**
```bash
python scripts/debug_black_images.py
```

Imprime estadísticas detalladas de los tensores para identificar el problema específico.

**Causas comunes:**
- Los valores de `IMAGENET_MEAN` y `IMAGENET_STD` en `src/constants.py` son incorrectos
- Problema con el tipo de datos (dtype) de los tensores
- Problema con el dispositivo (CPU/GPU) de los tensores
- La desnormalización no se está aplicando correctamente

**Verificación:**
```bash
# Verifica los valores de normalización
python -c "from src.constants import IMAGENET_MEAN, IMAGENET_STD; print('Mean:', IMAGENET_MEAN); print('Std:', IMAGENET_STD)"
```

Los valores correctos deberían ser:
- Mean: (0.485, 0.456, 0.406)
- Std: (0.229, 0.224, 0.225)

### Error: "File not found"
```bash
# Verifica que existan los archivos
ls data/train.csv
head -n 5 data/train.csv
```

### Error: "Failed to load image"
```bash
# Verifica que las rutas en el CSV sean correctas
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').head())"
```

### Las visualizaciones se ven raras
- Verifica que `IMAGENET_MEAN` y `IMAGENET_STD` sean correctos en `src/constants.py`
- Revisa las probabilidades de augmentación (podrían ser muy altas)
- Prueba con `num_views=1` para ver los patches sin augmentación

---

## 💡 Tips

1. **Empezar simple**: Usa `quick_test_data.py` primero
2. **Pocos tubos para debug**: Usa `--num_tubes 4` para visualizar más rápido
3. **Sin augmentación**: Usa `--num_views 1` para ver los patches originales
4. **Batch pequeño**: Usa `--batch_size 2` para no generar demasiadas imágenes

---

## ✅ Checklist de Verificación

Antes de continuar con el entrenamiento, verifica:

- [ ] `quick_test_data.py` ejecuta sin errores
- [ ] Las formas (shapes) son correctas: `[B, N, K, V, C, P, P]`
- [ ] Los labels corresponden a las imágenes (real/synthetic)
- [ ] Las visualizaciones muestran progresión correcta de escalas
- [ ] Las augmentaciones son sutiles y no destruyen detalles
- [ ] Los centros de los tubos están bien distribuidos en la imagen
- [ ] La memoria por batch es razonable (<1GB)

Una vez verificado todo, puedes proceder con la implementación del modelo! 🚀
