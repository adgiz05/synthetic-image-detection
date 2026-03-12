# ⚠️ Solución: Imágenes Negras en Visualización

Si las visualizaciones aparecen completamente en negro, sigue estos pasos:

## 🔧 Solución Rápida

### 1. Usar modo debug (SIN normalización)

```bash
./experiments/visualize_batch.sh --no-norm
```

Esto desactiva la normalización de ImageNet y muestra las imágenes directamente.

### 2. Verificar con script simplificado

```bash
python scripts/test_no_norm.py
```

Genera `test_no_normalization.png` - Si esta imagen se ve correcta, el problema está en la normalización/desnormalización.

## 🔍 Diagnóstico Detallado

### Opción A: Ver estadísticas de tensores

```bash
python scripts/debug_black_images.py
```

Muestra:
- Valores min/max/mean de tensores normalizados
- Valores después de desnormalización
- Si los valores están en el rango correcto [0, 1]
- Genera `debug_tensor_values.png` con 3 versiones de la imagen

### Opción B: Verificar constantes

```bash
python -c "from src.constants import IMAGENET_MEAN, IMAGENET_STD; print('Mean:', IMAGENET_MEAN, '\\nStd:', IMAGENET_STD)"
```

**Valores correctos:**
- Mean: `(0.485, 0.456, 0.406)`
- Std: `(0.229, 0.224, 0.225)`

Si son diferentes, corríge los en `src/constants.py`.

## 🛠️ Soluciones por Causa

### Causa 1: Normalización incorrecta

**Síntoma:** Imágenes negras incluso sin --no-norm

**Solución:**
```python
# Verifica src/constants.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

### Causa 2: Problema con denormalización

**Síntoma:** Modo --no-norm funciona, pero modo normal no

**Solución:** Ya está corregida en la versión actual de `scripts/visualize_batch.py`. Asegúrate de tener la última versión:

```python
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    tensor = tensor.clone()  # Evita modificar el original
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    # ... resto del código
```

### Causa 3: Entorno Python incorrecto

**Síntoma:** Error `ModuleNotFoundError: No module named 'torch'`

**Solución:**
```bash
# Activar entorno correcto
conda activate synthetic-generation
# O reinstalar dependencias
pip install -r requirements.txt
```

## ✅ Verificación

Una vez solucionado, deberías ver:

1. **Con --no-norm**: Imágenes claras y correctas
2. **Sin --no-norm**: Imágenes idénticas (con normalización funcionando)
3. **debug_tensor_values.png**: Las 3 versiones se ven similares

## 📚 Documentación Completa

Para más detalles, ver: [docs/VISUALIZACION_TUBOS.md](../docs/VISUALIZACION_TUBOS.md)
