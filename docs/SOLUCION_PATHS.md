# 🔧 Solución: Paths Duplicados (data/data/...)

Si ves errores como:
```
'data/data/images/ffhq/...' No such file or directory
```

El problema es una **duplicación de paths** causada por configuración incorrecta del `root_dir`.

## 🎯 Solución Rápida

### 1. Verifica tus paths primero

```bash
python scripts/check_paths.py data/train.csv
```

Esto te mostrará:
- Si los paths del CSV son accesibles
- Si hay duplicación de `data/`
- Qué valor de `root_dir` usar

### 2. Configuración correcta según tu CSV

#### Caso A: Paths en CSV son completos relativos al proyecto
```csv
image_path,label
data/images/ffhq/image1.png,0
data/images/imagenet/image2.jpg,1
```

**Solución**: NO uses `root_dir` o déjalo vacío
```python
dataset = MultiScaleTubeDataset(
    data_path="data/train.csv",
    root_dir=""  # Vacío!
)
```

O en scripts:
```bash
python scripts/visualize_batch.py --train_csv data/train.csv
# NO uses --root_dir
```

#### Caso B: Paths en CSV son relativos a un directorio
```csv
image_path,label
images/ffhq/image1.png,0
images/imagenet/image2.jpg,1
```

**Solución**: Usa `root_dir` para completar el path
```python
dataset = MultiScaleTubeDataset(
    data_path="data/train.csv",
    root_dir="data/"  # Completa el path
)
```

O en scripts:
```bash
python scripts/visualize_batch.py --train_csv data/train.csv --root_dir data/
```

#### Caso C: Paths en CSV son absolutos
```csv
image_path,label
/home/user/project/data/images/ffhq/image1.png,0
/home/user/project/data/images/imagenet/image2.jpg,1
```

**Solución**: NO uses `root_dir`
```python
dataset = MultiScaleTubeDataset(
    data_path="data/train.csv",
    root_dir=""  # Paths ya son absolutos
)
```

## 🔍 Diagnóstico Detallado

### Verificar paths con detalles

```bash
# Ver paths faltantes
python scripts/check_paths.py data/train.csv --show_missing

# Si crees que necesitas root_dir, prueba:
python scripts/check_paths.py data/train.csv --root_dir data/

# Ver más paths
python scripts/check_paths.py data/train.csv --max_check 500
```

### Entender el problema

El método `_resolve_path` ahora es inteligente y evita duplicación:

```python
# Si CSV tiene: "data/images/ffhq/img.png"
# Y usas root_dir="data/"
# Antes: "data/" + "data/images/ffhq/img.png" = "data/data/images/..." ❌
# Ahora: Detecta la duplicación y usa solo "data/images/ffhq/img.png" ✓
```

Pero es mejor **configurarlo correctamente desde el inicio**.

## 📋 Checklist

- [ ] Ejecuté `python scripts/check_paths.py data/train.csv`
- [ ] Entiendo cómo están formados los paths en mi CSV
- [ ] Configuré `root_dir` correctamente (o lo dejé vacío)
- [ ] Las imágenes se cargan correctamente (no hay "Using blank fallback")
- [ ] Las visualizaciones muestran imágenes reales, no negras

## 🛠️ Script de Verificación Automática

```bash
# Verifica todos los CSVs
for csv in data/*.csv; do
    echo "Checking $csv..."
    python scripts/check_paths.py "$csv"
    echo ""
done
```

## 💡 Recomendaciones

1. **Mantén paths consistentes**: Usa siempre el mismo estilo en todos tus CSVs
2. **Prefiere paths relativos al proyecto**: `data/images/...` en vez de absolutos
3. **Documenta tu estructura**: Añade un README explicando dónde están las imágenes
4. **Verifica antes de entrenar**: Usa `check_paths.py` antes de lanzar entrenamientos largos

## 📚 Más Información

- [docs/VISUALIZACION_TUBOS.md](../docs/VISUALIZACION_TUBOS.md) - Guía completa de visualización
- [docs/SOLUCION_IMAGENES_NEGRAS.md](../docs/SOLUCION_IMAGENES_NEGRAS.md) - Si las imágenes aparecen negras
