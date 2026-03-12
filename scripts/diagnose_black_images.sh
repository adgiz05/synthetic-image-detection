#!/bin/bash
# Diagnóstico completo del problema de imágenes negras
#
# Este script ejecuta múltiples pruebas para identificar el problema

cd "$(dirname "$0")/.."

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  DIAGNÓSTICO: Problema de Imágenes Negras             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. Verificar constantes
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Verificando constantes de normalización..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
from src.constants import IMAGENET_MEAN, IMAGENET_STD
print('IMAGENET_MEAN:', IMAGENET_MEAN)
print('IMAGENET_STD:', IMAGENET_STD)
print()
if IMAGENET_MEAN == (0.485, 0.456, 0.406) and IMAGENET_STD == (0.229, 0.224, 0.225):
    print('✓ Constantes son correctas')
else:
    print('✗ ADVERTENCIA: Constantes son incorrectas!')
    print('  Deberían ser:')
    print('  IMAGENET_MEAN = (0.485, 0.456, 0.406)')
    print('  IMAGENET_STD = (0.229, 0.224, 0.225)')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "✗ Error al importar constantes"
    echo "  Asegúrate de tener las dependencias instaladas"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Probando pipeline SIN normalización..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/test_no_norm.py 2>&1 | head -50

if [ $? -eq 0 ] && [ -f "test_no_normalization.png" ]; then
    echo ""
    echo "✓ Generado: test_no_normalization.png"
    echo "  → Abre este archivo para ver si las imágenes se ven correctas"
else
    echo ""
    echo "✗ Error en el test sin normalización"
    echo "  Esto sugiere un problema más fundamental en el pipeline"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Generando visualizaciones CON normalización..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./experiments/visualize_batch.sh 2>&1 | tail -20

if [ -d "visualizations" ] && [ "$(ls -A visualizations/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ""
    echo "✓ Visualizaciones generadas en: visualizations/"
    echo "  → Revisa estas imágenes"
else
    echo ""
    echo "✗ No se generaron visualizaciones"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Generando visualizaciones SIN normalización (debug)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./experiments/visualize_batch.sh --no-norm 2>&1 | tail -20

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  RESUMEN                                               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Archivos generados:"
echo "  • test_no_normalization.png       (sin normalización básica)"
echo "  • visualizations/*.png            (con normalización)"
echo ""
echo "Compara las imágenes:"
echo ""
echo "  Si test_no_normalization.png se ve BIEN:"
echo "    → El pipeline base funciona correctamente"
echo "    → El problema está en la normalización/desnormalización"
echo ""
echo "  Si visualizations/ se ve NEGRO pero test_no_normalization.png BIEN:"
echo "    → Problema confirmado en normalización"
echo "    → Revisa src/constants.py y scripts/visualize_batch.py"
echo ""
echo "  Si AMBOS se ven NEGRO:"
echo "    → Problema más fundamental (paths de imágenes, etc.)"
echo "    → Revisa que data/train.csv tiene paths correctos"
echo ""
echo "Para más ayuda: docs/SOLUCION_IMAGENES_NEGRAS.md"
echo ""
