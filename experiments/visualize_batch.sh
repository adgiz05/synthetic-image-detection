#!/bin/bash
# Script to visualize multi-scale tube batches
#
# This creates visualizations of the data loading pipeline:
# - Multi-scale tubes (different scales at same location)
# - Augmented views (different degradations)
# - Scale and view comparisons
#
# Usage:
#   ./experiments/visualize_batch.sh              # Normal mode (4 tubes, with normalization)
#   ./experiments/visualize_batch.sh --no-norm    # Debug mode (no normalization)
#   ./experiments/visualize_batch.sh --all        # Show ALL tubes (8 tubes per image)

cd "$(dirname "$0")/.."

# Create output directory
mkdir -p visualizations

echo "=========================================="
echo "Visualizing Multi-Scale Tube Batches"
echo "=========================================="
echo ""

# Parse arguments
MODE="normal"
NUM_IMAGES=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-norm)
            MODE="no-norm"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --num-images|-n)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage:"
            echo "  ./experiments/visualize_batch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-norm              Debug mode (no normalization)"
            echo "  --all                  Show ALL tubes (8 tubes per image)"
            echo "  --num-images N, -n N   Number of images to visualize (default: 4)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./experiments/visualize_batch.sh"
            echo "  ./experiments/visualize_batch.sh --all"
            echo "  ./experiments/visualize_batch.sh --num-images 8"
            echo "  ./experiments/visualize_batch.sh --all --num-images 6"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configure based on mode
if [[ "$MODE" == "no-norm" ]]; then
    echo "⚠️  Running in DEBUG mode (no normalization)"
    echo "📊 Visualizing $NUM_IMAGES images"
    echo ""
    
    python scripts/visualize_batch.py \
        --train_csv data/train.csv \
        --num_tubes 4 \
        --scales 64 128 256 \
        --target_size 128 \
        --num_views 1 \
        --batch_size $NUM_IMAGES \
        --num_samples $NUM_IMAGES \
        --num_tubes_per_sample 4 \
        --output_dir visualizations \
        --no-normalize

elif [[ "$MODE" == "all" ]]; then
    echo "📊 Running in ALL TUBES mode (showing all 8 tubes)"
    echo "📊 Visualizing $NUM_IMAGES images"
    echo ""
    
    python scripts/visualize_batch.py \
        --train_csv data/train.csv \
        --num_tubes 8 \
        --scales 64 128 256 \
        --target_size 128 \
        --num_views 2 \
        --batch_size $NUM_IMAGES \
        --num_samples $NUM_IMAGES \
        --num_tubes_per_sample 8 \
        --output_dir visualizations

else
    echo "Running in NORMAL mode (4 tubes, with ImageNet normalization)"
    echo "📊 Visualizing $NUM_IMAGES images"
    echo ""
    
    python scripts/visualize_batch.py \
        --train_csv data/train.csv \
        --num_tubes 8 \
        --scales 64 128 256 \
        --target_size 128 \
        --num_views 2 \
        --batch_size $NUM_IMAGES \
        --num_samples $NUM_IMAGES \
        --num_tubes_per_sample 4 \
        --output_dir visualizations
fi

echo ""
echo "=========================================="
echo "Done! Check the visualizations/ folder"
echo "=========================================="
echo ""

if [[ "$MODE" == "all" ]]; then
    echo "Note: Generated images for ALL 8 tubes per image ($NUM_IMAGES images total)"
    echo "This creates more files but shows complete coverage"
    echo ""
fi

echo "Generated files:"
ls -lh visualizations/*.png 2>/dev/null | tail -20

echo ""
echo "Tips:"
echo "  • To see ALL tubes: ./experiments/visualize_batch.sh --all"
echo "  • For debug mode: ./experiments/visualize_batch.sh --no-norm"
echo "  • Change number of images: ./experiments/visualize_batch.sh --num-images 8"
echo "  • Combine options: ./experiments/visualize_batch.sh --all --num-images 6"
echo ""
echo "If images appear black:"
echo "  1. Try: ./experiments/visualize_batch.sh --no-norm"
echo "  2. Run: python scripts/test_no_norm.py"
echo "  3. Check docs/VISUALIZACION_TUBOS.md for troubleshooting"
