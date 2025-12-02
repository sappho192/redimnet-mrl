#!/bin/bash
# Quick Start Script for MRL Training
# This script helps you set up data and start training

set -e  # Exit on error

echo "======================================"
echo "MRL-ReDimNet Quick Start"
echo "======================================"
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-/data/voxceleb}"
DOWNLOAD="${DOWNLOAD:-false}"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "⚠️  Data directory does not exist: $DATA_DIR"
    echo ""
    read -p "Do you want to download VoxCeleb2? This will take ~50GB and several hours. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD=true
        mkdir -p "$DATA_DIR"
    else
        echo "❌ Cannot proceed without data. Exiting."
        exit 1
    fi
fi

# Download VoxCeleb2 if requested
if [ "$DOWNLOAD" = true ]; then
    echo ""
    echo "📥 Downloading VoxCeleb2..."
    echo "This will take several hours depending on your connection."
    echo ""
    
    cd "$DATA_DIR"
    
    # VoxCeleb2 dev set
    if [ ! -f "vox2_dev_aac.zip" ]; then
        wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac.zip
    fi
    
    echo "📦 Extracting..."
    unzip -q vox2_dev_aac.zip
    
    echo "✅ VoxCeleb2 downloaded successfully"
fi

# Verify data
echo ""
echo "🔍 Verifying data..."

if [ ! -d "$DATA_DIR/voxceleb2/dev/aac" ]; then
    echo "❌ VoxCeleb2 data not found at: $DATA_DIR/voxceleb2/dev/aac"
    exit 1
fi

NUM_SPEAKERS=$(find "$DATA_DIR/voxceleb2/dev/aac" -maxdepth 1 -type d | wc -l)
echo "✅ Found $NUM_SPEAKERS speaker directories"

# Check uv is installed
echo ""
echo "🔍 Checking uv installation..."

if ! command -v uv &> /dev/null; then
    echo "❌ uv not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv installed"

# Sync dependencies
echo ""
echo "📦 Syncing dependencies with uv..."
uv sync || {
    echo "❌ Failed to sync dependencies"
    exit 1
}

echo "✅ Dependencies synced"

# Check GPU
echo ""
echo "🔍 Checking GPU availability..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Update config
echo ""
echo "📝 Updating configuration..."

CONFIG_FILE="$(dirname "$0")/config.yaml"

# Backup original config
if [ ! -f "${CONFIG_FILE}.backup" ]; then
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
    echo "✅ Backed up original config to ${CONFIG_FILE}.backup"
fi

# Update paths in config
sed -i.tmp "s|train_dataset: .*|train_dataset: '$DATA_DIR/voxceleb2/dev/aac'|" "$CONFIG_FILE"
sed -i.tmp "s|val_dataset: .*|val_dataset: '$DATA_DIR/voxceleb2/dev/aac'|" "$CONFIG_FILE"
rm "${CONFIG_FILE}.tmp"

echo "✅ Config updated with data paths"

# Test data loading
echo ""
echo "🧪 Testing data loading..."

uv run python << PYEOF
import sys
sys.path.insert(0, '$(dirname "$0")')
from dataset import VoxCelebDataset

try:
    dataset = VoxCelebDataset(
        data_dir='$DATA_DIR/voxceleb2/dev/aac',
        chunk_duration=3.0,
        augmentation=False
    )
    print(f"✅ Dataset loaded: {len(dataset)} utterances")

    # Try loading one sample
    waveform, label = dataset[0]
    print(f"✅ Sample shape: {waveform.shape}")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Data loading test failed"
    exit 1
fi

# Ready to train
echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "You can now start training with:"
echo ""
echo "  cd $(dirname "$0")"
echo "  uv run python train.py --config config.yaml"
echo ""
echo "Training will take approximately:"
echo "  - Stage 1 (5 epochs): ~1 day"
echo "  - Stage 2 (50 epochs): ~1 week"
echo ""
echo "Monitor progress with:"
echo "  tensorboard --logdir logs/mrl_redimnet"
echo ""
echo "Press any key to continue..."
read -n 1 -s
