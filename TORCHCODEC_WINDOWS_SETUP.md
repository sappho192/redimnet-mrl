# TorchCodec + FFmpeg Setup on Windows

This guide explains how to set up TorchCodec with FFmpeg shared libraries on Windows for audio/video processing.

## Problem

TorchCodec requires FFmpeg **shared libraries** (DLLs), but the standard `ffmpeg` package from Chocolatey only includes executables, not the shared libraries that torchcodec needs.

**Error you might see**:
```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment...
```

## Solution

### Step 1: Install FFmpeg Shared Libraries

Open **PowerShell or Command Prompt as Administrator** and run:

```powershell
choco install ffmpeg-shared -y
```

This installs FFmpeg version 8.0.1 with all the required DLL files:
- `avcodec-62.dll`
- `avdevice-62.dll`
- `avfilter-11.dll`
- `avformat-62.dll`
- `avutil-60.dll`
- `swresample-6.dll`
- `swscale-9.dll`

**Installation location**:
`C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin`

### Step 2: Configure Python Scripts

Add the FFmpeg DLL directory to your Python scripts **before importing torchcodec or torchaudio**:

```python
import sys
import os
from pathlib import Path

# Configure FFmpeg DLLs for torchcodec on Windows
if sys.platform == 'win32':
    ffmpeg_dll_paths = [
        r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin',
        r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-7.1.1-full_build-shared\bin',
    ]
    for dll_path in ffmpeg_dll_paths:
        if Path(dll_path).exists():
            os.add_dll_directory(dll_path)
            print(f"Added FFmpeg DLL directory: {dll_path}")
            break

# Now you can import torchaudio/torchcodec
import torchaudio
```

### Step 3: Test the Setup

```python
import sys
import os
from pathlib import Path

# Add DLL directory
if sys.platform == 'win32':
    dll_path = r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin'
    if Path(dll_path).exists():
        os.add_dll_directory(dll_path)

# Test torchcodec import
try:
    import torchcodec
    print("✅ TorchCodec loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load torchcodec: {e}")

# Test audio loading
try:
    import torchaudio
    waveform, sample_rate = torchaudio.load("path/to/your/audio.wav")
    print(f"✅ Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")
except Exception as e:
    print(f"❌ Failed to load audio: {e}")
```

## Alternative: Set System PATH (Permanent)

If you don't want to add the DLL directory in every script, you can add it to your system PATH:

1. Open **System Properties** → **Environment Variables**
2. Under **System variables**, find `Path` and click **Edit**
3. Click **New** and add:
   ```
   C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin
   ```
4. Click **OK** to save
5. **Restart your terminal/IDE** for changes to take effect

After this, torchcodec should work without needing `os.add_dll_directory()`.

## Verification

Check installed packages:
```bash
choco list ffmpeg
```

You should see:
```
ffmpeg 7.1.1              # Executables only
ffmpeg-shared 8.0.1       # Shared libraries (DLLs)
```

Check DLL files exist:
```powershell
dir "C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin\*.dll"
```

## Troubleshooting

### "Access Denied" during installation

Run PowerShell/CMD as **Administrator**:
- Right-click PowerShell/CMD
- Select "Run as administrator"
- Then run: `choco install ffmpeg-shared -y`

### Different FFmpeg version installed

If you have a different version (e.g., 7.1.1), adjust the path:
```python
dll_path = r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-7.1.1-full_build-shared\bin'
```

### Still getting errors

1. **Verify installation**:
   ```bash
   choco list ffmpeg-shared
   ```

2. **Check DLL directory exists**:
   ```python
   from pathlib import Path
   path = Path(r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin')
   print(f"Directory exists: {path.exists()}")
   print(f"DLL files: {list(path.glob('*.dll'))}")
   ```

3. **Reinstall** (as admin):
   ```bash
   choco uninstall ffmpeg-shared -y
   choco install ffmpeg-shared -y
   ```

## Why This Works

- **TorchCodec** is a PyTorch extension that uses FFmpeg libraries for efficient audio/video decoding
- On Windows, DLLs must be in:
  - The same directory as the executable
  - A directory in the system PATH
  - OR explicitly added via `os.add_dll_directory()`
- The `ffmpeg` package only includes `.exe` files
- The `ffmpeg-shared` package includes the necessary `.dll` files

## Alternative: Use soundfile

If you only need audio (not video) and don't want to deal with FFmpeg DLLs, you can use `soundfile` as a simpler alternative:

```bash
pip install soundfile
```

```python
import soundfile as sf
import torch

# Load audio
waveform, sample_rate = sf.read("audio.wav", dtype='float32')
waveform = torch.from_numpy(waveform).T  # Convert to torch tensor
```

However, torchcodec is recommended for:
- Better performance (C++ backend)
- Video support
- Better integration with PyTorch
- Direct use with torchaudio 2.9+

## References

- TorchCodec GitHub: https://github.com/pytorch/torchcodec
- FFmpeg Chocolatey: https://community.chocolatey.org/packages/ffmpeg-shared
- TorchAudio Documentation: https://pytorch.org/audio/stable/

---

**Status**: ✅ Working on Windows with `ffmpeg-shared` package
**Tested**: Windows 11, Python 3.12, PyTorch 2.9.1, TorchCodec 0.1.0
