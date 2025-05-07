## 📦 Installation Guide

### 🔧 Prerequisites

* Make sure **Python 3.10** is installed.
* Install **CUDA 12.4** on your system:
  👉 [CUDA Toolkit 12.4 Download](https://developer.nvidia.com/cuda-downloads)

### 🧠 Install PyTorch (with CUDA 12.4 support)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
--index-url https://download.pytorch.org/whl/cu124 \
--extra-index-url https://pypi.org/simple
```

### 📦 Install Other Dependencies

```bash
pip install -r requirements.txt
```

---

### 🧪 Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It should return `True` if CUDA is correctly installed and accessible by PyTorch.
