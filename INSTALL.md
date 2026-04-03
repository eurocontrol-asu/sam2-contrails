## Installation

### Requirements

- Linux with Python >= 3.10, PyTorch >= 2.5.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at https://pytorch.org to ensure this.
- [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation.
- If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

### Recommended: install with uv

```bash
git clone https://github.com/ramondalmau/sam2-contrails.git
cd sam2-contrails

# Inference only
uv sync --extra inference

# Full install (adds prompt generation from ADS-B + ERA5 data)
uv sync --extra all
```

The repo includes a `uv.lock` file for fully reproducible installs. PyTorch is pre-configured for **CUDA 12.8** via `[tool.uv.sources]` in `pyproject.toml`. If your system uses a different CUDA version, edit the index name (e.g. `cu128` -> `cu121`) before running `uv sync`. For CPU-only, remove the `[tool.uv.sources]` block.

### Alternative: install with pip

```bash
pip install -e ".[inference]"   # inference only
pip install -e ".[all]"         # full install
```

### Building the SAM 2 CUDA extension

By default, the installation proceeds even if the SAM 2 CUDA extension fails to build. (In this case, the build errors are hidden unless using `-v` for verbose output in `pip install`.)

If you see a message like `Skipping the post-processing step due to the error above` at runtime, it indicates that the CUDA extension failed to build. **You can still use SAM2-Contrails for both image and video applications.** The post-processing step (removing small holes and sprinkles in the output masks) will be skipped, but this should not affect results in most cases.

To force-build the CUDA extension:

```bash
pip uninstall -y sam2-contrails && \
rm -f ./sam2/*.so && \
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[inference]"
```

Note that PyTorch needs to be installed first before building the CUDA extension. It is also necessary to install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. After installing the CUDA toolkits, you can check its version via `nvcc --version`.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
I got <code>ImportError: cannot import name '_C' from 'sam2'</code>
</summary>
<br/>

This usually happens because the package is not installed or the installation failed. Please install the package first. In some systems, you may need to run `python setup.py build_ext --inplace` in the repo root.
</details>

<details>
<summary>
I got <code>MissingConfigException: Cannot find primary config ...</code>
</summary>
<br/>

This usually happens because `sam2` is not in your Python's `sys.path`. Please run the installation step. If it still fails, you may try manually adding the repo root to `PYTHONPATH`:
```bash
export SAM2_REPO_ROOT=/path/to/sam2-contrails
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```
</details>

<details>
<summary>
My installation failed with <code>CUDA_HOME environment variable is not set</code>
</summary>
<br/>

This usually happens because the CUDA toolkits (containing the NVCC compiler) are not found. Please install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) matching your PyTorch's CUDA version. If the error persists, explicitly set `CUDA_HOME`:
```bash
export CUDA_HOME=/usr/local/cuda  # change to your CUDA toolkit path
```

Verify the setup with:
```bash
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
This should print `(True, a directory with cuda)`.
</details>

<details>
<summary>
I got <code>RuntimeError: No available kernel. Aborting execution.</code>
</summary>
<br/>

This is probably because your machine does not have a GPU or a compatible PyTorch version for Flash Attention. You may resolve this by replacing the line:
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
```
in `sam2/modeling/sam/transformer.py` with:
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
```
to relax the attention kernel setting and use other kernels than Flash Attention.
</details>
