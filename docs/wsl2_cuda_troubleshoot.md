# WSL2 CUDA troubleshooting guide

This guide covers common failures when running CUDA workloads under WSL2, including setup.sh failures, DeepSpeed compilation issues, and training hangs. Each section describes the symptom, root cause, fix, and verification command.

## Quick checks

Run these commands in order to diagnose which layer is broken. If any command fails, jump to the corresponding section below.

```bash
# 1. Is the NVIDIA driver visible from WSL2?
nvidia-smi

# 2. Is the CUDA toolkit installed and in PATH?
nvcc --version

# 3. Does PyTorch see CUDA?
python -c "import torch; print(torch.cuda.is_available())"

# 4. Are DeepSpeed ops compiled?
ds_report
```

If `nvidia-smi` fails, the issue is at the Windows driver level (reinstall the NVIDIA driver from Windows). If `nvcc` fails, see the "CUDA toolkit not in PATH" section. If PyTorch returns `False`, see the "PyTorch CUDA version mismatch" section. If `ds_report` shows `cpu_adam: [NO]`, see the "DeepSpeedCPUAdam fell back to Python implementation" section.

## Common issues

The sections below cover the five most frequent failures encountered when setting up a CUDA training environment under WSL2.

### WSL2 RAM limit too low

**Symptom:** OOM errors during DeepSpeed initialization or training, `free -h` shows far less RAM than physically installed.

**Cause:** The `.wslconfig` file on the Windows side limits how much system RAM WSL2 can access. The default is 50% of system RAM, but an explicit low value (such as `memory=8GB`) overrides this. DeepSpeed CPU offloading needs substantial system RAM for optimizer states (approximately 8-12GB for a 500M parameter model), so an 8GB limit will cause immediate OOM.

**Fix:** Edit `C:\Users\<username>\.wslconfig` (create the file if it does not exist) and set the memory limit to your system RAM minus approximately 6GB for Windows. For a 64GB system, use the following configuration.

```ini
[wsl2]
memory=58GB
```

After saving the file, shut down WSL2 from PowerShell and restart it.

```powershell
wsl --shutdown
```

**Verify:** Open a new WSL2 terminal and check available memory.

```bash
free -h
```

The total should reflect the value you set in `.wslconfig`.

### CUDA toolkit not in PATH

**Symptom:** `which nvcc` returns nothing. `ds_report` shows DeepSpeed ops as "not buildable" because it cannot find the CUDA compiler.

**Cause:** The CUDA toolkit is installed at `/usr/local/cuda/` but the `bin` directory is not in the Linux PATH. DeepSpeed's JIT compilation of C++ extensions (including CPUAdam) requires `nvcc` to be accessible.

**Fix:** Add the following lines to your `~/.bashrc` file.

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

Then reload your shell configuration.

```bash
source ~/.bashrc
```

**Verify:** Confirm that `nvcc` is now accessible.

```bash
which nvcc
# Expected: /usr/local/cuda/bin/nvcc
```

### Windows TDR timeout causes CUDA hang

**Symptom:** Training hangs after a variable number of steps. GPU utilization drops to 0%. The process becomes unresponsive and cannot be interrupted with Ctrl+C. This is intermittent and may not appear during short runs.

**Cause:** Windows Timeout Detection and Recovery (TDR) monitors GPU operations and resets the GPU if a single operation exceeds approximately 2 seconds of continuous compute. Long CUDA kernels (such as large matrix multiplications or gradient accumulation across many microbatches) can exceed this threshold, triggering a GPU reset that manifests as a silent hang in WSL2.

**Fix:** Increase the TDR timeout by editing the Windows Registry. Open Registry Editor (`regedit`) as Administrator and navigate to the following key.

```
HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
```

Create a new DWORD (32-bit) value named `TdrDelay` and set its value to `60` (decimal). This gives the GPU 60 seconds per operation instead of the default 2 seconds. Reboot Windows for the change to take effect.

**Verify:** Open Registry Editor and confirm the `TdrDelay` value exists under the `GraphicsDrivers` key with a value of 60.

### DeepSpeedCPUAdam fell back to Python implementation

**Symptom:** `ds_report` shows `cpu_adam: [NO]` instead of `[YES]`. Training step times are 5-7x slower than expected because the optimizer is using the pure Python fallback instead of the compiled C++ extension.

**Cause:** The CPUAdam C++ extension failed to compile during DeepSpeed installation. Common reasons include `nvcc` not in PATH, incompatible gcc version (gcc 15+ may fail with CUDA 12.8), or missing Python development headers.

**Fix:** Ensure prerequisites are met, then force-reinstall DeepSpeed with the CPUAdam build flag.

```bash
# Verify prerequisites
which nvcc          # Must return a path
gcc --version       # Must be <= 14 for CUDA 12.8
python3-config --includes  # Must return include paths

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Force rebuild
DS_BUILD_CPU_ADAM=1 pip install --force-reinstall deepspeed==0.18.6
```

**Verify:** Run `ds_report` and confirm the CPUAdam status.

```bash
ds_report 2>&1 | grep cpu_adam
# Expected: cpu_adam .............. [YES] ...... [OKAY]
```

### PyTorch CUDA version mismatch

**Symptom:** DeepSpeed JIT compilation fails with cryptic C++ errors. `ds_report` shows ops as "incompatible". Training crashes with CUDA-related exceptions despite `nvidia-smi` working correctly.

**Cause:** PyTorch was installed from the wrong index. The default `pip install torch` installs the CPU-only build. Alternatively, a build targeting a different CUDA version (such as cu121) was installed on a system with CUDA 12.8.

**Fix:** Reinstall PyTorch from the correct CUDA index.

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

**Verify:** Check the CUDA version reported by PyTorch.

```bash
python -c "import torch; print(torch.version.cuda)"
# Expected: 12.8
```

## Nuclear option

If none of the targeted fixes above resolve the issue, perform a clean environment rebuild. This removes the virtual environment entirely and recreates it from scratch.

```bash
# From the project root
deactivate 2>/dev/null
rm -rf .venv
bash setup.sh
```

The setup script will recreate the virtual environment, reinstall all dependencies with the correct CUDA index, pre-build the DeepSpeed CPUAdam extension, and run the full validation suite including `ds_report`. If setup.sh itself fails, check the prerequisites section at the top of the script output for specific error messages about missing system packages.
