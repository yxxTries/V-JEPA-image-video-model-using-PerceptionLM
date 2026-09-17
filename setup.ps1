<#
  JEPA bench setup for Windows + NVIDIA RTX 50xx (Blackwell, compute capability 12.0).

  Run from this folder:
      powershell -ExecutionPolicy Bypass -File .\setup.ps1
  Options:
      -Cuda cu128        use CUDA 12.8 wheels instead (last torch there is 2.11; for drivers < 580)
      -Python 3.12       Python version uv should use (3.10-3.14 work; 3.12 is the safe default)
      -SkipFFmpeg        don't fetch FFmpeg (PyAV fallback will be used)
      -Recreate          delete .venv and start clean

  Creates .venv (uv), installs PyTorch CUDA wheels, transformers, nnsight, timm, einops,
  torchcodec (+ FFmpeg 8.1 shared DLLs in .\ffmpeg) and PyAV as fallback decoder.
  No model weights are downloaded here; each script fetches what it needs on first run.
#>
[CmdletBinding()]
param(
    [ValidateSet("cu130", "cu128", "cu129")] [string]$Cuda = "cu130",
    [string]$Python = "3.12",
    [switch]$SkipFFmpeg,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest ~10x faster
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }
function Invoke-Checked { param([scriptblock]$Cmd, [string]$What)
    & $Cmd
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- 1. GPU / driver
Step "Checking NVIDIA driver"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Warn "nvidia-smi not found. Install the latest NVIDIA Game Ready / Studio driver, then re-run."
} else {
    $info = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) | Select-Object -First 1
    Write-Host "    $info"
    try {
        $drv = [double](($info -split ",")[1].Trim() -replace "^(\d+\.\d+).*", '$1')
        $need = @{ "cu130" = 580; "cu129" = 575; "cu128" = 570 }[$Cuda]
        if ($drv -lt $need) {
            Warn "Driver $drv is older than $need required for $Cuda wheels. Update the driver or re-run with -Cuda cu128."
        }
    } catch { Warn "Could not parse driver version from: $info" }
}

# ---------------------------------------------------------------- 2. uv
Step "Checking uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "    installing uv (astral.sh)..."
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv install failed. Install manually: winget install astral-sh.uv" }
}
uv --version

# ---------------------------------------------------------------- 3. venv
Step "Creating .venv (Python $Python)"
if ($Recreate -and (Test-Path .venv)) { Remove-Item -Recurse -Force .venv }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Invoke-Checked { uv venv .venv --python $Python --seed } "uv venv"
}
$py = (Resolve-Path .venv\Scripts\python.exe).Path
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path

# ---------------------------------------------------------------- 4. PyTorch (CUDA)
Step "Installing PyTorch + torchvision + torchcodec from the $Cuda index"
$torchIndex = "https://download.pytorch.org/whl/$Cuda"
Invoke-Checked { uv pip install --python $py torch torchvision torchcodec --index-url $torchIndex } "PyTorch install"

# ---------------------------------------------------------------- 5. Everything else
Step "Installing transformers, nnsight, timm, einops, PyAV, Pillow"
Invoke-Checked { uv pip install --python $py "transformers>=4.56" nnsight timm einops av pillow numpy huggingface_hub } "library install"

# Guard: a dependency must not have swapped in a CPU-only torch from PyPI.
$cudaTag = & $py -c "import torch; print(torch.version.cuda or 'cpu')"
if ($cudaTag -eq "cpu") {
    Warn "A CPU-only torch was pulled in by a dependency; reinstalling the $Cuda build."
    Invoke-Checked { uv pip install --python $py --reinstall-package torch --reinstall-package torchvision --reinstall-package torchcodec torch torchvision torchcodec --index-url $torchIndex } "PyTorch reinstall"
}

# ---------------------------------------------------------------- 6. FFmpeg shared DLLs for torchcodec
if (-not $SkipFFmpeg) {
    Step "FFmpeg shared build for torchcodec (.\ffmpeg\bin)"
    $haveLocal = Test-Path ".\ffmpeg\bin\avcodec-*.dll"
    if ($haveLocal) {
        Write-Host "    already present"
    } else {
        $zipUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
        $zip = Join-Path $env:TEMP "ffmpeg-shared.zip"
        try {
            Write-Host "    downloading $zipUrl"
            Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing
            $tmp = Join-Path $env:TEMP "ffmpeg-shared-extract"
            if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
            Expand-Archive -Path $zip -DestinationPath $tmp -Force
            $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
            if (Test-Path .\ffmpeg) { Remove-Item -Recurse -Force .\ffmpeg }
            Move-Item $inner.FullName .\ffmpeg
            Remove-Item $zip -Force
            Write-Host "    installed to .\ffmpeg"
        } catch {
            Warn "FFmpeg download failed ($($_.Exception.Message))."
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                Write-Host "    trying: winget install Gyan.FFmpeg.Shared"
                winget install --id Gyan.FFmpeg.Shared -e --accept-source-agreements --accept-package-agreements
                Warn "If torchcodec still fails, set JEPA_FFMPEG_BIN to the winget FFmpeg 'bin' folder (see README)."
            } else {
                Warn "No winget. torchcodec will be skipped and PyAV used instead."
            }
        }
    }
}

# ---------------------------------------------------------------- 7. Verify
Step "Verifying CUDA + decoders"
$check = @'
import sys, time
import torch
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})")
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE -- see README: 'torch.cuda.is_available() is False'")
    sys.exit(2)
p = torch.cuda.get_device_properties(0)
print(f"GPU         {torch.cuda.get_device_name(0)}  |  {p.total_memory/2**30:.1f} GiB  |  compute capability {p.major}.{p.minor}")
arch = torch.cuda.get_arch_list()
print(f"arch list   {' '.join(arch)}")
if (p.major, p.minor) >= (12, 0) and not any(a.startswith("sm_12") for a in arch):
    print("This torch build has no Blackwell (sm_120) kernels -- reinstall with -Cuda cu128 or newer.")
    sys.exit(3)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t0 = time.perf_counter()
b = (a @ a).float().sum()
torch.cuda.synchronize()
print(f"CUDA tensor fp16 2048x2048 matmul OK on {b.device}: sum={b.item():.3e} in {(time.perf_counter()-t0)*1000:.1f} ms")
q = torch.randn(1, 16, 4608, 64, device="cuda", dtype=torch.float16)
torch.nn.functional.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
print("SDPA fp16 (ViT-L 16-frame @384 token count) OK")

import transformers, nnsight, timm, einops
print(f"transformers {transformers.__version__} | nnsight {nnsight.__version__} | timm {timm.__version__} | einops {einops.__version__}")
import jepa_common as jc
print(f"video decoder: {jc.video_backend()}")
try:
    import av; print(f"PyAV {av.__version__} (fallback) OK")
except Exception as e:
    print(f"PyAV missing: {e}")
'@
$checkFile = Join-Path $env:TEMP "jepa_setup_check.py"
Set-Content -Path $checkFile -Value $check -Encoding UTF8
$env:PYTHONPATH = $PSScriptRoot
& $py $checkFile
$code = $LASTEXITCODE
Remove-Item $checkFile -ErrorAction SilentlyContinue
if ($code -ne 0) { Fail "Verification failed (exit $code). See README troubleshooting." }

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then:      python extract.py clips\archery.mp4   (models download on first run)"
