# Forward-compat CUDA driver, sourced by the sweep scripts. Not executable on its own.
#
# The wheels in these venvs are built against a CUDA major that the host driver is often older
# than -- RunPod A100 hosts especially, and it is not fixable from inside the container because the
# driver comes from the host, not the image. Left alone it fails deep inside
# `torch.cuda._lazy_init` with a message that names neither the torch build it is being compared
# against nor any fix ("The NVIDIA driver on your system is too old (found version 12040)").
#
# Datacenter GPUs can run a newer *user-mode* driver over an older kernel driver, which is what
# `cuda-compat-<major>-<minor>` installs. The dynamic loader reads LD_LIBRARY_PATH at exec, so this
# has to be exported by the launching shell -- run_engine.py cannot set it for itself, which is why
# `interp_engine.cuda_preflight` can only diagnose this and not repair it.
#
# Set CUDA_COMPAT=0 to never touch the driver.

# True when CUDA version $1 is at least $2 (both "MAJOR.MINOR").
cuda_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, x, "."); split(b, y, ".");
    exit !(x[1] > y[1] || (x[1] == y[1] && x[2] >= y[2]))
  }'
}

# The CUDA version a venv's torch was built for, e.g. "13.0". Empty if the venv or a CUDA-enabled
# torch is missing. Asked per venv on purpose: the three comparison venvs pin different torch
# builds, so there is no single answer for the box.
cuda_build_version() {  # <venv-python>
  [ -x "$1" ] || return 0
  "$1" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null
}

# Put a forward-compat driver on LD_LIBRARY_PATH if, and only if, the host driver is too old for
# the given venv's torch build. Safe and quiet when the driver is already new enough.
setup_cuda_compat() {  # <venv-python>
  local need have dir major minor distro pkg
  [ "${CUDA_COMPAT:-1}" = 1 ] || return 0

  need=$(cuda_build_version "$1")
  [ -n "$need" ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[cuda] no nvidia-smi; leaving the driver alone"; return 0; }

  # Both bail-outs are deliberate: the compat driver refuses to load over a kernel driver that is
  # already newer than it, so activating it unconditionally would break the hosts that are fine.
  have=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  if [ -z "$have" ]; then
    echo "[cuda] could not read the host CUDA version from nvidia-smi; leaving the driver alone"
    return 0
  fi
  if cuda_ge "$have" "$need"; then
    echo "[cuda] host driver supports CUDA $have (>= $need); no compat driver needed"
    return 0
  fi

  major=${need%%.*}; minor=${need#*.}
  pkg="cuda-compat-${major}-${minor}"
  echo "[cuda] host driver supports CUDA $have, wheels need CUDA $need; installing $pkg"

  # Any compat of the same CUDA major will do -- compatibility is guaranteed across minors -- so
  # only install when nothing usable is already present, and then take the newest one found.
  dir=$(ls -d /usr/local/cuda-"$major".*/compat 2>/dev/null | sort -V | tail -n1)
  if [ -z "$dir" ]; then
    if ! apt-get install -y "$pkg" >/dev/null 2>&1; then
      echo "[cuda] $pkg is not in this image's apt sources; adding the NVIDIA CUDA repo"
      distro="ubuntu$(. /etc/os-release 2>/dev/null && echo "$VERSION_ID" | tr -d .)"
      curl -fsSL -o /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$(uname -m)/cuda-keyring_1.1-1_all.deb" \
        && dpkg -i /tmp/cuda-keyring.deb >/dev/null \
        && apt-get update -y >/dev/null \
        && apt-get install -y "$pkg" >/dev/null
    fi
    dir=$(ls -d /usr/local/cuda-"$major".*/compat 2>/dev/null | sort -V | tail -n1)
  fi

  if [ -z "$dir" ]; then
    echo "[cuda] WARNING: no CUDA $need compat driver available -- every engine will fail to reach the GPU"
    return 0
  fi

  export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[cuda] forward-compat driver at $dir is on LD_LIBRARY_PATH"
}
