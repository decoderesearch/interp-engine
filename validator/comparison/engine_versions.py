"""What each engine's capture was actually produced by: package version, and commit when it is a checkout.

A cell in the comparison table is a claim about a *version* of an engine, and "SGLang disagrees on
gemma-2-27b" is only reproducible — or refutable, once it's fixed — if the reader knows which SGLang.
Every bug we have filed upstream had to state this by hand, and the gemma-2 one needed `flashinfer`'s
version too, because that is the layer the defect turned out to live in. So each engine records its own
stack at capture time.

It has to happen *in the engine's process*: the sweep runs three virtualenvs, and the aggregator (in
`.venv-cmp`) cannot see vLLM's or SGLang's site-packages at all. :func:`engine_versions` is therefore
called by ``run_engine`` and stored in the capture meta, which aggregation copies into the detail JSON.

Resolution is metadata-only — no imports of vllm/torch/etc. just to read a number, since importing an
engine's deps out of order is how you get a CUDA context in the wrong process.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys

# Per engine, the packages whose version changes what a capture *is*: the engine itself, the loader in
# front of it, and the kernel libraries under it. `torch` is on every row because it moves numerics for
# all of them. `flashinfer`/`sgl_kernel` are on SGLang's because the attention backend is where a
# divergence we chased actually lived (sgl-project/sglang#33915), and the SGLang version alone would not
# have said which kernel library the capture ran against.
ENGINE_PACKAGES: dict[str, tuple[str, ...]] = {
    "eager": ("interp_engine", "transformers", "torch"),
    "vllm": ("interp_engine", "vllm", "transformers", "torch"),
    "vllm-static": ("interp_engine", "vllm", "transformers", "torch"),
    "tlens_v2": ("transformer_lens", "transformers", "torch"),
    "tlens_v3": ("transformer_lens", "transformers", "torch"),
    "nnsight": ("nnsight", "nnterp", "transformers", "torch"),
    "sglang": ("sglang", "sgl_kernel", "flashinfer", "torch"),
}

# setuptools-scm writes the commit into the local version segment of a dev build, which is how a
# checkout identifies itself with no git call: "0.5.17.dev721+g18e6c61c2" -> "18e6c61c2".
_SCM_LOCAL_COMMIT = re.compile(r"\+(?:.*\.)?g([0-9a-f]{7,40})")


def _dist_names(module: str) -> list[str]:
    """Distribution names that could provide ``module``, best guess first. Import names and dist names
    diverge often enough to matter here (`flashinfer` ships as `flashinfer-python`, `sgl_kernel` as
    `sglang-kernel`), so ask the installed metadata rather than guessing from the module name."""
    try:
        provided = list(importlib.metadata.packages_distributions().get(module, ()))
    except Exception:  # noqa: BLE001 - a broken dist listing shouldn't cost us the version
        provided = []
    for candidate in (module, module.replace("_", "-")):
        if candidate not in provided:
            provided.append(candidate)
    return provided


def _version_from_metadata(module: str) -> tuple[str, str]:
    """``(version, dist_name)`` for ``module``, or ``("", "")``."""
    for dist in _dist_names(module):
        try:
            return importlib.metadata.version(dist), dist
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            continue
    return "", ""


def _module_dir(module: str) -> str:
    """Where ``module`` lives on disk, without executing it (``find_spec`` imports parent packages
    only, and these are all top-level). Empty when it isn't installed."""
    try:
        spec = importlib.util.find_spec(module)
    except Exception:  # noqa: BLE001 - a module that raises on lookup has no directory to report
        return ""
    if spec is None:
        return ""
    if spec.submodule_search_locations:
        return next(iter(spec.submodule_search_locations), "")
    return os.path.dirname(spec.origin or "")


def _is_checkout(directory: str) -> bool:
    """Whether ``directory`` is source we could ask git about, rather than an installed copy.

    An installed package must never be asked: these virtualenvs live *inside* this repo
    (``.venv-cmp/lib/python3.11/site-packages/torch``), so `git rev-parse` from a wheel's directory
    happily returns **interp-engine's** HEAD and would stamp every third-party package with our commit —
    a plausible-looking hash pointing at the wrong project. For an installed package the metadata
    version is the whole truth; git only has something to add for a checkout (an editable install, or a
    source tree on ``PYTHONPATH``, which is how we test an engine's `main`).
    """
    if not directory:
        return False
    parts = set(os.path.normpath(directory).split(os.sep))
    return not parts & {"site-packages", "dist-packages"}


def _git(directory: str, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001 - no git binary, or it hung
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _commit_from_direct_url(dist: str) -> str:
    """The commit pip recorded for a VCS install (PEP 610 ``direct_url.json``)."""
    try:
        files = importlib.metadata.distribution(dist).files or ()
    except Exception:  # noqa: BLE001
        return ""
    for path in files:
        if path.name != "direct_url.json":
            continue
        try:
            with open(str(path.locate())) as f:
                return str(json.load(f).get("vcs_info", {}).get("commit_id") or "")
        except Exception:  # noqa: BLE001
            return ""
    return ""


def package_version(module: str) -> dict:
    """Version + provenance for one package, as small as the truth allows.

    Keys are omitted rather than left empty, so a plain wheel install records just ``{"version": ...}``
    and only a checkout carries ``commit``/``dirty``. ``dirty`` is the point of including git at all: an
    editable install with uncommitted edits is *not* the commit it reports, and a bug write-up that
    claims otherwise sends a maintainer chasing code we never ran.
    """
    version, dist = _version_from_metadata(module)
    if not version:
        # Already-imported modules can still answer (a source checkout on PYTHONPATH has no metadata).
        version = str(getattr(sys.modules.get(module), "__version__", "") or "")
    info: dict = {}
    if version:
        info["version"] = version

    commit = ""
    scm = _SCM_LOCAL_COMMIT.search(version or "")
    if scm:
        commit = scm.group(1)
    if not commit and dist:
        commit = _commit_from_direct_url(dist)
    if commit:
        info["commit"] = commit
        return info

    directory = _module_dir(module)
    if _is_checkout(directory) and _git(directory, "rev-parse", "--is-inside-work-tree") == "true":
        commit = _git(directory, "rev-parse", "HEAD")
        if commit:
            info["commit"] = commit
            # Scoped to the package, because `git status` is repo-wide wherever it is run from and
            # the validator ships in the engine's own repo: unscoped, editing a scoring test or the
            # visualizer marks an engine capture dirty, and a flag that fires on unrelated work is
            # one a reader learns to ignore. The commit stays the repo's, which is the commit
            # someone would check out.
            if _git(directory, "status", "--porcelain", "--", directory):
                info["dirty"] = True
    return info


# The one package a column's version *means*: which vLLM, which TransformerLens. Every column also
# depends on torch and transformers, but a header can carry one number, and the rest of the stack is in
# the cell's JSON. Both interp-engine columns report interp-engine itself for `eager` and vLLM for `vllm`,
# since "which vLLM did this agree with" is the question that column answers.
PRIMARY_PACKAGE = {
    "eager": "interp_engine",
    "vllm": "vllm",
    "vllm-static": "vllm",
    "tlens_v2": "transformer_lens",
    "tlens_v3": "transformer_lens",
    "nnsight": "nnsight",
    "sglang": "sglang",
}

# Where a version links to. Only for resolving a version string to a commit/tag URL.
REPOS = {
    # interp_engine is deliberately absent, which yields no link rather than a wrong one. The engine
    # is `decoderesearch/interp-engine`, this repo, and every capture here runs the editable checkout
    # one directory up -- so `commit` above is already exact and a tag link would add nothing. Adding
    # a row here would also need the tag convention settled first: releases were previously prefixed
    # (`interp-engine-v1.0.1`) to share a namespace with the monorepo, and the bare `v<version>` label
    # this module builds would resolve to nothing.
    "vllm": "vllm-project/vllm",
    "transformer_lens": "TransformerLensOrg/TransformerLens",
    "nnsight": "ndif-team/nnsight",
    "sglang": "sgl-project/sglang",
    "flashinfer": "flashinfer-ai/flashinfer",
}

# Resolved release-tag -> commit lookups, so the "exact commit" link works for a wheel install too. Kept
# on disk (and in git) because a version's commit never changes: one network call per version, ever.
_RELEASES_CACHE = os.path.join(os.path.dirname(__file__), "engine_releases.json")


def _version_label(version: str) -> str:
    """``v`` + the version, minus the local segment (the commit is in the link, not the label):
    ``0.5.17.dev721+g18e6c61c2`` -> ``v0.5.17.dev721``."""
    return "v" + version.split("+", 1)[0]


def _load_releases() -> dict:
    try:
        with open(_RELEASES_CACHE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - a missing or corrupt cache just means we look it up again
        return {}


def _resolve_release_commit(repo: str, version: str) -> str:
    """The commit a release tag points at, via the GitHub API, cached on disk.

    A release tag *is* an exact commit, so this is how a wheel install still gets a commit link. Network
    failures are not errors here: the caller falls back to a link that needs no lookup.
    """
    key = f"{repo}@{version}"
    cache = _load_releases()
    if key in cache:
        return str(cache[key] or "")

    commit = ""
    for tag in (f"v{version}", version):
        try:
            import urllib.request

            url = f"https://api.github.com/repos/{repo}/commits/{tag}"
            with urllib.request.urlopen(url, timeout=8) as resp:  # noqa: S310 - fixed https host
                commit = str(json.load(resp).get("sha") or "")
        except Exception:  # noqa: BLE001 - offline, rate-limited, or no such tag
            continue
        if commit:
            break
    else:
        return ""  # never resolved and never cached, so a later run with network can try again

    cache[key] = commit
    try:
        with open(_RELEASES_CACHE, "w") as f:
            json.dump(dict(sorted(cache.items())), f, indent=2)
            f.write("\n")
    except Exception:  # noqa: BLE001 - read-only checkout: the lookup still worked for this run
        pass
    return commit


def engine_release(engine: str, versions: dict, *, resolve: bool = True) -> dict:
    """``{"version", "label", "url"}`` for the engine behind a cell, from its recorded ``versions``.

    ``url`` prefers the commit we actually ran (a checkout records one; a release tag resolves to one),
    and falls back to the project's releases page, which always exists — a wrong-looking link in a table
    of correctness claims is worse than a general one. ``{}`` when the cell recorded no version, which is
    every dump captured before versions were recorded.
    """
    package = PRIMARY_PACKAGE.get(engine, "")
    info = (versions or {}).get(package) or {}
    version = str(info.get("version") or "")
    if not version:
        return {}

    label = _version_label(version)
    if info.get("dirty"):
        # The commit does not describe the code that ran, so say so rather than link a clean tree.
        label += "+dirty"

    repo = REPOS.get(package, "")
    commit = str(info.get("commit") or "")
    if not commit and repo and resolve:
        commit = _resolve_release_commit(repo, version)
    if not repo:
        url = ""
    elif commit:
        url = f"https://github.com/{repo}/commit/{commit}"
    else:
        url = f"https://github.com/{repo}/releases"
    return {"version": version, "label": label, "url": url}


def engine_versions(engine: str) -> dict[str, dict]:
    """``{package: {version, commit?, dirty?}}`` for the stack behind ``engine``'s capture.

    Packages that aren't installed are left out, so this stays honest in a venv that only has some of
    them, and never raises: a version we cannot read must not cost us the capture.
    """
    out: dict[str, dict] = {}
    for module in ENGINE_PACKAGES.get(engine, ("torch",)):
        try:
            info = package_version(module)
        except Exception:  # noqa: BLE001 - metadata is nice to have, the capture is the point
            info = {}
        if info:
            out[module] = info
    return out
