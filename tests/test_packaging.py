"""Distribution-level checks: the version and the public surface.

These guard the packaging rather than the engine. They are what catch a release where
``pyproject.toml`` was bumped but ``interp_engine.__version__`` reports something else, or
where a name was listed in ``__all__`` and then renamed out from under it -- both of which
only surface for a downstream consumer, after the tag is cut.
"""

import tomllib
from pathlib import Path

import pytest

import interp_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Both lockfiles carry the engine's version: the root one because the engine is its own project,
# and the validator's because it depends on the engine through a `{ path = ".." }` source. Neither
# is shipped in the sdist, hence the skip in the test that reads them.
LOCKFILES = ("uv.lock", "validator/uv.lock")


def _declared_version() -> str:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _locked_version(lockfile: Path) -> str | None:
    with open(lockfile, "rb") as f:
        packages = tomllib.load(f).get("package", [])
    return next((p.get("version") for p in packages if p.get("name") == "interp-engine"), None)


class TestTheVersion:
    def test_the_package_reports_the_version_pyproject_declares(self):
        # __version__ reads importlib.metadata, so this only agrees when the installed
        # distribution matches the source tree it was installed from.
        assert interp_engine.__version__ == _declared_version()

    def test_the_version_is_not_the_uninstalled_fallback(self):
        # The fallback means the package resolved but its distribution metadata did not,
        # i.e. an un-editable source checkout. Tests should never run against that.
        assert interp_engine.__version__ != "0.0.0.dev0"

    def test_the_distribution_is_named_interp_engine(self):
        with open(PYPROJECT, "rb") as f:
            assert tomllib.load(f)["project"]["name"] == "interp-engine"

    def test_both_lockfiles_record_the_version_pyproject_declares(self):
        # A lockfile left behind by a version bump is not a packaging nit: validator-tests.yml runs
        # `uv sync --locked`, which fails when a lock records a version the pyproject no longer
        # declares -- and it fails on the next unrelated PR rather than on the bump. Releases keep
        # all three in step (.github/scripts/release.py); this is what catches a bump by hand.
        declared = _declared_version()
        for name in LOCKFILES:
            lockfile = REPO_ROOT / name
            if not lockfile.is_file():
                pytest.skip(f"{name} is not shipped in the sdist")
            assert _locked_version(lockfile) == declared, (
                f"{name} records interp-engine {_locked_version(lockfile)}, but pyproject.toml "
                f"declares {declared}. Update the lock (or let the release script stamp it)."
            )


class TestThePublicSurface:
    def test_every_exported_name_resolves(self):
        missing = [name for name in interp_engine.__all__ if not hasattr(interp_engine, name)]
        assert missing == [], f"__all__ names the module does not define: {missing}"

    def test_no_name_is_exported_twice(self):
        duplicates = sorted({n for n in interp_engine.__all__ if interp_engine.__all__.count(n) > 1})
        assert duplicates == [], f"listed more than once in __all__: {duplicates}"

    def test_the_autograd_surface_is_exported(self):
        # Phase-4 additions: the verdict object, the refusal, and both probes. A consumer
        # gating on gradient support reaches for these by name.
        for name in ("GradSupport", "GradientsUnsupported", "eager_grad_support", "vllm_grad_support"):
            assert name in interp_engine.__all__, f"{name} is public API but not in __all__"
