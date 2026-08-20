#!/usr/bin/env python3
"""Decide what the next release is, and stamp that version into every file that has to agree.

Tags are the release history, and this script is what picks the next one. It reads the commit
messages since the newest `v*` tag and takes the highest bump any of them asks for: `[major]`, a
conventional-commit `!` marker or a `BREAKING CHANGE:` footer give a major; `[minor]` or a `feat:`
subject give a minor; anything else is a patch. That is the same vocabulary the Neuronpedia repo
releases on, so a commit written for one repo behaves the same in the other.

`.github/workflows/release.yml` runs this on every push to `main`. Run it yourself to see what the
next push would cut -- with no arguments it resolves, prints and writes nothing:

    python3 .github/scripts/release.py            # what would the next release be?
    python3 .github/scripts/release.py --write    # what CI does: also stamp the files

Three files record the engine's version and all three are rewritten together:

    pyproject.toml     the declared version, and what `interp_engine.__version__` reports back
    uv.lock            the engine's own entry in its lockfile
    validator/uv.lock  the same entry again, because the validator depends on the engine by path

Leaving the lockfiles behind is not cosmetic. `validator-tests.yml` runs `uv sync --locked`, which
fails when a lockfile records a version the pyproject no longer declares, so a release that stamped
only the pyproject would turn the next unrelated PR red instead of failing here.
`tests/test_packaging.py` asserts the three agree, so a hand bump that forgets one is caught too.

**A hand-edited pyproject version wins.** `resolve_version` takes the higher of the newest tag and
the declared version, so writing `version = "2.0.0"` by hand names the next release outright. That
is how a deliberate major or minor lands without relying on a commit-message marker, and it is also
what reconciles a version that reached PyPI while its tag went missing: the tag is created for the
version already published, and the upload below is skipped rather than failing on a duplicate.

**Not every push is a release**, which is the second question answered here. A version is what a
downstream pin moves to, so it is cut only when something an importing caller gets changed: not the
visualizer, the validator or `plans/`, which the sdist does not ship, and not `tests/`, `docs/`,
`benchmarks/` or any Markdown, which it ships but which cannot change what `import interp_engine`
does. A push carrying only those is left untagged and unreleased. Both hand overrides still win
outright -- a declared version ahead of the newest tag, or an explicit `--release-type` -- so a
docs-only fix can still be released when someone means it.

**Not every tag is a PyPI upload**, which is the third. A release whose version PyPI already has is
tagged and released on GitHub but not uploaded. The comparison is made against the newest version
*on PyPI* rather than the newest tag, so a publish that failed for any reason is retried by the next
push instead of being skipped as "already done".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DISTRIBUTION = "interp-engine"

# Every file that records the engine's own version, in the order the plan prints them. The two
# lockfiles hold it in a `[[package]]` block keyed by name; the pyproject holds it in `[project]`.
VERSIONED_FILES = ("pyproject.toml", "uv.lock", "validator/uv.lock")

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Anchored to a line start so the `version = ` of a dependency table cannot match; the search is
# additionally bounded to one table by `_version_span`.
VERSION_LINE = re.compile(r'^version = "[^"]+"$', re.M)
PROJECT_TABLE = re.compile(r"^\[project\]$", re.M)
PACKAGE_ENTRY = re.compile(rf'^name = "{re.escape(DISTRIBUTION)}"$', re.M)
NEXT_TABLE = re.compile(r"^\[", re.M)

# What a commit message has to say to ask for more than a patch. The bracket markers are matched in
# the body as well as the subject, so a squashed PR can carry one in its description.
MAJOR_MARKER = re.compile(r"\[major\]", re.I)
MINOR_MARKER = re.compile(r"\[minor\]", re.I)
BREAKING_SUBJECT = re.compile(r"^[a-zA-Z]+(\([^)]*\))?!:")
BREAKING_FOOTER = re.compile(r"^BREAKING[ -]CHANGE:", re.M)
FEATURE_SUBJECT = re.compile(r"^feat(\([^)]*\))?:")

# `git log` field separators, so a subject or body containing any plausible delimiter still parses:
# RECORD opens each commit, FIELD splits its subject from its body.
#
# NUL and SOH rather than the ASCII record/unit separators that would read better here, because
# `str.strip()` counts \x1c-\x1f as whitespace and would eat the leading RECORD -- which silently
# dropped the first commit of every range, i.e. exactly the `feat:` in a one-commit push. They are
# written as git's own `%x00` escapes: a literal NUL cannot be passed in an argv string.
RECORD, FIELD = "\x00", "\x01"
LOG_FORMAT = f"--format=%x{ord(RECORD):02x}%s%x{ord(FIELD):02x}%b"


class Failure(Exception):
    """A condition the operator has to fix; reported without a traceback."""


# ---------------------------------------------------------------------------------- versions --


def version_key(version: str) -> tuple[int, int, int] | None:
    """Parse `x.y.z` into a comparable tuple, or None when it is not a plain release version.

    Pre-releases and local versions return None so they are skipped rather than sorting as zero,
    which would silently make `0.1.0rc1` look newer than `1.1.0`.
    """
    match = SEMVER.match(version.strip())
    if not match:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def bump_version(version: str, kind: str) -> str:
    key = version_key(version)
    if key is None:
        raise Failure(f"cannot bump {version!r}: not a plain x.y.z version")
    major, minor, patch = key
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def classify(commits: Sequence[tuple[str, str]]) -> str:
    """The highest bump any of these commits asks for, defaulting to patch.

    Unrecognized subjects fall through to patch rather than to "no release", so a repo that does
    not write conventional commits still gets a version for every push.
    """
    kind = "patch"
    for subject, body in commits:
        message = f"{subject}\n{body}"
        if MAJOR_MARKER.search(message) or BREAKING_SUBJECT.match(subject) or BREAKING_FOOTER.search(body):
            return "major"
        if MINOR_MARKER.search(message) or FEATURE_SUBJECT.match(subject):
            kind = "minor"
    return kind


def resolve_version(declared: str, newest_tag: str | None, kind: str) -> tuple[str, str]:
    """The version to release, and one line saying why -- see the module docstring's third rule."""
    declared_key = version_key(declared)
    if declared_key is None:
        raise Failure(f"pyproject.toml declares {declared!r}, which is not a plain x.y.z version")
    if newest_tag is None:
        return declared, f"no v* tag exists yet, so the declared version {declared} becomes the first"
    tag_key = version_key(newest_tag)
    if tag_key is None:
        raise Failure(f"newest tag v{newest_tag} is not a plain x.y.z version")
    if declared_key > tag_key:
        return declared, f"pyproject declares {declared}, ahead of the newest tag v{newest_tag}: releasing that"
    return bump_version(newest_tag, kind), f"{kind} bump from the newest tag v{newest_tag}"


# --------------------------------------------------------------------------------------- git --


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Failure(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def newest_tag() -> str | None:
    """The highest `vX.Y.Z` tag in the repo, by version rather than by date or by reachability.

    Deliberately not `git describe --abbrev=0`, which answers "nearest tag on this branch" -- on a
    branch that missed a release that is an older tag, and bumping from it would reuse a version.
    """
    versions = [v for tag in git("tag", "--list", "v[0-9]*").splitlines() if (v := tag.strip().lstrip("v"))]
    keyed = [(key, v) for v in versions if (key := version_key(v)) is not None]
    if not keyed:
        return None
    return max(keyed)[1]


def parse_log(out: str) -> list[tuple[str, str]]:
    """(subject, body) per commit, from the RECORD/FIELD-delimited `git log` above."""
    entries = []
    for chunk in out.split(RECORD)[1:]:
        subject, _, body = chunk.partition(FIELD)
        entries.append((subject.strip(), body.strip()))
    return entries


def commits_in(rev_range: str) -> list[tuple[str, str]]:
    """(subject, body) for every commit in the range, newest first."""
    return parse_log(git("log", LOG_FORMAT, rev_range))


def tag_exists(tag: str) -> bool:
    return bool(git("tag", "--list", tag))


# -------------------------------------------------------------------- what a change is worth --

# Shipped in the sdist, and incapable of changing what an importing caller gets: the test suite, the
# docs tree, the benchmark harness and the numbers it records, and Markdown anywhere. They belong in
# the tarball and they are not worth a version -- every downstream pin has to move for one, and the
# release notes then describe a fixed typo or a re-run sweep as a new engine. What is left is
# `interp_engine/` itself, `pyproject.toml` (the dependency set is part of the contract) and LICENSE.
INERT_PREFIXES = ("tests", "docs", "benchmarks")
INERT_SUFFIXES = (".md",)


def packaged_prefixes(pyproject: dict) -> list[str]:
    """The sdist's include list, as repo-relative prefixes.

    Read from the packaging config rather than restated here, so a directory added to the sdist is
    considered by `changes_the_package` below without this file having to hear about it.
    """
    include = pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("sdist", {})
    patterns = include.get("include", [])
    if not patterns:
        raise Failure("pyproject.toml has no [tool.hatch.build.targets.sdist] include list to read")
    return [pattern.strip("/") for pattern in patterns]


def is_packaged(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def changes_the_package(path: str, prefixes: Sequence[str]) -> bool:
    """True when this path both ships and can change what an importing caller gets.

    Two exclusions, and they are different things. Everything outside the sdist -- the visualizer,
    the validator, `plans/`, `.github/` -- never reaches a user at all. The prefixes and suffixes
    above do reach one, and still cannot change the engine's behavior.
    """
    return is_packaged(path, prefixes) and not is_packaged(path, INERT_PREFIXES) and not path.endswith(INERT_SUFFIXES)


def only_the_version_changed(diff: str) -> bool:
    """True when every added/removed line in this diff is a `version = "..."` line.

    The previous release's own stamp commit touches pyproject.toml, which is a packaged file, so
    without this a push that changes nothing shippable would still look like a reason to publish --
    one no-op upload per release, forever.
    """
    changed = [
        line[1:].strip() for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    return bool(changed) and all(VERSION_LINE.match(line) for line in changed)


def package_changes_since(reference: str, prefixes: Sequence[str]) -> list[str]:
    """Every file changed between `reference` and HEAD that is worth a new version."""
    diffed = git("diff", "--name-only", f"{reference}..HEAD").splitlines()
    changed = [path for path in diffed if changes_the_package(path, prefixes)]
    if changed == ["pyproject.toml"] and only_the_version_changed(
        git("diff", "-U0", f"{reference}..HEAD", "--", "pyproject.toml")
    ):
        return []
    return changed


def release_plan(declared: str, newest_tag: str | None, release_type: str, prefixes: Sequence[str]) -> tuple[bool, str]:
    """Whether this push is worth a version at all, and one line saying why.

    The two hand overrides come first and are unconditional: someone who declares a version or
    dispatches a bump has said what they want, and a "nothing changed" reading must not overrule it.
    """
    if newest_tag is None:
        return True, "no v* tag exists yet, so this is the first release"
    if release_type != "auto":
        return True, f"a {release_type} release was asked for by hand"
    declared_key, tag_key = version_key(declared), version_key(newest_tag)
    if declared_key and tag_key and declared_key > tag_key:
        return True, f"pyproject declares {declared}, ahead of v{newest_tag}, so a release was asked for by hand"

    changed = package_changes_since(f"v{newest_tag}", prefixes)
    if not changed:
        return False, f"nothing that changes the package has changed since v{newest_tag}"
    return True, f"{len(changed)} file(s) that change the package since v{newest_tag}, e.g. {changed[0]}"


# -------------------------------------------------------------------------------------- pypi --


def pypi_releases(name: str) -> set[str] | None:
    """Every version on PyPI, or None when PyPI could not be asked.

    None is not the same as empty: an unreachable index means "assume a release is needed", where
    an empty set means the project has never been published.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        # Fixed https URL built from a constant; nothing here is caller-controlled.
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return set(json.load(response).get("releases", {}))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        return None
    except (OSError, json.JSONDecodeError):
        return None


def publish_plan(version: str, published: set[str] | None, prefixes: Sequence[str]) -> tuple[bool, str]:
    """Whether this release should also be uploaded, and one line saying why."""
    if published is None:
        return True, "PyPI could not be reached, so assume the upload is needed"
    if version in published:
        return False, f"PyPI already has {version}: tagging it, uploading nothing"
    keyed = [(key, v) for v in published if (key := version_key(v)) is not None]
    if not keyed:
        return True, "nothing is published yet"

    reference = f"v{max(keyed)[1]}"
    if not tag_exists(reference):
        return True, f"{reference} is the newest on PyPI but is not tagged here, so there is nothing to compare"

    changed = package_changes_since(reference, prefixes)
    if not changed:
        return False, f"nothing that changes the package has changed since {reference}"
    return True, f"{len(changed)} packaged file(s) changed since {reference}, e.g. {changed[0]}"


# ------------------------------------------------------------------------------------ writing --


def _version_span(relative: str, text: str) -> tuple[int, int]:
    """The slice of the file holding the engine's own version line."""
    opener = PROJECT_TABLE if relative == "pyproject.toml" else PACKAGE_ENTRY
    match = opener.search(text)
    if match is None:
        raise Failure(f"{relative} has no {opener.pattern} to stamp a version into")
    following = NEXT_TABLE.search(text, match.end())
    return match.end(), following.start() if following else len(text)


def stamp(relative: str, version: str, root: Path = REPO_ROOT) -> bool:
    """Write `version` into one file. True when the file changed, False when it already said that."""
    path = root / relative
    text = path.read_text()
    start, end = _version_span(relative, text)
    match = VERSION_LINE.search(text, start, end)
    if match is None:
        raise Failure(f"{relative} has no version line where one was expected")
    updated = f'{text[: match.start()]}version = "{version}"{text[match.end() :]}'
    if updated == text:
        return False
    path.write_text(updated)
    return True


def emit(outputs: dict[str, str]) -> None:
    """Publish the plan to the workflow, if we are running in one."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


# --------------------------------------------------------------------------------------- main --


def plan(release_type: str, write: bool) -> int:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = str(pyproject["project"]["version"])
    tag = newest_tag()
    rev_range = f"v{tag}..HEAD" if tag else "HEAD"

    commits = commits_in(rev_range)
    kind = classify(commits) if release_type == "auto" else release_type

    version, why = resolve_version(declared, tag, kind)
    prefixes = packaged_prefixes(pyproject)

    # PyPI is asked only when there is going to be a release to upload, so a push that changes
    # nothing shippable makes no network call and cannot be delayed by an unreachable index.
    cut, cut_why = release_plan(declared, tag, release_type, prefixes)
    upload, upload_why = (
        publish_plan(version, pypi_releases(DISTRIBUTION), prefixes)
        if cut
        else (False, "there is no release to upload")
    )

    print(f"declared      {declared} (pyproject.toml)")
    print(f"newest tag    {'v' + tag if tag else '(none)'}")
    print(f"commits       {len(commits)} in {rev_range}")
    print(f"release type  {kind}{'' if release_type == 'auto' else ' (forced)'}")
    print(f"release       {'yes' if cut else 'no'} -- {cut_why}")
    print(f"version       {version if cut else '(none)'} -- {why if cut else 'nothing is tagged or stamped'}")
    print(f"publish       {'yes' if upload else 'no'} -- {upload_why}")

    stamped = []
    if write and cut:
        stamped = [f for f in VERSIONED_FILES if stamp(f, version)]
        print(
            f"stamped       {', '.join(stamped) if stamped else '(nothing: every file already said ' + version + ')'}"
        )

    emit(
        {
            "release": "true" if cut else "false",
            "version": version,
            "tag": f"v{version}",
            "previous_tag": f"v{tag}" if tag else "",
            "range": rev_range,
            "bump": kind,
            "publish": "true" if upload else "false",
            "publish_why": upload_why,
            "stamped": "true" if stamped else "false",
            "skipped": "" if cut else cut_why,
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--release-type",
        default="auto",
        choices=("auto", "major", "minor", "patch"),
        help="override the bump the commit messages ask for.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="stamp the resolved version into pyproject.toml and both lockfiles. Off by default, so "
        "a bare run is a safe way to ask what the next release would be.",
    )
    args = parser.parse_args()

    try:
        return plan(args.release_type, args.write)
    except Failure as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
