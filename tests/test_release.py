"""Unit tests for the release script's decisions.

``.github/scripts/release.py`` picks the version every tag and every PyPI upload is named after,
and it runs exactly once per push -- on a runner, with a token, against the real registry. There is
no dry run that exercises the interesting half, so the arithmetic is tested here instead: what each
commit message asks for, which version wins when the tag and the pyproject disagree, and that
stamping rewrites the engine's version line and only that line.

The script is loaded by path because ``.github`` is not an importable package name. It is also not
shipped in the sdist, so these skip rather than fail when the suite runs from an unpacked release.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "release.py"

if not SCRIPT.is_file():
    pytest.skip("release.py is not part of the sdist", allow_module_level=True)


def _load():
    spec = importlib.util.spec_from_file_location("interp_engine_release", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release = _load()


class TestWhatACommitAsksFor:
    def test_an_ordinary_commit_is_a_patch(self):
        assert release.classify([("fix a dtype mismatch in mlp_act", "")]) == "patch"

    def test_no_commits_at_all_is_still_a_patch(self):
        # A dispatch with nothing new since the last tag. A version that goes nowhere is easier to
        # explain than a workflow that fails on an empty range.
        assert release.classify([]) == "patch"

    def test_a_feat_subject_is_a_minor(self):
        assert release.classify([("feat: capture attention sinks", "")]) == "minor"
        assert release.classify([("feat(vllm): capture attention sinks", "")]) == "minor"

    def test_a_bracket_marker_is_read_in_the_subject_or_the_body(self):
        assert release.classify([("add a point [minor]", "")]) == "minor"
        assert release.classify([("add a point", "squashed from #12 [MINOR]")]) == "minor"

    def test_a_bang_subject_is_a_major(self):
        assert release.classify([("feat!: rename resid_mid", "")]) == "major"
        assert release.classify([("refactor(points)!: drop the legacy names", "")]) == "major"

    def test_a_breaking_change_footer_is_a_major(self):
        assert release.classify([("rename a point", "BREAKING CHANGE: resid_mid is now resid_norm2")]) == "major"
        assert release.classify([("rename a point", "BREAKING-CHANGE: resid_mid is now resid_norm2")]) == "major"

    def test_the_highest_bump_wins_whatever_order_it_arrives_in(self):
        after = [("feat: a new point", ""), ("fix: a typo", ""), ("feat!: a rename", "")]
        assert release.classify(after) == "major"
        assert release.classify(list(reversed(after))) == "major"

    def test_a_feat_word_that_is_not_a_conventional_prefix_is_not_a_minor(self):
        # "feature" and "features" both start with feat; only `feat:` and `feat(scope):` count.
        assert release.classify([("features are documented in USAGE.md", "")]) == "patch"


class TestReadingGitLog:
    """`git()` strips its output, and the separators have to survive that.

    The ASCII record separator does not: Python counts \\x1c-\\x1f as whitespace, so `.strip()`
    removed the one that opens the first commit and every range came back one commit short -- a
    single-commit push looked like no commits at all, and its `feat:` silently became a patch.
    """

    def test_the_separators_are_not_whitespace(self):
        assert not release.RECORD.isspace()
        assert not release.FIELD.isspace()

    def test_a_single_commit_survives_the_strip_that_git_applies(self):
        raw = f"{release.RECORD}feat: capture attention sinks{release.FIELD}".strip()
        assert release.parse_log(raw) == [("feat: capture attention sinks", "")]

    def test_subjects_and_bodies_are_kept_apart(self):
        raw = (
            f"{release.RECORD}fix: a thing{release.FIELD}Co-authored-by: someone\n"
            f"{release.RECORD}feat!: a rename{release.FIELD}BREAKING CHANGE: resid_mid moved"
        )
        assert release.parse_log(raw.strip()) == [
            ("fix: a thing", "Co-authored-by: someone"),
            ("feat!: a rename", "BREAKING CHANGE: resid_mid moved"),
        ]

    def test_an_empty_range_is_no_commits(self):
        assert release.parse_log("") == []

    def test_the_log_format_asks_git_for_those_separators(self):
        assert release.LOG_FORMAT == "--format=%x00%s%x01%b"


class TestWhichVersionWins:
    def test_the_first_release_takes_the_declared_version(self):
        version, _ = release.resolve_version("1.1.0", None, "patch")
        assert version == "1.1.0"

    def test_a_hand_edited_pyproject_ahead_of_the_tags_names_the_release(self):
        # Both the deliberate major and the repair of a version that reached PyPI untagged.
        assert release.resolve_version("2.0.0", "1.4.3", "patch")[0] == "2.0.0"
        assert release.resolve_version("1.1.0", "1.0.0", "patch")[0] == "1.1.0"

    def test_otherwise_the_newest_tag_is_bumped(self):
        assert release.resolve_version("1.1.0", "1.1.0", "patch")[0] == "1.1.1"
        assert release.resolve_version("1.1.0", "1.1.0", "minor")[0] == "1.2.0"
        assert release.resolve_version("1.1.0", "1.1.0", "major")[0] == "2.0.0"

    def test_a_tag_ahead_of_the_pyproject_is_what_gets_bumped(self):
        # The state right after a release: the tag is authoritative, and a stale checkout of the
        # pyproject must not walk the version backwards.
        assert release.resolve_version("1.1.0", "1.3.0", "patch")[0] == "1.3.1"

    def test_a_version_that_is_not_x_y_z_is_refused_rather_than_guessed(self):
        with pytest.raises(release.Failure):
            release.resolve_version("1.2.0rc1", "1.1.0", "patch")

    def test_a_bump_resets_the_lower_parts(self):
        assert release.bump_version("1.4.7", "minor") == "1.5.0"
        assert release.bump_version("1.4.7", "major") == "2.0.0"

    def test_a_prerelease_is_not_a_comparable_version(self):
        assert release.version_key("1.2.3") == (1, 2, 3)
        assert release.version_key("1.2.3.dev0") is None
        assert release.version_key("v1.2.3") is None


class TestWhetherAChangeIsWorthAVersion:
    """A version is what a downstream pin moves to, so only the imported engine earns one.

    Two exclusions with different reasons: `visualizer-web/` and the validator never reach a user,
    while `tests/`, `docs/`, `benchmarks/` and the README do reach one and still cannot change what
    `import interp_engine` gives them.
    """

    PREFIXES = ["interp_engine", "tests", "docs", "benchmarks", "README.md", "pyproject.toml"]

    def test_the_engine_and_its_dependency_set_are_worth_a_version(self):
        assert release.changes_the_package("interp_engine/points.py", self.PREFIXES)
        # The dependency table is part of what an install resolves, so pyproject counts.
        assert release.changes_the_package("pyproject.toml", self.PREFIXES)

    def test_the_sidecar_projects_are_not(self):
        assert not release.changes_the_package("visualizer-web/data/points.ts", self.PREFIXES)
        assert not release.changes_the_package("validator/comparison/score.py", self.PREFIXES)
        assert not release.changes_the_package("plans/deepseek-v4-b200-bringup.md", self.PREFIXES)

    def test_prose_tests_and_measurements_ship_without_being_worth_a_version(self):
        # Every one of these is inside the sdist -- `is_packaged` says yes and the answer is no.
        assert release.is_packaged("docs/USAGE.md", self.PREFIXES)
        assert not release.changes_the_package("docs/USAGE.md", self.PREFIXES)
        assert not release.changes_the_package("README.md", self.PREFIXES)
        assert not release.changes_the_package("tests/test_points.py", self.PREFIXES)
        assert not release.changes_the_package("benchmarks/results/gpt2.json", self.PREFIXES)

    def test_a_markdown_file_beside_the_engine_is_still_only_prose(self):
        assert not release.changes_the_package("interp_engine/README.md", self.PREFIXES)
        assert release.changes_the_package("interp_engine/vllm_capture/mhc.py", self.PREFIXES)

    def test_the_first_release_needs_no_comparison(self):
        cut, why = release.release_plan("1.1.0", None, "auto", self.PREFIXES)
        assert cut is True
        assert "no v* tag" in why

    def test_a_hand_declared_version_releases_whatever_changed(self):
        # The escape hatch for a docs-only release, and for repairing a version that reached PyPI
        # untagged. It is read before anything is diffed, so it cannot be overruled.
        cut, why = release.release_plan("2.0.0", "1.4.3", "auto", self.PREFIXES)
        assert cut is True
        assert "by hand" in why

    def test_a_dispatched_bump_releases_whatever_changed(self):
        cut, why = release.release_plan("1.4.3", "1.4.3", "minor", self.PREFIXES)
        assert cut is True
        assert "by hand" in why


class TestWhetherToUpload:
    PREFIXES = ["interp_engine", "tests", "docs", "README.md", "pyproject.toml"]

    def test_a_version_pypi_already_has_is_tagged_but_not_uploaded(self):
        upload, why = release.publish_plan("1.1.0", {"1.0.0", "1.1.0"}, self.PREFIXES)
        assert upload is False
        assert "already has" in why

    def test_an_unreachable_pypi_assumes_the_upload_is_needed(self):
        upload, _ = release.publish_plan("1.1.1", None, self.PREFIXES)
        assert upload is True

    def test_a_project_that_has_never_been_published_is_uploaded(self):
        upload, _ = release.publish_plan("1.1.1", set(), self.PREFIXES)
        assert upload is True

    def test_only_paths_the_sdist_ships_count_as_a_reason_to_publish(self):
        assert release.is_packaged("interp_engine/points.py", self.PREFIXES)
        assert release.is_packaged("README.md", self.PREFIXES)
        assert not release.is_packaged("validator/comparison/score.py", self.PREFIXES)
        assert not release.is_packaged("visualizer-web/data/points.ts", self.PREFIXES)
        assert not release.is_packaged("plans/deepseek-v4-b200-bringup.md", self.PREFIXES)

    def test_a_prefix_matches_a_directory_and_not_a_lookalike(self):
        assert not release.is_packaged("tests_helpers/thing.py", ["tests"])
        assert release.is_packaged("tests/test_facts.py", ["tests"])

    def test_the_previous_releases_own_stamp_is_not_a_change_worth_publishing(self):
        diff = '--- a/pyproject.toml\n+++ b/pyproject.toml\n-version = "1.1.0"\n+version = "1.1.1"\n'
        assert release.only_the_version_changed(diff)

    def test_a_real_pyproject_edit_beside_the_stamp_still_publishes(self):
        diff = (
            "--- a/pyproject.toml\n+++ b/pyproject.toml\n"
            '-version = "1.1.0"\n+version = "1.1.1"\n'
            '-    "transformers>=4.57.1",\n+    "transformers>=5.14",\n'
        )
        assert not release.only_the_version_changed(diff)


class TestStamping:
    PYPROJECT = '[project]\nname = "interp-engine"\nversion = "1.1.0"\nrequires-python = ">=3.11"\n\n[project.urls]\nHomepage = "https://example.invalid"\n'

    LOCK = (
        '[[package]]\nname = "einops"\nversion = "0.8.0"\n\n'
        '[[package]]\nname = "interp-engine"\nversion = "1.1.0"\nsource = { editable = "." }\n\n'
        '[package.metadata]\nrequires-dist = [{ name = "torch", specifier = ">=1.10" }]\n\n'
        '[[package]]\nname = "torch"\nversion = "2.9.0"\n'
    )

    def test_it_rewrites_the_project_version(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(self.PYPROJECT)
        assert release.stamp("pyproject.toml", "1.2.0", root=tmp_path)
        assert 'version = "1.2.0"' in (tmp_path / "pyproject.toml").read_text()

    def test_it_rewrites_only_the_engines_entry_in_a_lockfile(self, tmp_path):
        (tmp_path / "uv.lock").write_text(self.LOCK)
        assert release.stamp("uv.lock", "1.2.0", root=tmp_path)
        updated = (tmp_path / "uv.lock").read_text()
        assert 'name = "interp-engine"\nversion = "1.2.0"' in updated
        # The neighbours are what a careless regex eats: one sorts before the engine, one after.
        assert 'name = "einops"\nversion = "0.8.0"' in updated
        assert 'name = "torch"\nversion = "2.9.0"' in updated

    def test_stamping_the_version_that_is_already_there_changes_nothing(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(self.PYPROJECT)
        assert release.stamp("pyproject.toml", "1.1.0", root=tmp_path) is False
        assert (tmp_path / "pyproject.toml").read_text() == self.PYPROJECT

    def test_a_file_with_nowhere_to_stamp_is_an_error_rather_than_a_silent_no_op(self, tmp_path):
        (tmp_path / "uv.lock").write_text('[[package]]\nname = "torch"\nversion = "2.9.0"\n')
        with pytest.raises(release.Failure):
            release.stamp("uv.lock", "1.2.0", root=tmp_path)
