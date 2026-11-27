"""The shim tree contract: ``check_tree`` reports a tree, ``plan_migration`` moves it.

These tests lived in ``test_new_shim.py`` while the check and the migration were
part of the scaffolder; they moved with the code to
``llm_rosetta.authoring.tree_contract``.
"""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.authoring.tree_contract import (
    apply_migration,
    check_tree,
    plan_migration,
    run_migrate,
)


class TestCheckTree:
    """`shims check` reports what the loader would not read, and nothing else."""

    @staticmethod
    def _manifest(vendor_dir: Path) -> None:
        vendor_dir.mkdir(parents=True, exist_ok=True)
        (vendor_dir / "provider.yaml").write_text(
            "connection:\n  api_key_env: ACME_API_KEY\n", encoding="utf-8"
        )

    @staticmethod
    def _leaf(directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provider.yaml"
        path.write_text(
            f"name: {name}\nmodel_type: llm\nformat: chat_completions\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )
        return path

    def test_a_canonical_tree_has_no_findings(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        assert check_tree(tmp_path) == []

    def test_a_vendor_directory_used_as_a_leaf_is_reported(self, tmp_path: Path):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "a vendor directory is a leaf" in findings[0].message

    def test_a_relative_token_command_without_its_file_is_reported(
        self, tmp_path: Path
    ):
        """The command runs in the leaf directory, so the file must be there.

        A moved leaf leaves `../token.py` one level short, and nothing fails
        until the first credential exchange.
        """

        self._manifest(tmp_path / "acme")
        leaf = tmp_path / "acme" / "llm"
        leaf.mkdir(parents=True)
        (leaf / "provider.yaml").write_text(
            "name: acme--chat_completions\nmodel_type: llm\n"
            "format: chat_completions\nconnection:\n"
            "  token_command: [python3, ../token.py]\n",
            encoding="utf-8",
        )

        findings = check_tree(tmp_path)

        assert len(findings) == 1
        assert "connection.token_command" in findings[0].message

        (tmp_path / "acme" / "token.py").write_text("", encoding="utf-8")
        assert check_tree(tmp_path) == []

    def test_a_broken_fields_block_is_reported_before_it_is_installed(
        self, tmp_path: Path
    ):
        """The loader refuses the whole root, so the check must say which file.

        The panel renders those entries verbatim; a typo in one would otherwise
        surface as a gateway that will not load the tree at all.
        """

        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "provider.yaml").write_text(
            "fields:\n  - key: email\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        findings = check_tree(tmp_path)

        assert [f.path.name for f in findings] == ["provider.yaml"]
        assert "declares no label" in findings[0].message
        # The finding carries the path, so the sentence does not repeat it.
        assert str(tmp_path) not in findings[0].message

    def test_a_well_formed_fields_block_is_not_a_finding(self, tmp_path: Path):

        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "provider.yaml").write_text(
            "fields:\n  - key: email\n    label: Account email\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        assert check_tree(tmp_path) == []

    def test_a_retired_exchange_in_a_private_directory_is_not_a_second_one(
        self, tmp_path: Path
    ):
        """``_old/`` is where a package parks what the loader must not read.

        No scan reaches into it, so an ``auth.py`` there is not a competing
        exchange — reporting it would send the author looking for a conflict
        that does not exist.
        """

        self._manifest(tmp_path / "acme")
        (tmp_path / "acme" / "auth.py").write_text(
            "EXCHANGE = None\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        retired = tmp_path / "acme" / "_old"
        retired.mkdir()
        (retired / "auth.py").write_text("EXCHANGE = None\n", encoding="utf-8")

        assert check_tree(tmp_path) == []

    def test_a_type_directory_holding_a_leaf_and_a_lane_is_reported(
        self, tmp_path: Path
    ):
        """The half-migrated shape: the flat leaf is the one that moves.

        Reporting the nested lane as "the only lane ... it belongs directly
        under <type>/" names a move that would collide with the leaf already
        there, so the finding has to name the flat leaf and send it down.
        """

        self._manifest(tmp_path / "acme")
        flat = self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        self._leaf(tmp_path / "acme" / "llm" / "messages", "acme--messages")

        findings = check_tree(tmp_path)

        assert len(findings) == 1
        assert findings[0].path == flat
        assert "holds a leaf and format directories at once" in findings[0].message
        assert "Move this leaf down" in findings[0].message

    def test_a_leaf_without_a_type_directory_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "chat_completions", "acme--chat_completions")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "no <type>/ directory" in findings[0].message

    def test_the_only_lane_at_a_format_level_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "the only lane of llm" in findings[0].message

    def test_two_lanes_at_a_format_level_conform(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )
        second = tmp_path / "acme" / "llm" / "messages"
        second.mkdir()
        (second / "provider.yaml").write_text(
            "name: acme--messages\nmodel_type: llm\nformat: messages\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )

        assert check_tree(tmp_path) == []

    def test_a_vendor_level_transforms_file_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "transforms.py").write_text("", encoding="utf-8")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "vendor-level one is never read" in findings[0].message

    def test_a_second_exchange_below_the_vendor_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "auth.py").write_text("", encoding="utf-8")
        (tmp_path / "acme" / "llm" / "auth.py").write_text("", encoding="utf-8")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "already ships its exchange" in findings[0].message

    def test_a_private_helper_module_is_not_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "model_utils.py").write_text("", encoding="utf-8")

        assert check_tree(tmp_path) == []


class TestMigrate:
    """`shims migrate` moves a tree onto the contract, and only what it can."""

    @staticmethod
    def _leaf(
        directory: Path, name: str, *, model_type: str = "llm", fmt="chat_completions"
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "provider.yaml").write_text(
            f"name: {name}\nmodel_type: {model_type}\nformat: {fmt}\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )

    @staticmethod
    def _manifest(vendor_dir: Path) -> None:
        vendor_dir.mkdir(parents=True, exist_ok=True)
        (vendor_dir / "provider.yaml").write_text(
            "connection:\n  api_key_env: ACME_API_KEY\n", encoding="utf-8"
        )

    def test_a_vendor_used_as_a_leaf_gets_a_manifest_and_a_type_level(
        self, tmp_path: Path
    ):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        moves, notes = plan_migration(tmp_path)
        assert notes == []
        apply_migration(tmp_path, moves)

        assert (tmp_path / "acme" / "llm" / "provider.yaml").is_file()
        manifest = (tmp_path / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert manifest.endswith("connection: {}\n")

    def test_a_format_directory_under_the_vendor_moves_under_its_type(
        self, tmp_path: Path
    ):

        self._manifest(tmp_path / "acme")
        for fmt in ("chat_completions", "messages"):
            self._leaf(tmp_path / "acme" / fmt, f"acme--{fmt}", fmt=fmt)

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (
            tmp_path / "acme" / "llm" / "chat_completions" / "provider.yaml"
        ).is_file()
        assert (tmp_path / "acme" / "llm" / "messages" / "provider.yaml").is_file()

    def test_a_lane_gets_its_type_directory_below_it(self, tmp_path: Path):

        self._manifest(tmp_path / "alcf")
        for cluster in ("metis", "minerva"):
            self._leaf(tmp_path / "alcf" / cluster, f"alcf_{cluster}--chat_completions")
            (tmp_path / "alcf" / cluster / "transforms.py").write_text(
                "", encoding="utf-8"
            )

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        for cluster in ("metis", "minerva"):
            leaf = tmp_path / "alcf" / cluster / "llm"
            assert (leaf / "provider.yaml").is_file()
            assert (leaf / "transforms.py").is_file()
        assert not (tmp_path / "alcf" / "metis" / "llm" / "llm").exists()

    def test_the_only_lane_at_a_format_level_moves_up(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (tmp_path / "acme" / "llm" / "provider.yaml").is_file()
        assert not (tmp_path / "acme" / "llm" / "chat_completions").exists()

    def test_a_dry_run_moves_nothing(self, tmp_path: Path, capsys):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        code = run_migrate(tmp_path)

        out = capsys.readouterr().out
        assert code == 1
        assert "rerun with --write" in out
        assert (tmp_path / "acme" / "provider.yaml").is_file()
        assert not (tmp_path / "acme" / "llm").exists()

    def test_a_second_lane_leaves_the_sibling_lane_where_it_is(self, tmp_path: Path):
        """The flat lane moves down; the lane already at a format level stays.

        The type directory is both the flat lane's home and the parent of a
        sibling lane, so moving it wholesale would drag the sibling one level
        deeper — a tree this migration's own `check` then rejects.
        """

        self._manifest(tmp_path / "acme")
        lane = tmp_path / "acme" / "llm"
        self._leaf(lane, "acme--messages", fmt="messages")
        (lane / "transforms.py").write_text("", encoding="utf-8")
        self._leaf(lane / "chat_completions", "acme--chat_completions")

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (lane / "messages" / "provider.yaml").is_file()
        assert (lane / "messages" / "transforms.py").is_file()
        assert (lane / "chat_completions" / "provider.yaml").is_file()
        assert not (lane / "messages" / "chat_completions").exists()
        assert check_tree(tmp_path) == []

    def test_a_vendor_leaf_with_a_sibling_lane_keeps_the_sibling(self, tmp_path: Path):
        """The vendor leaf moves down without taking its sibling with it.

        Dragging the sibling would leave the second move with no source, so the
        migration would fail halfway through.
        """

        self._leaf(tmp_path / "acme", "acme--chat_completions")
        self._leaf(tmp_path / "acme" / "messages", "acme--messages", fmt="messages")

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (
            tmp_path / "acme" / "llm" / "chat_completions" / "provider.yaml"
        ).is_file()
        assert (tmp_path / "acme" / "llm" / "messages" / "provider.yaml").is_file()
        assert not (
            tmp_path / "acme" / "llm" / "chat_completions" / "messages"
        ).exists()
        assert check_tree(tmp_path) == []

    def test_a_conforming_tree_has_nothing_to_do(self, tmp_path: Path, capsys):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        code = run_migrate(tmp_path, write=True)

        assert code == 0
        assert "nothing to migrate" in capsys.readouterr().out
