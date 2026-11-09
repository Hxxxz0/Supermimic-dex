#!/usr/bin/env python3
"""Verify the released policy, exact winning configuration and frozen source contracts."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PublicationNormalizer(ast.NodeTransformer):
    """Normalize only the explicitly documented runtime and formatting changes."""

    def __init__(self, path):
        self.path = path

    def generic_visit(self, node):
        node = super().generic_visit(node)
        # Python 3.12 adds an empty type_params field to functions/classes.
        # Source contracts must be identical under both 3.11 and 3.12 parsers.
        if hasattr(node, "type_params"):
            if node.type_params:
                raise ValueError("The frozen Python 3.11 baseline has no generic type parameters")
            del node.type_params
        return node

    def visit_Import(self, node):
        if self.path == "contrack_supermimic/trainer.py":
            if len(node.names) == 1 and node.names[0].name in ("random", "numpy"):
                return None
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if self.path == "contrack_supermimic/experiment.py" and names == ["RESULT_ROOT"]:
            node.value = ast.Constant(value="publication-configurable-results-directory")
        elif self.path == "scripts/supermimic/train.py" and names == ["source_repo"]:
            node.value = ast.Constant(value="publication-default-source-checkout")
        elif self.path in ("scripts/supermimic/report.py", "scripts/supermimic/suite.py"):
            if isinstance(node.value, ast.Lambda) and names in (["avg"], ["mean"]):
                return ast.FunctionDef(
                    name=names[0],
                    args=node.value.args,
                    body=[ast.Return(value=node.value.body)],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                )
        return node


def canonical_ast(source, path):
    tree = PublicationNormalizer(path).visit(ast.parse(source))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def verify(repo=REPO):
    repo = Path(repo)
    frozen = json.loads((repo / "docs/frozen_source.json").read_text())
    checked = 0
    for name, record in frozen["files"].items():
        path = repo / name
        if name == "scripts/supermimic/python.sh":
            # This shell entry point was rewritten for relocatable Isaac/result paths.
            continue
        if name.startswith(("source/ConTrack/", "rsl_rl_contrack/")) or not name.endswith(".py"):
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen upstream source changed: {name}")
        elif canonical_ast(path.read_text(), name) != record["canonical_ast_sha256"]:
            raise ValueError(f"Executable source differs from the winning baseline: {name}")
        checked += 1
    expected = {name for name in frozen["files"] if name.startswith("contrack_supermimic/")}
    actual = {str(p.relative_to(repo)) for p in (repo / "contrack_supermimic").glob("*.py")}
    if actual != expected:
        raise ValueError("Core module set differs from the winning baseline")
    # Normalized runtime changes must still have exactly the intended release expressions.
    intended = {
        "contrack_supermimic/experiment.py": (
            "RESULT_ROOT",
            'Path(os.environ.get("CONTRACK_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "outputs"))).expanduser().resolve()',
        ),
        "scripts/supermimic/train.py": ("source_repo", "args.source_repo or REPO"),
    }
    for name, (variable, expression) in intended.items():
        nodes = [
            n
            for n in ast.walk(ast.parse((repo / name).read_text()))
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)
        ]
        wanted = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
        if len(nodes) != 1 or ast.dump(nodes[0].value, include_attributes=False) != wanted:
            raise ValueError(f"Unexpected publication runtime change: {name}")
    manifest = json.loads((repo / "results/hammer_best/manifest.json").read_text())
    evaluation = json.loads((repo / "results/hammer_best/evaluation.json").read_text())
    config = json.loads((repo / "configs/hammer_best.json").read_text())
    if config != manifest["config"] or manifest["algorithm"] != "supermimic":
        raise ValueError("Winning BPTT configuration is not preserved")
    if manifest["git_sha"] != frozen["source_sha"] or manifest["git_status"] != "":
        raise ValueError("Winning run source provenance changed")
    if sha256(repo / evaluation["checkpoint"]) != evaluation["checkpoint_sha256"]:
        raise ValueError("Released BPTT policy differs from the evaluated policy")
    if evaluation["episodes"] != 4096 or evaluation["success_rate"] != 4095 / 4096:
        raise ValueError("Recorded evaluation result changed")
    reference = repo / evaluation["reference"]
    if reference.is_file() and sha256(reference) != evaluation["reference_sha256"]:
        raise ValueError("Hammer reference differs from the evaluated clip")
    result = {
        "verified": True,
        "source_files_checked": checked,
        "source_sha": frozen["source_sha"],
        "policy_sha256": evaluation["checkpoint_sha256"],
        "successes": 4095,
        "episodes": 4096,
        "scope": "Source/configuration/artifact integrity; recorded success, not a new training experiment.",
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    verify(args.repo)


if __name__ == "__main__":
    main()
