"""Build the literature-guided next-phase research plan notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "15_literature_guided_research_plan.ipynb"


def md(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": "md-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str, hidden: bool = False) -> dict:
    text = dedent(source).strip()
    metadata = {"jupyter": {"source_hidden": True}, "collapsed": True} if hidden else {}
    return {
        "cell_type": "code",
        "id": "code-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """
        # 15 - Literature-Guided Research Plan

        The previous notebooks gave us a useful negative result: the original
        long-lived `H1` story does not survive controls as a universal claim.
        Mapper also looks more useful as a node-level diagnostic than as a
        graph-loop detector.

        This notebook turns that into a tighter research program. The aim is
        not to keep sweeping until something looks pretty. The aim is to ask
        which geometric, topological, and Mapper summaries become stable enough
        to be useful.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        from pathlib import Path
        import sys

        import pandas as pd
        from IPython.display import display, Markdown

        for candidate in [Path.cwd(), Path.cwd().parent]:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        ROOT = Path.cwd()
        if not (ROOT / "notebooks").exists() and (ROOT.parent / "notebooks").exists():
            ROOT = ROOT.parent
        """,
        hidden=True,
    ),
    md(
        """
        ## Literature Anchors

        These are not decorative citations. Each one changes what the next
        experiment should measure.
        """
    ),
    code(
        """
        literature = pd.DataFrame(
            [
                {
                    "thread": "Mapper",
                    "source": "Singh, Memoli, Carlsson, 2007",
                    "link": "https://diglib.eg.org/items/d4e204e8-5dd2-4b5d-bccd-5da3c4a1e925",
                    "lesson": "Mapper is a cover-plus-clustering summary of a lens, not a unique latent-space truth.",
                    "project_use": "Report lens, cover, clusterer, coverage, and stability; interpret nodes before graph cycles.",
                },
                {
                    "thread": "Mapper stability",
                    "source": "Carriere and Oudot, 2015",
                    "link": "https://arxiv.org/abs/1511.05823",
                    "lesson": "Mapper output can move under lens and cover perturbations.",
                    "project_use": "Match nodes across seeds/covers and call a node interesting only if it recurs.",
                },
                {
                    "thread": "Activation atlas",
                    "source": "Carter et al., Distill 2019",
                    "link": "https://distill.pub/2019/activation-atlas/",
                    "lesson": "Local activation regions become more interpretable when many patches are aggregated.",
                    "project_use": "Make patch atlases for stable Mapper nodes rather than cherry-picking single patches.",
                },
                {
                    "thread": "TDA for neural networks",
                    "source": "Naitzat, Zhitnikov, Lim, JMLR 2020",
                    "link": "https://jmlr.org/papers/v21/20-345.html",
                    "lesson": "Topology can summarize representation changes, but the claim must be tied to a concrete pipeline.",
                    "project_use": "Use TDA features as controlled comparative statistics, not as standalone visual evidence.",
                },
                {
                    "thread": "Vectorized persistence summaries",
                    "source": "Adams et al., JMLR 2017",
                    "link": "https://jmlr.org/papers/v18/16-337.html",
                    "lesson": "Persistence diagrams can be converted into stable feature vectors for statistical comparison.",
                    "project_use": "Move from diagram storytelling to feature tables, deltas, and confidence intervals.",
                },
                {
                    "thread": "Representation similarity",
                    "source": "Kornblith et al., 2019",
                    "link": "https://arxiv.org/abs/1905.00414",
                    "lesson": "Representations should be compared with explicit similarity metrics and controls.",
                    "project_use": "Compare geometry/TDA/Mapper features across datasets and representations in one table.",
                },
            ]
        )
        display(literature)
        """,
    ),
    md(
        """
        ## What We Already Learned

        The useful signal is narrower than the original project hope, but also
        cleaner.
        """
    ),
    code(
        """
        current_readout = pd.DataFrame(
            [
                {
                    "result": "Universal long-lived H1 is not supported.",
                    "meaning": "Stop optimizing around the old H1 delta as if it were the target.",
                    "next_action": "Use H1 as one diagnostic column among many.",
                },
                {
                    "result": "Metric and preprocessing choices matter a lot.",
                    "meaning": "Raw, unit, and whitened views can tell different stories.",
                    "next_action": "Every table row must record the view and control.",
                },
                {
                    "result": "Mapper graph loop count is not the main readout.",
                    "meaning": "Graph-H1 mostly reflects cover/lens/clustering choices.",
                    "next_action": "Inspect stable node regimes and compare them to nulls.",
                },
                {
                    "result": "Patch back-mapping is essential.",
                    "meaning": "A numeric feature is only useful if we can see what tokens created it.",
                    "next_action": "Build activation-atlas-style panels for stable nodes.",
                },
            ]
        )
        display(current_readout)
        """,
    ),
    md(
        """
        ## Next Experiments

        The next phase should be a small stack, not another sprawling sweep.
        """
    ),
    code(
        """
        experiments = pd.DataFrame(
            [
                {
                    "experiment": "Feature table",
                    "object": "dataset x representation x view",
                    "primary_outputs": "norm/PCA/ID/density/spatial/TDA/Mapper columns",
                    "why": "Turns exploration into comparable fingerprints.",
                },
                {
                    "experiment": "Stable Mapper-node matching",
                    "object": "norm-density Mapper graphs across overlapping bootstrap samples",
                    "primary_outputs": "node recurrence, best-match Jaccard, size variation",
                    "why": "Separates recurring local regimes from cover artifacts.",
                },
                {
                    "experiment": "Patch atlas",
                    "object": "stable Beans Mapper nodes",
                    "primary_outputs": "medoid/diverse source patches plus node metadata",
                    "why": "Checks whether nodes correspond to visual regimes or just statistics.",
                },
                {
                    "experiment": "Control panel",
                    "object": "observed vs channel shuffle vs norm-random directions vs metadata shuffle",
                    "primary_outputs": "observed minus strongest-control deltas",
                    "why": "Keeps positive claims honest.",
                },
                {
                    "experiment": "Thin confirmatory report",
                    "object": "frozen config and frozen decision rule",
                    "primary_outputs": "pass/fail plus failure interpretation",
                    "why": "Prevents the workbench from becoming the claim.",
                },
            ]
        )
        display(experiments)
        """,
    ),
    md(
        """
        ## Confirmatory Question Draft

        This is intentionally conservative. The question is not "does Mapper
        look interesting?" The question is whether observed FLUX token clouds
        have more stable, label-enriched Mapper node tracks than matched
        controls.
        """
    ),
    code(
        """
        config_path = ROOT / "experiment_configs" / "mapper_node_stability_v1.json"
        prereg = json.loads(config_path.read_text())

        display(Markdown(f"**Config:** `{config_path.relative_to(ROOT)}`"))
        display(
            pd.DataFrame(
                [
                    {"field": "primary_question", "value": prereg["primary_question"]},
                    {"field": "primary_statistic", "value": prereg["primary_statistic"]},
                    {"field": "datasets", "value": ", ".join(prereg["target_object"]["datasets"])},
                    {"field": "lens", "value": prereg["mapper"]["lens"]},
                    {"field": "matching", "value": prereg["matching"]["method"]},
                    {"field": "controls", "value": ", ".join(prereg["controls"]["representation_controls"])},
                    {"field": "metadata_control", "value": prereg["controls"]["metadata_control"]},
                ]
            )
        )

        display(pd.DataFrame({"decision_rule_pass": prereg["decision_rule"]["pass"]}))
        """,
    ),
    md(
        """
        ## Realistic Success And Failure Modes

        A positive result would be modest: stable Mapper nodes isolate recurring
        local regimes in the token distribution, and those regimes have
        metadata or visual structure beyond controls.

        A negative result would still be valuable: it would say Mapper is useful
        for intuition and visualization here, but not yet a reliable statistical
        feature for this representation. That would be a clean boundary, not a
        dead end.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)}")
