import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 2026S1 COMP8221 Assignment 1\n",
                "\n",
                "## Cold Diffusion for Flower Image Generation: Reversing Deterministic Blur from Scratch with a Time-Conditioned U-Net\n",
                "\n",
                "This is the rebuilt notebook file.\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path("notebooks/03_results_report.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("saved:", out.resolve())
print("exists:", out.exists())
