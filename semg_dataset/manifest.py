"""Dataset Catalog and Split Manifest Generators with Segregated REST Auditing.

Generates deterministic, audit-grade JSON manifests documenting dataset
inventories, canonical recording metadata, and zero-leakage split partitions.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from semg_dataset.contract import RecordingData
from semg_dataset.splits import SplitManifest


def associate_rest_to_repetition_units(repetitions: np.ndarray) -> np.ndarray:
    """Deterministically associate inter-repetition REST intervals to repetition units.

    Rules:
      - Initial lead-in rest before repetition 1 is assigned unit 0.
      - Each active repetition r (1..6) is assigned unit r.
      - The trailing rest interval immediately following repetition r is assigned to unit r.

    Returns:
        np.ndarray: Integer unit assignment array matching repetitions.shape.
    """
    n = len(repetitions)
    units = np.zeros(n, dtype=np.int32)
    current_unit = 0

    for i in range(n):
        rep_val = int(repetitions[i])
        if rep_val > 0:
            current_unit = rep_val
            units[i] = current_unit
        else:
            units[i] = current_unit

    return units


def _get_current_git_commit() -> str:
    """Retrieve the current Git commit hash, or return fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class DatasetManifestGenerator:
    """Generates dataset_catalog.json with full schema compliance and REST auditing."""

    def generate_catalog(
        self,
        recordings: Sequence[RecordingData],
        output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Generate canonical recording catalog and segregated REST audit summary."""
        records: List[Dict[str, Any]] = []
        total_samples = 0
        total_active_samples = 0
        total_rest_samples = 0

        for rec in recordings:
            view_refined = rec.get_view("refined")
            view_stimulus = rec.get_view("stimulus")

            n_samples = rec.num_samples
            n_active = int(np.sum(rec.restimulus > 0))
            n_rest = int(np.sum(rec.restimulus == 0))

            total_samples += n_samples
            total_active_samples += n_active
            total_rest_samples += n_rest

            rec_dict: Dict[str, Any] = {
                "subject_id": rec.subject_id,
                "exercise_id": rec.exercise_id,
                "num_samples": n_samples,
                "num_channels": rec.num_channels,
                "sampling_rate_hz": rec.sampling_rate_hz,
                "dtype": "float32",
                "alignment": {
                    "status": rec.alignment.status.value,
                    "delta_samples": rec.alignment.delta_samples,
                    "start_anchored_proven": rec.alignment.start_anchored_proven,
                },
                "views": {
                    "refined": {
                        "label_source": "restimulus",
                        "repetition_source": "rerepetition",
                        "view_hash": view_refined.view_hash,
                    },
                    "stimulus": {
                        "label_source": "stimulus",
                        "repetition_source": "repetition",
                        "view_hash": view_stimulus.view_hash,
                    },
                },
            }
            records.append(rec_dict)

        sampling_rate = recordings[0].sampling_rate_hz if recordings else 2000.0
        rest_duration = total_rest_samples / sampling_rate if sampling_rate > 0 else 0.0
        active_duration = total_active_samples / sampling_rate if sampling_rate > 0 else 0.0
        rest_ratio = total_rest_samples / total_samples if total_samples > 0 else 0.0

        rest_summary: Dict[str, Any] = {
            "total_samples": total_samples,
            "active_samples": total_active_samples,
            "rest_samples": total_rest_samples,
            "rest_ratio": rest_ratio,
            "active_duration_seconds": round(active_duration, 4),
            "rest_duration_seconds": round(rest_duration, 4),
        }

        catalog: Dict[str, Any] = {
            "schema_version": "1.0",
            "dataset_name": "NinaPro DB2",
            "rest_summary": rest_summary,
            "records": records,
        }

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, sort_keys=True)

        return catalog


class SplitManifestGenerator:
    """Generates split manifest JSON files conforming to split_schema.json."""

    def export_split_manifest(
        self,
        manifest: SplitManifest,
        output_path: Optional[Path | str] = None,
        timestamp: Optional[str] = None,
        git_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize a SplitManifest to JSON and compute deterministic manifest_hash."""
        ts = (
            timestamp
            if timestamp is not None
            else datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        commit = git_commit if git_commit is not None else _get_current_git_commit()

        d = manifest.to_dict()
        d["created_at"] = ts
        d["git_commit"] = commit

        # Deterministic hash of all fields except manifest_hash
        content_for_hash = json.dumps(
            {k: v for k, v in d.items() if k != "manifest_hash"},
            sort_keys=True,
        ).encode("utf-8")
        manifest_hash = hashlib.sha256(content_for_hash).hexdigest()
        d["manifest_hash"] = manifest_hash

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, sort_keys=True)

        return d
