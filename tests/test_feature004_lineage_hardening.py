"""Adversarial scientific lineage tests for Feature 004 real validation.

Add to repository tests/ after applying feature004_lineage_hardening.patch.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from semg_baseline.lineage import (
    build_verified_scientific_lineage,
    compute_dataset_view_hash,
    compute_file_sha256,
    verify_feature_003_frozen_contracts,
)


def _repo(tmp_path: Path):
    acceptance = tmp_path / 'reports' / 'feature_003_acceptance'
    (acceptance / 'raw').mkdir(parents=True)
    alignment = acceptance / 'raw' / 'alignment_audit.jsonl'
    alignment.write_text('accepted alignment\n')
    acceptance_manifest = acceptance / 'manifest.sha256'
    acceptance_manifest.write_text(
        f'{compute_file_sha256(alignment)}  raw/alignment_audit.jsonl\n'
    )
    manifests = tmp_path / 'data' / 'manifests' / 'ninapro_db2'
    manifests.mkdir(parents=True)
    raw = manifests / 'intake_manifest.json'
    raw.write_text('{"real":true}\n')
    split = manifests / 'splits_within_subject.json'
    split.write_text('{"protocol_id":"within_subject"}\n')
    return acceptance_manifest, alignment, raw, split


def _record():
    return SimpleNamespace(
        subject_id='S1', exercise_id='E1',
        get_view=lambda name: SimpleNamespace(view_hash='1' * 64),
    )


def _params(tmp_path, recs=None):
    acceptance, alignment, raw, split = _repo(tmp_path)
    recs = recs if recs is not None else [_record()]
    return {
        'repo_root': tmp_path,
        'recordings': recs,
        'split_manifest_path': split,
        'expected_dataset_view_sha256': compute_dataset_view_hash(recs),
        'expected_split_manifest_sha256': compute_file_sha256(split),
        'expected_acceptance_sha256': compute_file_sha256(acceptance),
        'expected_raw_manifest_sha256': compute_file_sha256(raw),
        'expected_alignment_sha256': compute_file_sha256(alignment),
        'protocol_id': 'within_subject',
        'partition_id': 'S1',
        'git_commit': '1' * 40,
    }


def test_empty_manifest_is_fail_closed(tmp_path):
    acceptance, *_ = _repo(tmp_path)
    acceptance.write_text('')
    ok, problems = verify_feature_003_frozen_contracts(tmp_path)
    assert not ok
    assert any('no valid' in p for p in problems)


def test_malformed_manifest_is_fail_closed(tmp_path):
    acceptance, *_ = _repo(tmp_path)
    acceptance.write_text('broken-line\n')
    ok, problems = verify_feature_003_frozen_contracts(tmp_path)
    assert not ok
    assert any('Malformed' in p for p in problems)


def test_escape_path_is_fail_closed(tmp_path):
    acceptance, *_ = _repo(tmp_path)
    acceptance.write_text(f'{"0" * 64}  ../../secret-file\n')
    ok, problems = verify_feature_003_frozen_contracts(tmp_path)
    assert not ok
    assert any('Unsafe' in p for p in problems)


def test_duplicate_entry_is_fail_closed(tmp_path):
    acceptance, alignment, *_ = _repo(tmp_path)
    line = f'{compute_file_sha256(alignment)}  raw/alignment_audit.jsonl\n'
    acceptance.write_text(line + line)
    ok, problems = verify_feature_003_frozen_contracts(tmp_path)
    assert not ok
    assert any('Duplicate' in p for p in problems)


def test_unpinned_manifest_tampering_is_fail_closed(tmp_path):
    params = _params(tmp_path)
    manifest = tmp_path / 'reports' / 'feature_003_acceptance' / 'manifest.sha256'
    manifest.write_text('')
    ok, problems = verify_feature_003_frozen_contracts(
        tmp_path, expected_acceptance_sha256=params['expected_acceptance_sha256']
    )
    assert not ok
    assert any('differs' in p for p in problems)


def test_real_lineage_positive(tmp_path):
    params = _params(tmp_path)
    lineage = build_verified_scientific_lineage(**params)
    assert lineage.dataset_view_hash == params['expected_dataset_view_sha256']
    assert lineage.split_manifest_hash == params['expected_split_manifest_sha256']


def test_real_lineage_rejects_wrong_view(tmp_path):
    params = _params(tmp_path)
    params['expected_dataset_view_sha256'] = 'a' * 64
    with pytest.raises(RuntimeError, match='dataset view'):
        build_verified_scientific_lineage(**params)


def test_real_lineage_rejects_wrong_split(tmp_path):
    params = _params(tmp_path)
    params['expected_split_manifest_sha256'] = 'b' * 64
    with pytest.raises(RuntimeError, match='split manifest'):
        build_verified_scientific_lineage(**params)


def test_real_lineage_rejects_wrong_intake(tmp_path):
    params = _params(tmp_path)
    params['expected_raw_manifest_sha256'] = 'c' * 64
    with pytest.raises(RuntimeError, match='raw manifest'):
        build_verified_scientific_lineage(**params)


def test_real_lineage_rejects_synthetic_partition(tmp_path):
    params = _params(tmp_path)
    params['partition_id'] = 'SYNTH_S1'
    with pytest.raises(ValueError, match='Invalid'):
        build_verified_scientific_lineage(**params)


def test_real_lineage_rejects_non_git_commit(tmp_path):
    params = _params(tmp_path)
    params['git_commit'] = 'unknown'
    with pytest.raises(RuntimeError, match='Git commit'):
        build_verified_scientific_lineage(**params)
