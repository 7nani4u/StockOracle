import json
import os

from market_briefing import technique_prune as prune


def test_recent_same_size_rewrite_cannot_keep_obsolete_gate(tmp_path):
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps({'rules': {}, 'v': 1}), encoding='utf-8')
    stamp = path.stat()
    assert prune.load_rules(str(path))['v'] == 1
    path.write_text(json.dumps({'rules': {}, 'v': 2}), encoding='utf-8')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert prune.load_rules(str(path))['v'] == 2
