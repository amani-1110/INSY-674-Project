"""Verify saved study artifacts and update the documentation index."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

root=Path(__file__).resolve().parent.parent
out=root/'output/completed_study'
tuning=json.loads((out/'tuning.json').read_text())
final=json.loads((out/'final_test.json').read_text())
protocol=json.loads((out/'final_test_protocol.json').read_text())
assert final['protocol']==protocol
assert tuning['training_laps']==[5,11]
assert len(tuning['trials'])==18
assert len(tuning['summary'])==4
assert set(final['results'])=={'Planner only','Random Forest','Support vector','Neural network','Equal RF and SVR ensemble','Frozen full-action driver'}
for name,digest in protocol['model_hashes'].items():
    path=out/(name.replace(' ','_')+'.joblib')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
assert hashlib.sha256((root/'output/learned_driver/policy.joblib').read_bytes()).hexdigest()==protocol['full_action_hash']
for result in final['results'].values():
    assert result['valid_lap']==(result['completed_lap'] and result['off_track_events']==0)
    assert 0 < result['elapsed_time'] <= 600.00001
lines=['# Completed model study','',
       'This follow-up completes the tuning, explicit ensemble comparison and fresh third-circuit simulator evaluation that were missing from the earlier analysis.',
       '', 'Only Monza laps 5 and 11 select settings. The two-fold selection scores are optimistic because they are reused for tuning; Spa is the separate frozen-model driving test.',
       '', '| Candidate | Throttle MAE | Brake MAE | Combined selection MAE |', '|---|---:|---:|---:|']
for row in tuning['summary']:
    lines.append(f"| {row['model']} | {row['throttle_mae']:.4f} | {row['brake_mae']:.4f} | {row['combined_mae']:.4f} |")
lines+=['','Selected before Spa: **'+tuning['selected']+'**. Equal ensemble weights were fixed at 0.5/0.5. No settings changed after final testing.', '', '| Spa driver | Elapsed (s) | Departures | Valid completed lap |', '|---|---:|---:|---|']
for name,r in final['results'].items():
    lines.append(f"| {name} | {r['elapsed_time']:.1f} | {r['off_track_events']} | {r['valid_lap']} |")
lines+=['', 'The tuned pedal candidates retain the same planner and steering rules. The frozen full-action driver is evaluated separately. All predefined results are included; final-test results do not select a replacement model.',
        '', 'This is one deterministic simulator condition. It is not independent real-telemetry prediction accuracy or evidence of statistical significance. More sessions and conditions remain useful future extensions.',
        '', 'Evidence: tuning.json (all 18 settings and fold scores), comparison.csv, final_test_protocol.json (recorded before fetching Spa), final_test.json (all runs and unchanged model hashes), data/spa.source.json (pinned geometry provenance).',
        '', 'Implementation: complete_model_study.py. Preserve this initial test record when reproducing the study.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
analysis=root/'output/analysis_pack/ANALYSIS.md'
text=analysis.read_text(encoding='utf-8')
start=text.index('## 7. Remaining work against the rubric')
end=text.index('## 8. Assumptions',start)
text=text[:start]+'''## 7. Completed follow-up against the rubric

The originally missing training-only tuning, explicit ensemble comparison and fresh circuit evaluation have now been performed. See [completed study](../completed_study/REPORT.md) and the repository [project documentation](../../DOCUMENTATION.md).

The follow-up uses 18 seeded candidate settings across three families, whole-lap training-only validation, a fixed equal-weight RF/SVR ensemble, and a frozen-model Spa test defined before retrieving its geometry. All predefined final results are retained. The original lap-14 measurements remain exploratory; the fresh evaluation is a separate closed-loop simulator test, not newly collected real-driver telemetry.

More independent training sessions, more evaluation conditions and a broader error analysis remain useful future improvements. These limitations do not mean the added tuning and final-test experiments were omitted.

'''+text[end:]
analysis.write_text(text,encoding='utf-8')
tex=root/'output/overleaf_documentation/main.tex'
for asset in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',tex.read_text(encoding='utf-8')):
    assert (tex.parent/asset).exists(),asset
with zipfile.ZipFile(root/'output/F1_Project_Overleaf.zip') as z:
    assert z.testzip() is None
assert 'Work still needed for the rubric:' not in (root/'DOCUMENTATION.md').read_text(encoding='utf-8')
print('Verified 18 trials, four candidate comparisons, six final runs, frozen hashes, validity rules, image references and ZIP integrity.')
