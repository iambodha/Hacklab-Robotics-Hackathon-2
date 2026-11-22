"""Publish a compact, reproducible report from an already completed learning run.

python -m tasks.kitting.report DATA_ROOT/r5-kitsim/runs/RUN --out docs/kitting
Requires matplotlib only for report generation, not for training or real operation.
"""
import argparse
import json
import shutil
from pathlib import Path


def write(run: Path, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    if (run/'results.json').exists():
        result=json.loads((run/'results.json').read_text())
    else:
        result=json.loads((run/'recovered_results.json').read_text())
        result['curve']=json.loads((Path(__file__).resolve().parents[2]/'docs/kitting/historical_curve.json').read_text())
        result['minutes']='interrupted; see journal'
        result['selection']={'selected_iteration':4,'original_selected_version':11,'original_restored_version':14,
                             'recovered_version':result['final_version']}
        result['history_note']='Historical curve transcribed from completed console output after external drive disconnect; held-out and replay metrics freshly verified locally'
        for label,index in [('generic',0),('selected',9)]:
            for view in ['overview','top']:
                source=run/'views'/f'{label}_seed10000_{view}.jpg'
                if source.exists():
                    shutil.copyfile(source,run/'views'/f'iter{index:02d}_seed10000_{view}.jpg')
    curve=result['curve']
    out.mkdir(parents=True,exist_ok=True)
    images=out/'images'; images.mkdir(exist_ok=True)
    (out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    fig, axes=plt.subplots(2,1,figsize=(7.2,5.5),sharex=True,layout='constrained')
    it=[r['iteration'] for r in curve]
    axes[0].plot(it,[100*r['success_rate'] for r in curve],'o-',color='#254b72',label='all assignments')
    for c in ('screw','cap','box'):
        axes[0].plot(it,[100*r['per_class'][c]['success_rate'] for r in curve],'.--',alpha=.65,label=c)
    axes[0].set(ylabel='Placement success (%)',ylim=(-3,103),title='Kitting in MuJoCo — fixed randomized evaluation seeds')
    axes[0].legend(ncol=4,fontsize=8)
    axes[1].plot(it,[r['mean_error_mm'] for r in curve],'o-',color='#9b542e')
    axes[1].set(ylabel='Centre error (mm)\nplaced objects only',xlabel='Evaluation index',xticks=it)
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.savefig(images/'learning_curve.png',dpi=140)
    plt.close(fig)
    views=sorted((run/'views').glob('iter*_overview.jpg'))
    if len(views) < 2 or not views[-1].name.startswith(f"iter{curve[-1]['iteration']:02d}_"):
        raise ValueError('Final iteration view missing; do not label an earlier view as the final kit')
    for label,path in [('initial',views[0]),('final',views[-1])]:
        shutil.copyfile(path,images/f'{label}_cpu_overview.jpg')
    rows=['# Measured kitting results','',f"Source: `{run}`. Renderer: `{result.get('renderer','unknown')}`. Runtime accounting: {result['minutes']}.",
          '', 'This is a scripted heuristic search with privileged simulation diagnoses. No real-arm success is claimed.',
          '', result.get('history_note','All curve rows come from the original run records.'), '', 'The exported profile is selected by fixed evaluation success, with lower placed error breaking ties, before the learned held-out score is read. The recovery run replays selection at index 9 and uses a new config version lineage.', '', '![Learning curve](images/learning_curve.png)','',
          '| iteration | version | placed / attempted | complete kits | placed error mm | screw | cap | box |',
          '|---|---|---|---|---|---|---|---|']
    for r in curve:
        placed=sum(v['placed'] for v in r['per_class'].values())
        rows.append(f"| {r['iteration']} | {r['version']} | {placed}/{r['objects']} ({100*r['success_rate']:.1f}%) | "
                    f"{r['kits_complete']}/{r['episodes']} | {r['mean_error_mm']} | "+
                    ' | '.join(f"{100*r['per_class'][c]['success_rate']:.1f}%" for c in ('screw','cap','box'))+' |')
    rows+=['','Three kits per fixed-seed iteration are a small sample, not a confidence interval. The same kits repeat; '
           'iterations are not independent trials. Read the held-out comparison separately. Per-class rates describe '
           'executed objects; overall rates include aborted kits as failures.','', '## Held-out comparison','',
           '| profile / condition | placed rate | complete kits | mean placed error mm | aborted kits |',
           '|---|---|---|---|---|']
    for label,m in (result['test'] or {}).items():
        if isinstance(m,dict):
            rows.append(f"| {label} | {100*m['success_rate']:.1f}% | {m['kits_complete']}/{m['episodes']} | "
                        f"{m['mean_error_mm']} | {m['aborted']} |")
    rows+=['','## Changes accepted during search (later changes can be reverted)','']
    rows += [f"- After iteration {r['iteration']}: {change}" for r in curve for change in r['applied']]
    rows+=['','## Images and provenance','', '![Initial CPU overview](images/initial_cpu_overview.jpg)',
           '![Final CPU overview](images/final_cpu_overview.jpg)','',
           'These are final states of the first evaluation seed before/after training. CPU projections omit arm '
           'occlusion and shadows. The separately stored `images/devin_gl_overview.png` was generated by Devin '
           'before this takeover at 640×480; the perception test downsamples it to deployment resolution (299×224).',
           '', 'The original episode reports and proposal evidence were on the external drive when it disconnected. Fresh recovery episode records are under the local run '
           'directory, and all 22 fresh episodes are archived in `evidence/recovery_episodes.jsonl.gz`. Nominal kits repeat the same scene; they are not independent robustness trials. `measurements.json` is a small in-repo copy of the aggregate data. See README and REAL_RUN '
           'for transfer risks and exact commands.','']
    (out/'RESULTS.md').write_text('\n'.join(rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path); ap.add_argument('--out',type=Path,default=Path('docs/kitting'))
    a=ap.parse_args(); write(a.run,a.out)

if __name__=='__main__':
    main()
