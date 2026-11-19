"""Export scientific curves only from all four completed long-run observations."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('independent_audit', Path(__file__).with_name('audit_mixer_long_campaign.py'))
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    receipt = audit.read(args.audit)
    audit.need(receipt['status'] == 'FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY'
               and receipt['tests_checked'] and receipt['checkpoints_checked']
               and receipt['campaign_id'] == audit.CAMPAIGN, 'Complete audited observations required')
    audit.need(receipt['identity'] == audit.read(args.goal_root / 'formal/identity.json'), 'Figure/audit campaign identity differs')
    progress = {}
    for arm in audit.ARMS:
        directory = args.goal_root / 'formal' / arm
        audit.need(receipt['source_receipt_sha256'][arm] ==
                   {name:audit.sha(directory / name) for name in ('TRAIN_COMPLETE.json','progress.json')},
                   'Figure source bytes differ from the completed audit')
        complete = audit.read(directory / 'TRAIN_COMPLETE.json')
        audit.need(complete['identity'] == receipt['identity'] and complete['progress'] == audit.read(directory / 'progress.json'),
                   'Figure source identity/progress differs')
        progress[arm] = complete['progress']
    for arm, value in progress.items():
        summary = audit.audit_progress(value, arm, complete=True)
        audit.need(summary['final_validation'] == receipt['fixed_epoch35_validation'][arm], 'Audit/curve observations differ')
    audit.paired_controls(progress, complete=True)
    audit.need(not args.output.exists(), 'Use a new figure directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    args.output.mkdir(parents=True)
    colors = dict(zip(audit.ARMS, ('#4c78a8','#f58518','#54a24b','#b279a2')))
    styles = {a:('--' if a.endswith('_frozen') else '-') for a in audit.ARMS}
    outputs = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png','pdf'):
            path = args.output / (name + '.' + extension)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append(dict(file=path.name,sha256=audit.sha(path)))
        plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for axis, metric, label in zip(axes, ('hit_rate@10','ndcg@10','weighted_score'), ('HR@10','NDCG@10','0.8 NDCG@10 + 0.2 HR@10')):
        for arm, value in progress.items():
            rows = value['validation']
            axis.plot([r['decoder_epoch'] for r in rows],[r['metrics'][metric] for r in rows],
                      color=colors[arm],linestyle=styles[arm],marker='o',markersize=3,label=arm)
        axis.set(xlabel='Recommender epoch',ylabel=label,xticks=range(0,36,5))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-budget validation (SID-level, beam 32; seed 2024)',y=1.02)
    save(fig,'validation_curves')
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for arm, value in progress.items():
        rows = value['training']; r = t = 0; series = []
        for row in rows:
            if row['phase'] == 'recommender':
                r += 1
                axes[0,0].plot([r],[row['loss_means']['total_loss']],marker='.',color=colors[arm])
            else:
                t += 6
                axes[0,1].plot([t],[row['loss_means']['total_loss']],marker='o',color=colors[arm])
            series.append((r,t,row))
        # SID is measured after each committed phase, so the x-axis is the stage.
        for axis, values in ((axes[1,0],[row['formal_SID_change_since_initial']['changed_items']/12101 for row in rows]),
                             (axes[1,1],[row['formal_SID_statistics']['full_sid']['collision_rate'] for row in rows])):
            axis.plot(range(1,len(rows)+1),values,color=colors[arm],linestyle=styles[arm],label=arm)
    for axis, xlabel, ylabel in ((axes[0,0],'Recommender epoch','Recommender mean total loss'),
            (axes[0,1],'Cumulative Tokenizer updates','Tokenizer mean total loss'),
            (axes[1,0],'Committed stage (35 Frozen / 45 Joint)','Fraction of items with changed SID'),
            (axes[1,1],'Committed stage (35 Frozen / 45 Joint)','Fraction of items in SID collisions')):
        axis.set(xlabel=xlabel,ylabel=ylabel); axis.grid(alpha=.2)
    axes[1,0].legend(fontsize=8)
    save(fig,'loss_and_sid_curves')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for arm, value in progress.items():
        rs = [r for r in value['training'] if r['phase'] == 'recommender']
        axes[0].plot([r['logical_index']*104 for r in rs],[r['lr_end'] for r in rs],
                     color=colors[arm],linestyle=styles[arm],label=arm)
        ts = [r for r in value['training'] if r['phase'] == 'tokenizer']
        if ts:
            points = [ts[0]['tokenizer_schedule_start']] + [r['tokenizer_schedule_end'] for r in ts]
            axes[1].plot([p['optimizer_step'] for p in points],[p['gumbel_noise_scale'] for p in points],
                         color=colors[arm],linestyle=styles[arm],marker='o',label=arm)
    axes[0].set(xlabel='Cumulative recommender updates',ylabel='Recommender learning rate')
    axes[1].set(xlabel='Cumulative Tokenizer updates',ylabel='Gumbel noise scale',ylim=(.15,1.05))
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    save(fig,'update_schedules')
    audit.write(args.output / 'FIGURES.json',dict(status='FIGURES_FROM_COMPLETED_OBSERVATIONS',
        campaign_id=audit.CAMPAIGN,artifact_audit_sha256=audit.sha(args.audit),plotter_sha256=audit.sha(Path(__file__)),
        files=outputs,seed=2024,confidence_intervals=False,goal_complete=False))
    print(json.dumps(dict(figures=len(outputs),output=str(args.output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
