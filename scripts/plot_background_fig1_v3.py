"""Compact Fig. 1 assets: qualitative Decode dataflow and canonical capacity."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, FancyArrowPatch
from om3dthermal.serving.mixed_phase_e2e import resolve_conventional_hbm_backend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/background_fig1_v3'
FORMAL = ROOT / 'runs/formal_long_context_v3'
SIZE = (3.65, 2.4)
PURPLE, ORANGE, BLUE, INK = '#806589', '#CB9454', '#6687A0', '#343434'


def font(size=9, bold=False):
    f = FontProperties(fname=r'C:\Windows\Fonts\timesbd.ttf' if bold else r'C:\Windows\Fonts\times.ttf', size=size)
    f.set_family('Times New Roman')
    f.set_weight('bold' if bold else 'normal')
    return f


def text(ax, x, y, value, size=9, bold=False, **kwargs):
    return ax.text(x, y, value, fontproperties=font(size, bold),
                   ha=kwargs.pop('ha', 'center'), va=kwargs.pop('va', 'center'), **kwargs)


def save(fig, stem):
    for ext in ('pdf', 'svg'):
        target = OUT / f'{stem}.{ext}'
        metadata = {'Date': None} if ext == 'svg' else {'CreationDate': None, 'ModDate': None}
        try:
            fig.savefig(target, metadata=metadata)
        except PermissionError:
            target = OUT / f'{stem}_new.{ext}'
            print(f'Locked output: {target}')
            fig.savefig(target, metadata=metadata)
        if ext == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    fig.savefig(OUT / f'{stem}_preview.png', dpi=200)
    plt.close(fig)


def movement():
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=.025, right=.975, bottom=.06, top=.96)
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis('off')
    for x, fill in ((.10, '#E4EDF3'), (7.10, '#EEE6EF')):
        ax.add_patch(Rectangle((x, 1.22), 2.8, 3.62, facecolor=fill, edgecolor=INK, linewidth=.9))
    text(ax, 1.50, 4.33, 'GPU', 12, True)
    text(ax, 8.50, 4.33, 'HBM', 12, True)
    for y in (3.20, 2.69):
        for x in (.50, 1.20, 1.90):
            ax.add_patch(Rectangle((x, y), .50, .35, facecolor=BLUE, edgecolor=INK, linewidth=.45))
    text(ax, 1.50, 1.87, 'Compute', 10)
    ax.add_patch(Rectangle((7.39, 2.85), 2.22, .73, facecolor=PURPLE, edgecolor=INK, linewidth=.5))
    text(ax, 8.50, 3.215, 'Model Weights', 9, color='white')
    ax.add_patch(Rectangle((7.39, 1.77), 2.22, .84, facecolor='#DCCCE1', edgecolor=INK, linewidth=.5))
    text(ax, 8.50, 2.19, 'Growing\nKV Cache', 9)
    text(ax, 5, 4.62, 'Repeated Weight /\nKV Transfer', 9, True)
    for y in (3.66, 3.10, 2.54):
        ax.add_patch(FancyArrowPatch((7.03, y), (2.97, y), arrowstyle='-|>', mutation_scale=10, color=ORANGE, linewidth=1.6))
    ax.add_patch(FancyArrowPatch((2.97, 2.02), (7.03, 2.02), arrowstyle='-|>', mutation_scale=9, color=ORANGE, linewidth=.9))
    text(ax, 5, 1.18, 'GPU–Memory\nBottleneck', 10, True, color='#894E43')
    text(ax, 5, .30, 'Autoregressive Decode', 9)
    save(fig, 'fig1a_data_movement')


def capacity():
    with (FORMAL / 'capacity_audit.csv').open(encoding='utf-8', newline='') as stream:
        source = list(csv.DictReader(stream))
    models = ('Llama-3.1-8B', 'Qwen2.5-32B')
    rows = []
    cap = resolve_conventional_hbm_backend(ROOT).capacity_bytes
    for model in models:
        for batch in (1, 8, 32):
            r = next(r for r in source if r['model'] == model and int(r['H']) == 126000 and int(r['B']) == batch)
            assert int(r['P']) == 128 and int(r['G']) == 32
            assert float(r['HBM_capacity_bytes']) == cap
            weights, kv = float(r['weight_bytes']), float(r['final_KV_bytes'])
            rows.append(dict(model=model, H=126000, P=128, G=32, B=batch,
                             weight_GB=weights/1e9, KV_GB=kv/1e9, total_GB=(weights+kv)/1e9,
                             HBM_usable_GB=cap/1e9, overflow=weights+kv>cap))
    with (OUT/'capacity_data.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=SIZE)
    x = [0, 1, 2, 3.5, 4.5, 5.5]
    ax.bar(x, [r['weight_GB'] for r in rows], .70, color=PURPLE, edgecolor=INK, linewidth=.45, label='Model Weights', zorder=3)
    ax.bar(x, [r['KV_GB'] for r in rows], .70, bottom=[r['weight_GB'] for r in rows], color=ORANGE, edgecolor=INK, linewidth=.45, label='KV Cache', zorder=3)
    for xx, r in zip(x, rows):
        text(ax, xx, r['total_GB']+28, f"{r['total_GB']:.1f}", 8, va='bottom')
    ax.axhline(cap/1e9, color='#894E43', linestyle='--', linewidth=.8, zorder=4)
    text(ax, .98, 225, 'H200 HBM Capacity', 7.5, ha='right', color='#894E43', transform=ax.get_yaxis_transform(),
         bbox=dict(facecolor='white', edgecolor='none', pad=.8), zorder=5)
    ax.set_xticks(x, ['B1', 'B8', 'B32']*2)
    for center, model in zip((1, 4.5), models):
        text(ax, center, -.22, model, 8.5, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1280)
    ax.set_ylabel('Working-set Capacity (GB)', fontproperties=font(9))
    ax.tick_params(direction='in', top=True, right=True, width=.65, pad=3)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontproperties(font(8.5))
    ax.grid(axis='y', linestyle='--', color='#dddddd', linewidth=.4)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, prop=font(8.5), columnspacing=1)
    fig.subplots_adjust(left=.17, right=.985, bottom=.24, top=.84)
    save(fig, 'fig1b_capacity')
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Byte hashes are integrity checks only, not E2E result parsing.
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in FORMAL.rglob('*') if p.is_file()}
    plt.rcParams.update({'font.family':'Times New Roman', 'svg.fonttype':'none', 'pdf.fonttype':42,
                         'svg.hashsalt':'background-fig1-v3', 'axes.linewidth':.65, 'figure.facecolor':'white'})
    movement(); rows = capacity()
    assert all(hashlib.sha256(p.read_bytes()).digest() == h for p,h in before.items())
    (OUT/'README.md').write_text('''# Compact Background Fig. 1

The left asset is a conceptual GPU–HBM data-movement schematic, not to scale. Repeated leftward arrows indicate weight/KV reads to the GPU; the smaller return arrow indicates KV/state writes, not weight modification. Arrow widths and block sizes encode no quantitative ratios. Autoregressive Decode and Growing KV Cache distinguish the schematic from a generic memory-wall diagram.

The right asset uses current formal benchmark definitions: Llama-3.1-8B and Qwen2.5-32B; cached H=126000; P=128; G=32; B={1,8,32}. Weights and final KV(H+P+G) come from runs/formal_long_context_v3/capacity_audit.csv. Decimal GB = 1e9 bytes. Formal capacity preflight additionally includes workspace/runtime state. The H200 HBM reference comes from the current resolved conventional-HBM backend (resolve_conventional_hbm_backend) and is checked against the audit capacity field, rather than hard-coded.

Both independent assets are 3.65 × 2.40 inches with Times New Roman, vector PDF/SVG text and matching palettes. No title, panel letter, caption, solution architecture, AI/roofline, performance or thermal numbers appear. PNGs are previews only. Formal benchmark data are byte-identical before/after generation. No benchmark, thermal solve or placement optimization is run. Existing figures and manuscript remain unchanged.

Generate in the om3dthermal Conda environment:
python scripts/plot_background_fig1_v3.py
''', encoding='utf-8')
    print(json.dumps({'capacity':rows,'formal_files_unchanged':len(before)},indent=2))

if __name__ == '__main__':
    main()

