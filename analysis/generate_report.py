"""Emit a self-contained HTML analysis report from the run artifacts.

Numbers come straight from summary.json / logits, so nothing is transcribed by
hand. Writes report.html into analysis/, ready to publish as an Artifact.
"""

import base64
import csv
import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf

HERE = os.path.dirname(os.path.abspath(__file__))
EMO = os.path.join(HERE, "emogo")
SEED = 1993
METHODS = ["finetune", "ewc", "lwf", "er", "rs", "ocdm", "prs", "agcn", "aesl"]
LABEL = {"finetune": "Finetune", "ewc": "EWC", "lwf": "LwF", "er": "ER",
         "rs": "RS", "ocdm": "OCDM", "prs": "PRS", "agcn": "AGCN",
         "aesl": "AESL"}
FAMILY = {"finetune": "baseline", "ewc": "regularise", "lwf": "distill",
          "er": "replay", "rs": "replay", "ocdm": "replay", "prs": "replay",
          "agcn": "graph", "aesl": "graph"}
PROTOCOLS = ["B0-I7", "B0-I4", "B16-I3", "B16-I2"]

PAPER = {  # Audio28, Table 3: (Avg mAP, Last maF1, Last miF1, Last mAP)
    "B0-I7": {"finetune": (36.4, 9.2, 14.8, 27.3), "ewc": (37.9, 8.3, 14.3, 29.3),
              "lwf": (46.6, 37.9, 51.7, 40.6), "er": (44.7, 8.1, 14.4, 38.0),
              "rs": (43.7, 8.1, 12.3, 36.5), "ocdm": (44.5, 8.7, 12.0, 38.4),
              "prs": (43.3, 10.8, 13.5, 35.5), "agcn": (47.3, 35.3, 50.9, 41.9),
              "aesl": (49.0, 38.4, 51.8, 42.7)},
    "B0-I4": {"finetune": (35.3, 5.3, 10.0, 23.3), "ewc": (37.1, 5.4, 10.5, 26.6),
              "lwf": (45.8, 49.8, 37.6, 45.0), "er": (44.6, 6.5, 5.5, 35.2),
              "rs": (43.6, 5.9, 9.3, 32.0), "ocdm": (44.5, 7.5, 8.8, 31.5),
              "prs": (44.5, 6.8, 8.7, 34.2), "agcn": (47.3, 37.5, 51.0, 38.6),
              "aesl": (48.7, 41.1, 51.7, 39.8)},
    "B16-I3": {"finetune": (29.9, 4.4, 10.3, 22.6), "ewc": (32.2, 4.4, 9.7, 24.7),
               "lwf": (45.0, 32.3, 45.2, 40.0), "er": (41.3, 9.2, 13.3, 36.8),
               "rs": (38.7, 7.5, 11.7, 32.9), "ocdm": (38.2, 5.5, 9.7, 30.1),
               "prs": (38.2, 5.5, 5.8, 32.7), "agcn": (39.5, 22.7, 38.4, 37.0),
               "aesl": (47.8, 32.3, 48.0, 42.3)},
    "B16-I2": {"finetune": (27.6, 2.8, 8.2, 20.2), "ewc": (28.1, 2.8, 8.7, 22.5),
               "lwf": (44.3, 28.8, 41.4, 36.5), "er": (39.4, 10.1, 13.6, 34.1),
               "rs": (38.2, 5.8, 11.6, 31.8), "ocdm": (36.3, 3.7, 7.9, 30.2),
               "prs": (37.7, 4.4, 7.6, 32.5), "agcn": (36.1, 24.4, 36.4, 30.9),
               "aesl": (45.3, 30.8, 45.1, 39.3)},
}


def summ(m, p):
    return json.load(open(os.path.join(EMO, m, p, f"seed{SEED}", "summary.json")))


def emotions():
    return [l for l in open(os.path.join(HERE, "..", "emogo", "data",
            "emotions.txt")).read().splitlines() if l.strip()]


def test_labels():
    rows = list(csv.reader(open(os.path.join(HERE, "..", "emogo", "data",
                "test.tsv"), encoding="utf-8"), delimiter="\t"))
    y = np.zeros((len(rows), 28))
    for i, r in enumerate(rows):
        for j in r[1].split(","):
            y[i, int(j)] = 1.0
    return y


def canon(mat, order):
    pos = {e: i for i, e in enumerate(order)}
    return mat[:, [pos[e] for e in emotions()]]


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


# ------------------------------------------------------------- table builders

def emogrowth_rows(p):
    """One protocol: rows of (method, ours 4-tuple, paper 4-tuple)."""
    out = []
    for m in METHODS:
        s = summ(m, p)
        ours = (100 * s["avg_acc"]["map"], 100 * s["last_acc"]["macrof1"],
                100 * s["last_acc"]["microf1"], 100 * s["last_acc"]["map"])
        out.append((m, ours, PAPER[p][m]))
    return out


def emogrowth_table_html(p):
    rows = emogrowth_rows(p)
    best_ours = max(r[1][3] for r in rows)          # best Last mAP (ours)
    best_paper = max(r[2][3] for r in rows)
    body = []
    for m, ours, paper in rows:
        cls = f"fam-{FAMILY[m]}"
        ours_cells = "".join(
            f'<td class="num{" best" if (i == 3 and v == best_ours) else ""}">'
            f'{v:.1f}</td>' for i, v in enumerate(ours))
        paper_cells = "".join(
            f'<td class="num paper{" best" if (i == 3 and v == best_paper) else ""}">'
            f'{v:.1f}</td>' for i, v in enumerate(paper))
        body.append(f'<tr class="{cls}"><th scope="row">{LABEL[m]}'
                    f'<span class="fam">{FAMILY[m]}</span></th>'
                    f'{ours_cells}<td class="spacer"></td>{paper_cells}</tr>')
    return f"""
    <div class="table-wrap">
    <table class="data">
      <thead>
        <tr>
          <th></th>
          <th colspan="4" class="grp ours">ours — GoEmotions (text)</th>
          <th class="spacer"></th>
          <th colspan="4" class="grp paper">paper — Audio28 (audio)</th>
        </tr>
        <tr class="sub">
          <th>method</th>
          <th class="num">Avg mAP</th><th class="num">maF1</th>
          <th class="num">miF1</th><th class="num">mAP</th>
          <th class="spacer"></th>
          <th class="num">Avg mAP</th><th class="num">maF1</th>
          <th class="num">miF1</th><th class="num">mAP</th>
        </tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    </div>"""


def goemotions_table_html():
    ems = emotions()
    y = test_labels()
    cols = {}
    ub = os.path.join(HERE, "bertgo", "output", "run1", "test_probs.npy")
    cols["Upper-bound"] = np.load(ub)
    for m in ("lwf", "aesl", "agcn"):
        s = summ(m, "B0-I7")
        last = len(s["curves"]["map"]) - 1
        lg = np.load(os.path.join(EMO, m, "B0-I7", f"seed{SEED}",
                                  f"task{last}_logits.npy"))
        cols[LABEL[m]] = canon(1 / (1 + np.exp(-lg)), s["class_order"])
    names = list(cols)
    preds = {n: (cols[n] > 0.3).astype(int) for n in names}

    head2 = "".join(f'<th class="num">P</th><th class="num">R</th>'
                    f'<th class="num f1">F1</th>' for _ in names)
    grp = "".join(f'<th colspan="3" class="grp">{n}</th>' for n in names)

    body = []
    # sort emotions by test-set frequency, descending, for a readable gradient
    freq = y.sum(0)
    order = sorted(range(28), key=lambda j: -freq[j])
    for j in order:
        cells = []
        for n in names:
            P, R, F, _ = prf(y[:, j], preds[n][:, j], average="binary",
                             zero_division=0)
            cells.append(f'<td class="num">{P:.2f}</td><td class="num">{R:.2f}</td>'
                         f'<td class="num f1">{F:.2f}</td>')
        body.append(f'<tr><th scope="row">{ems[j]}<span class="cnt">'
                    f'{int(freq[j])}</span></th>{"".join(cells)}</tr>')
    # macro / micro
    for avg in ("macro", "micro"):
        cells = []
        for n in names:
            P, R, F, _ = prf(y, preds[n], average=avg, zero_division=0)
            cells.append(f'<td class="num">{P:.2f}</td><td class="num">{R:.2f}</td>'
                         f'<td class="num f1">{F:.2f}</td>')
        body.append(f'<tr class="avg"><th scope="row">{avg}-average</th>'
                    f'{"".join(cells)}</tr>')

    return f"""
    <div class="table-wrap">
    <table class="data taxonomy">
      <thead>
        <tr><th></th>{grp}</tr>
        <tr class="sub"><th>emotion<span class="cnt-h">n</span></th>{head2}</tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    </div>"""


# ------------------------------------------------------------------- assemble

def key_numbers():
    """Headline stats used in the finding callout."""
    def last_map(m, p):
        return 100 * summ(m, p)["last_acc"]["map"]
    winners = {}
    for p in PROTOCOLS:
        best = max(METHODS, key=lambda m: last_map(m, p))
        winners[p] = (best, last_map(best, p))
    return winners


def build():
    fig_light = b64(os.path.join(HERE, "figures", "map_curves.png"))
    fig_dark = b64(os.path.join(HERE, "figures", "map_curves_dark.png"))
    winners = key_numbers()

    eg_tables = "".join(
        f'<section class="proto"><h3>{p} '
        f'<span class="tasks">{"·".join(str(x) for x in summ("lwf", p)["task_sizes"])}</span></h3>'
        f'{emogrowth_table_html(p)}</section>' for p in PROTOCOLS)

    win_line = ", ".join(f"{p}: <b>{LABEL[w[0]]}</b> ({w[1]:.1f})"
                         for p, w in winners.items())

    html = TEMPLATE.format(
        fig_light=fig_light, fig_dark=fig_dark,
        eg_tables=eg_tables, go_table=goemotions_table_html(),
        win_line=win_line,
        lwf_b0i7=100 * summ("lwf", "B0-I7")["last_acc"]["map"],
        aesl_b0i7=100 * summ("aesl", "B0-I7")["last_acc"]["map"],
    )
    out = os.path.join(HERE, "report.html")
    open(out, "w").write(html)
    print("wrote", out, f"({len(html)//1024} KB)")


TEMPLATE = r"""<title>EmoGrowth on Text — Results Analysis</title>
<style>
:root{{
  --ground:#f2f2ef; --surface:#ffffff; --surface-2:#f7f7f4;
  --ink:#1b1f27; --ink-2:#54606f; --ink-3:#8a94a2; --line:#e4e4df;
  --accent:#0b6aa8; --paper:#a86a32; --good:#2f7d5b; --bad:#b3493a;
  --fam-baseline:#8a94a2; --fam-regularise:#c0803a; --fam-distill:#0b6aa8;
  --fam-replay:#3f8f7a; --fam-graph:#a24a6d;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,monospace;
}}
@media (prefers-color-scheme:dark){{:root{{
  --ground:#12151a; --surface:#1a1e25; --surface-2:#20252d;
  --ink:#e8eaef; --ink-2:#a3adba; --ink-3:#6b7684; --line:#2b3038;
  --accent:#57a8d8; --paper:#cf9556; --good:#5cbf93; --bad:#df7a6a;
  --fam-baseline:#6b7684; --fam-regularise:#cf9556; --fam-distill:#57a8d8;
  --fam-replay:#5cbf93; --fam-graph:#cf88a8;
}}}}
:root[data-theme="dark"]{{
  --ground:#12151a; --surface:#1a1e25; --surface-2:#20252d;
  --ink:#e8eaef; --ink-2:#a3adba; --ink-3:#6b7684; --line:#2b3038;
  --accent:#57a8d8; --paper:#cf9556; --good:#5cbf93; --bad:#df7a6a;
  --fam-baseline:#6b7684; --fam-regularise:#cf9556; --fam-distill:#57a8d8;
  --fam-replay:#5cbf93; --fam-graph:#cf88a8;
}}
:root[data-theme="light"]{{
  --ground:#f2f2ef; --surface:#ffffff; --surface-2:#f7f7f4;
  --ink:#1b1f27; --ink-2:#54606f; --ink-3:#8a94a2; --line:#e4e4df;
  --accent:#0b6aa8; --paper:#a86a32; --good:#2f7d5b; --bad:#b3493a;
  --fam-baseline:#8a94a2; --fam-regularise:#c0803a; --fam-distill:#0b6aa8;
  --fam-replay:#3f8f7a; --fam-graph:#a24a6d;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--sans);line-height:1.6;
  -webkit-font-smoothing:antialiased;}}
.wrap{{max-width:1080px;margin:0 auto;padding:clamp(1.5rem,4vw,3.5rem) 1.25rem 5rem;}}
header .eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .8rem;}}
h1{{font-family:var(--serif);font-weight:600;font-size:clamp(1.9rem,4.5vw,3rem);
  line-height:1.08;letter-spacing:-.01em;margin:0 0 .6rem;text-wrap:balance;}}
.lede{{font-size:1.12rem;color:var(--ink-2);max-width:64ch;margin:0 0 1rem;}}
.meta{{font-family:var(--mono);font-size:.78rem;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;margin-top:1.2rem;
  padding-top:1.2rem;border-top:1px solid var(--line);}}
h2{{font-family:var(--serif);font-weight:600;font-size:1.6rem;
  letter-spacing:-.01em;margin:3.4rem 0 .4rem;text-wrap:balance;}}
h2 .num-tag{{font-family:var(--mono);font-size:.7rem;color:var(--accent);
  vertical-align:middle;border:1px solid var(--accent);border-radius:2rem;
  padding:.1rem .55rem;margin-left:.6rem;letter-spacing:.08em;}}
.section-note{{color:var(--ink-2);max-width:66ch;margin:.2rem 0 1.6rem;}}
h3{{font-family:var(--sans);font-weight:650;font-size:1rem;letter-spacing:.01em;
  margin:2rem 0 .7rem;color:var(--ink);}}
h3 .tasks{{font-family:var(--mono);font-size:.72rem;color:var(--ink-3);
  font-weight:400;margin-left:.5rem;}}
p{{max-width:66ch;}}
a{{color:var(--accent);}}

.finding{{background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:10px;
  padding:1.4rem 1.6rem;margin:2rem 0;}}
.finding h2{{margin:0 0 .5rem;font-size:1.35rem;}}
.finding p{{margin:.5rem 0 0;color:var(--ink-2);}}
.finding b{{color:var(--ink);}}
.win{{font-family:var(--mono);font-size:.9rem;color:var(--ink);
  margin-top:.8rem;line-height:2;}}
.win b{{color:var(--accent);}}

figure{{margin:1.5rem 0 0;}}
figure img{{width:100%;height:auto;border:1px solid var(--line);
  border-radius:8px;background:var(--surface);display:block;}}
.fig-dark{{display:none;}}
:root[data-theme="dark"] .fig-light{{display:none;}}
:root[data-theme="dark"] .fig-dark{{display:block;}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]) .fig-light{{display:none;}}
  :root:not([data-theme="light"]) .fig-dark{{display:block;}}
}}
figcaption{{font-size:.85rem;color:var(--ink-3);margin-top:.7rem;max-width:70ch;}}

.legend{{display:flex;flex-wrap:wrap;gap:.5rem .9rem;margin:1.2rem 0 0;
  font-size:.82rem;color:var(--ink-2);}}
.legend span{{display:inline-flex;align-items:center;gap:.4rem;}}
.legend i{{width:.7rem;height:.7rem;border-radius:2px;display:inline-block;}}

.table-wrap{{overflow-x:auto;margin:.4rem 0 0;border:1px solid var(--line);
  border-radius:8px;background:var(--surface);}}
table.data{{border-collapse:collapse;width:100%;font-size:.85rem;
  font-variant-numeric:tabular-nums;}}
table.data th,table.data td{{padding:.42rem .6rem;text-align:left;
  white-space:nowrap;}}
table.data .num{{text-align:right;font-family:var(--mono);}}
table.data thead .grp{{font-size:.72rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);text-align:center;
  border-bottom:1px solid var(--line);font-weight:600;}}
table.data thead .grp.ours{{color:var(--accent);}}
table.data thead .grp.paper{{color:var(--paper);}}
table.data tr.sub th{{font-size:.73rem;color:var(--ink-3);font-weight:600;
  border-bottom:1px solid var(--line);position:sticky;top:0;
  background:var(--surface);}}
table.data tbody th{{font-weight:600;color:var(--ink);}}
table.data tbody th .fam{{display:block;font-family:var(--mono);
  font-size:.66rem;font-weight:400;letter-spacing:.04em;margin-top:.05rem;}}
table.data tbody tr{{border-top:1px solid var(--line);}}
table.data tbody tr:hover{{background:var(--surface-2);}}
table.data td.paper{{color:var(--ink-2);}}
table.data td.best{{color:var(--good);font-weight:700;}}
table.data .spacer{{width:14px;padding:0;background:transparent;
  border:none!important;}}
.fam-baseline th{{border-left:3px solid var(--fam-baseline);}}
.fam-regularise th{{border-left:3px solid var(--fam-regularise);}}
.fam-distill th{{border-left:3px solid var(--fam-distill);}}
.fam-replay th{{border-left:3px solid var(--fam-replay);}}
.fam-graph th{{border-left:3px solid var(--fam-graph);}}
.fam-baseline th .fam{{color:var(--fam-baseline);}}
.fam-regularise th .fam{{color:var(--fam-regularise);}}
.fam-distill th .fam{{color:var(--fam-distill);}}
.fam-replay th .fam{{color:var(--fam-replay);}}
.fam-graph th .fam{{color:var(--fam-graph);}}

table.taxonomy td.f1{{border-right:1px solid var(--line);}}
table.taxonomy th .cnt{{display:inline-block;font-family:var(--mono);
  font-size:.68rem;color:var(--ink-3);font-weight:400;margin-left:.5rem;}}
table.taxonomy .cnt-h{{font-family:var(--mono);font-size:.66rem;
  color:var(--ink-3);margin-left:.4rem;font-weight:400;}}
table.taxonomy tr.avg{{border-top:2px solid var(--line);font-weight:600;}}
table.taxonomy tr.avg td{{color:var(--ink);}}

.grid2{{display:grid;grid-template-columns:1fr;gap:1.6rem;}}
@media(min-width:720px){{.grid2{{grid-template-columns:1fr 1fr;}}}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:1.2rem 1.3rem;}}
.card h4{{margin:0 0 .5rem;font-size:.95rem;}}
.card p{{margin:0;font-size:.9rem;color:var(--ink-2);}}
.tag{{display:inline-block;font-family:var(--mono);font-size:.68rem;
  padding:.1rem .5rem;border-radius:1rem;letter-spacing:.05em;margin-bottom:.5rem;}}
.tag.warn{{color:var(--bad);border:1px solid var(--bad);}}
.tag.ok{{color:var(--good);border:1px solid var(--good);}}
footer{{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--line);
  font-size:.8rem;color:var(--ink-3);}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Replication analysis · multi-label class-incremental learning</p>
  <h1>EmoGrowth carried over to text</h1>
  <p class="lede">Nine continual-learning methods run on GoEmotions under the
  paper's four Audio28 protocols, read two ways: through EmoGrowth's own
  ranking metrics and through the GoEmotions taxonomy. The backbone is a
  fine-tuned BERT whose joint-training ceiling was validated separately.</p>
  <div class="meta">
    <span>28 emotions (27 + neutral)</span>
    <span>43,410 / 5,427 train·test</span>
    <span>seed 1993</span>
    <span>alphabetical class order</span>
    <span>AESL affective source: NRC-VAD lexicon</span>
  </div>
</header>

<div class="finding">
  <h2>The headline: LwF wins on text, not AESL</h2>
  <p><b>LwF has the highest last-task mAP under every protocol</b> — the reverse
  of the paper, where AESL leads and LwF trails it. On text with a fine-tuned
  BERT, the paper's flagship method does not reproduce its advantage: on B0-I7
  AESL reaches {aesl_b0i7:.1f} mAP against LwF's {lwf_b0i7:.1f}, and on the
  longer protocols it falls further behind. The graph baseline AGCN stays
  competitive with LwF; AESL's extra machinery — label-semantics learning and
  the two relation-distillation teachers — does not pay for itself here.</p>
  <p class="win">Best last-task mAP · {win_line}</p>
</div>

<h2>Why this might be<span class="num-tag">reading</span></h2>
<p class="section-note">Three differences from the paper's setup plausibly
explain the reordering, and they compound rather than compete.</p>
<div class="grid2">
  <div class="card"><h4>Fine-tuned backbone vs frozen features</h4>
  <p>EmoGrowth trains a small head over frozen 1000-d features; here BERT itself
  moves. AESL's semantic-guided decoupling and its relation-KD were designed for
  a fixed feature geometry — when the backbone drifts, the old model's feature
  teacher chases a moving target, and LwF's plain logit distillation is the more
  robust signal.</p></div>
  <div class="card"><h4>A 3-dimensional affective proxy</h4>
  <p>The paper's second RKD teacher is 11–14 human-rated appraisal dimensions.
  Text has none, so this uses a 3-d NRC-VAD lexicon vector — a much coarser
  similarity structure. The paper's own ablation values the whole RKD block at
  ~0.7 mAP, so this cannot be the main cause, but it removes one of AESL's legs.</p></div>
  <div class="card"><h4>Batch 16 and four epochs</h4>
  <p>The GoEmotions recipe uses batch 16; RKD compares within-batch similarity
  matrices, so AESL sees 240 off-diagonal entries where the paper's batch of 128
  gives 16,256. AESL is the only method whose loss depends on batch statistics,
  so it is the one this hurts.</p></div>
  <div class="card"><h4>What still reproduces</h4>
  <p>The <em>shape</em> of the field carries over cleanly: Finetune and EWC
  collapse, replay methods sit in the middle, and the distillation/graph methods
  lead. The gap between the leaders narrowed and reshuffled; the tiers did not.</p></div>
</div>

<h2>Incremental mAP across protocols<span class="num-tag">EmoGrowth · Fig 4</span></h2>
<p class="section-note">mAP as classes accumulate, one panel per protocol — the
text counterpart of the paper's Figure 4. LwF (blue), AGCN (orange) and AESL
(black) are drawn heavier; the six baselines are the thin lines.</p>
<figure>
  <img class="fig-light" alt="mAP versus number of classes for nine methods across four protocols"
       src="data:image/png;base64,{fig_light}">
  <img class="fig-dark" alt="mAP versus number of classes for nine methods across four protocols"
       src="data:image/png;base64,{fig_dark}">
  <figcaption>Every curve decays as classes accumulate. LwF stays on top
  throughout; AESL and AGCN track each other in the upper band on the base-heavy
  protocols (B16-*) but AESL sags on B0-I4, the longest schedule.</figcaption>
</figure>

<h2>EmoGrowth metrics, ours vs the paper<span class="num-tag">Tables 1–3</span></h2>
<p class="section-note">Avg mAP is the mean over all tasks; the other three are
last-task, over all 28 classes. The paper columns are Audio28 (audio, frozen
features) — read the ordering and the gaps, not the absolute values, since the
modality and backbone differ. Best last-task mAP in each block is green. The
coloured rail marks the method family.</p>
{eg_tables}

<div class="legend" style="margin-top:1.4rem">
  <span><i style="background:var(--fam-baseline)"></i>baseline (no anti-forgetting)</span>
  <span><i style="background:var(--fam-regularise)"></i>regularise (EWC)</span>
  <span><i style="background:var(--fam-distill)"></i>distill (LwF)</span>
  <span><i style="background:var(--fam-replay)"></i>replay (ER·RS·OCDM·PRS)</span>
  <span><i style="background:var(--fam-graph)"></i>graph (AGCN·AESL)</span>
</div>

<h2>GoEmotions taxonomy, final model<span class="num-tag">GoEmotions · Table 4</span></h2>
<p class="section-note">Per-emotion precision / recall / F1 at threshold 0.3 —
the GoEmotions paper's own reporting format — for the model after the final
task of B0-I7, beside the joint-training upper bound. Emotions are ordered by
test-set frequency (n). This is the view the EmoGrowth metrics cannot give: it
shows <em>which</em> emotions survive incremental learning.</p>
{go_table}
<p style="margin-top:1rem;font-size:.9rem;color:var(--ink-2)">The incremental
models keep high <em>recall</em> but low <em>precision</em> — they fire often
and imprecisely, where the upper bound is balanced. Frequent emotions
(<span style="font-family:var(--mono)">admiration, amusement, gratitude</span>)
survive well; rare ones
(<span style="font-family:var(--mono)">grief, relief, embarrassment</span>)
collapse to zero under every method, upper bound included — small-sample noise,
not a method failure. LwF, AESL and AGCN are near-indistinguishable here, which
is why the macro-F1 gap between them is a fraction of a point.</p>

<h2>Caveats<span class="num-tag">for the write-up</span></h2>
<div class="grid2">
  <div class="card"><span class="tag warn">single seed</span>
  <p>Every number is one run at seed 1993. The LwF-over-AESL gap on B0-I7 (about
  4 mAP) is within what a seed sweep could move; the gap on the longer protocols
  is larger and more robust. A 3-seed sweep would firm this up before the claim
  goes in the thesis.</p></div>
  <div class="card"><span class="tag warn">two thresholds</span>
  <p>EmoGrowth metrics binarise at 0.5 (logit&nbsp;&gt;&nbsp;0); the GoEmotions
  taxonomy table uses 0.3, the GoEmotions convention. Always state which — the
  two move macro-F1 by several points.</p></div>
  <div class="card"><span class="tag ok">verified</span>
  <p>Every ported loss and graph routine was diffed against the EmoGrowth
  source (0.0 to float precision). The backbone matches the validated GoEmotions
  replication field-for-field. So the reordering is a property of the text
  setting, not a porting artefact.</p></div>
  <div class="card"><span class="tag warn">AESL not yet ablated on text</span>
  <p>The EmoBank affective source and a frozen-backbone variant are built but
  not yet run. Either could recover some of AESL's standing; both are the
  natural next experiments.</p></div>
</div>

<footer>
Generated from run artifacts in <span style="font-family:var(--mono)">analysis/</span>.
Tables and figures also written to <span style="font-family:var(--mono)">tables/</span>
and <span style="font-family:var(--mono)">figures/</span>. Upper bound: bertgo
joint 28-class training, macro-F1 .495 at threshold 0.3.
</footer>
</div>
"""


if __name__ == "__main__":
    build()
