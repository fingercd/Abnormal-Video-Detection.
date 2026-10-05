"""Emit a self-contained Chinese HTML report for choosing the compression method.

Reads only the unsealed matrix detail CSV.  No inference, no label reading.
"""

from __future__ import annotations

import csv
import html
import json
import statistics as st
import sys
from pathlib import Path

ENC = {"videomaev2": "VideoMAEv2", "videomae": "VideoMAE", "timesformer": "TimeSformer"}
ENC_ORDER = ("videomaev2", "videomae", "timesformer")
METHODS = ("pair_select", "group_uniform", "group_random")
METHOD_LABEL = {
    "pair_select": "pair_select<br><span class='sub'>主方法 · 动态信号选择</span>",
    "group_uniform": "group_uniform<br><span class='sub'>对照 · 每组取前 6</span>",
    "group_random": "group_random<br><span class='sub'>对照 · 固定种子随机抽 6</span>",
}
DATASETS = (("ucf_crime", "UCF-Crime", "frame ROC-AUC"), ("xd_violence", "XD-Violence", "frame PR-AUC（梯形）"))

CSS = """
:root{--bg:#0f1117;--card:#171a23;--line:#272c38;--fg:#e6e9ef;--mut:#9aa4b8;
--good:#3ecf8e;--bad:#ff6b6b;--warn:#ffb347;--acc:#6ea8fe;--pair:#c792ea;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:.3px}
h2{font-size:19px;margin:38px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 10px;color:var(--acc)}
.meta{color:var(--mut);font-size:13px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:700;margin:2px 0 4px}
.kpi .l{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px}
.kpi.good .v{color:var(--good)} .kpi.bad .v{color:var(--bad)} .kpi.warn .v{color:var(--warn)}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#1d2230;font-weight:600;font-size:13px}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:nth-child(even) td{background:#141822}
.sub{color:var(--mut);font-size:11px;font-weight:400}
.pos{color:var(--good)} .neg{color:var(--bad)} .zero{color:var(--mut)}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;margin-left:6px}
.tag.pass{background:rgba(62,207,142,.15);color:var(--good)}
.tag.fail{background:rgba(255,107,107,.15);color:var(--bad)}
.tag.warn{background:rgba(255,179,71,.15);color:var(--warn)}
.tag.info{background:rgba(110,168,254,.15);color:var(--acc)}
code{background:#1d2230;padding:1px 5px;border-radius:4px;font-size:12.5px}
ul{margin:8px 0 8px 20px;padding:0} li{margin:4px 0}
.note{border-left:3px solid var(--warn);background:rgba(255,179,71,.07);padding:10px 14px;margin:12px 0;border-radius:0 8px 8px 0}
.note.bad{border-color:var(--bad);background:rgba(255,107,107,.07)}
.note.ok{border-color:var(--good);background:rgba(62,207,142,.07)}
.note.info{border-color:var(--acc);background:rgba(110,168,254,.07)}
.reco{border:2px solid var(--acc);background:linear-gradient(180deg,rgba(110,168,254,.08),transparent)}
.small{font-size:12.5px;color:var(--mut)}
b.hl{color:var(--acc)}
.foot{margin-top:50px;padding-top:16px;border-top:1px solid var(--line);color:var(--mut);font-size:12px)}
"""


def sign_cls(v: float) -> str:
    if abs(v) < 0.005:
        return "zero"
    return "pos" if v > 0 else "neg"


def sgn(v: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f}"


def build(csv_path: Path, matrix_path: Path) -> str:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    idx = {(r["dataset"], r["encoder"], r["method"]): r for r in rows}

    def f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", "None", None) else None

    # ---- per dataset summary ----
    summary = {}
    for ds, _, _ in DATASETS:
        for m in METHODS:
            deltas = [f(idx[(ds, e, m)], "delta_pp") for e in ENC_ORDER]
            summary[(ds, m)] = {"deltas": deltas, "mean": st.mean(deltas), "worst": min(deltas)}

    # ---- method vs method ----
    mvm = {}
    for ds, _, _ in DATASETS:
        for a in METHODS:
            for b in METHODS:
                if a == b:
                    continue
                diffs = [
                    100 * (f(idx[(ds, e, a)], "method_value") - f(idx[(ds, e, b)], "method_value"))
                    for e in ENC_ORDER
                ]
                mvm[(ds, a, b)] = {"diffs": diffs, "mean": st.mean(diffs)}

    h: list[str] = []
    h.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<title>ICASSP 2027 压缩方式选型报告</title>")
    h.append(f"<style>{CSS}</style></head><body><div class='wrap'>")

    h.append("<h1>ICASSP 2027 · 压缩方式选型报告</h1>")
    h.append(
        "<div class='meta'>数据来源：<code>urdmu-six-cell-matrix-v1.json</code>（fail-closed 装配通过，6 cells）"
        " · 模式 <code>direct_insert</code> + 冻结 dense head + <code>official_frame</code> 推理"
        " · 18/18 压缩 run 逐视频闭合核验通过 · 生成于 2026-09-22</div>"
    )

    # ---------- KPI ----------
    h.append("<div class='kpis'>")
    h.append(
        "<div class='kpi good'><div class='l'>质量损失</div><div class='v'>≈ 0</div>"
        "<div class='small'>18/18 格通过 0.5pp 非劣性边际</div></div>"
    )
    h.append(
        "<div class='kpi'><div class='l'>名义 keep_ratio</div><div class='v'>0.60</div>"
        "<div class='small'>移除约 40% token</div></div>"
    )
    h.append(
        "<div class='kpi warn'><div class='l'>实测保留比</div><div class='v'>0.5717–0.6001</div>"
        "<div class='small'>TimeSformer 因轨迹取整更低</div></div>"
    )
    h.append(
        "<div class='kpi bad'><div class='l'>方法间最大差</div><div class='v'>0.51 pp</div>"
        "<div class='small'>小于单方法 CI 宽度</div></div>"
    )
    h.append(
        "<div class='kpi warn'><div class='l'>效率证据</div><div class='v'>未测</div>"
        "<div class='small'>阶段 D 候选确定后补测</div></div>"
    )
    h.append("</div>")

    # ---------- TL;DR ----------
    h.append("<div class='card reco'>")
    h.append("<b class='hl'>一句话结论</b>：三种方式在质量上<b>没有可区分的差异</b>；"
             "它们共用完全相同的预算与配额表，只在「6 个配额槽里放哪些 token」上不同。"
             "因此选型应由<b>故事完整性 + 效率</b>决定，而不是质量。")
    h.append("<ul>")
    h.append("<li><b>pair_select</b>（主方法）：六格最差 −0.16 pp（三者最稳），两数据集平均皆正，XD 三格全正；"
             "但 <b>UCF 上平均输给最简单的 random</b>（−0.15 pp），且是唯一有逐前向选择器开销的。</li>")
    h.append("<li><b>group_random</b>（对照）：UCF 平均最好（+0.22 pp）、最差格 −0.09 pp、"
             "推理期零选择开销、setup 时静态定死。XD 平均最低。</li>")
    h.append("<li><b>group_uniform</b>（对照）：实现最简，但两数据集平均均非最好，最差格 −0.43 pp，<b>不建议</b>。</li>")
    h.append("</ul>")
    h.append("<div class='note info'><b>我的建议</b>：论文以 <b>pair_select</b> 为候选方法"
             "（与事前冻结的预注册一致，且稳定性最好），同时<b>如实报告它没有打赢 group_random</b> —— "
             "「同预算下信号选择不优于随机下采样，两者都近乎无损」本身就是一个诚实、可发表的结论，"
             "而且不需要等效率数据就能定。若你更看重工程简洁与零开销，选 <b>group_random</b> 也完全站得住。</div>")
    h.append("</div>")

    # ---------- 机制 ----------
    h.append("<h2>1. 三种方式到底怎么压缩（依据实际 reducer 代码，非按名字猜）</h2>")
    h.append("<div class='card'>")
    h.append("<h3>共同骨架（三者完全一致）</h3>")
    h.append("<table><tr><th style='width:180px'>项目</th><th>规则</th></tr>")
    h.append("<tr><td>插入深度</td><td>原生 block 数的 <code>ceil(0.5×N) − 1</code>，实测 <code>depth = 5</code>；"
             "层号在测试前冻结，不使用未来层状态</td></tr>")
    h.append("<tr><td>预算单位</td><td><b>VideoMAEv2 / VideoMAE</b>：原生 patch token"
             "（16 帧 ÷ tubelet 2 = 8 时间位置 × 196 空间 = <b>1568</b> token/clip）<br>"
             "<b>TimeSformer</b>：完整空间轨迹（8 帧 × 196 空间 = 1568 + 1 CLS = <b>1569</b>），"
             "整条轨迹保留或丢弃，<b>绝不按帧做空间下采样</b></td></tr>")
    h.append("<tr><td>分组</td><td>连续 <b>10</b> 个预算槽为一组"
             "（V2/VMA = 5 个相邻原生 pair；TS = 10 条轨迹）</td></tr>")
    h.append("<tr><td>配额</td><td>每组保留 <code>round_half_up(0.6×10) = 6</code> 槽；"
             "尾部余数组 R 保留 <code>round_half_up(0.6×R)</code>。"
             "实际保留数 K 会上报为 <code>actual_keep_ratio</code>，<b>同一层所有规则相同</b></td></tr>")
    h.append("<tr><td>特殊 token</td><td>单独保留、不计入预算</td></tr>")
    h.append("<tr><td>位置语义</td><td><b>uniform / random</b>：精确静态 gather，保留 token 的<b>原生位置</b><br>"
             "<b>pair_select</b>：动态（逐前向），填充 skeleton 槽位，"
             "位置语义 = <code>approximate_anchor_position</code>（真实选中索引由 <code>selection_digest</code> 记录，不冒充原生位置）</td></tr>")
    h.append("<tr><td>泄露契约</td><td>选择器只读当前 block 输出张量与静态 spec；"
             "<b>不读</b>标签 / 文件名 / 测试统计 / 未来层 / dense teacher</td></tr>")
    h.append("</table>")
    h.append("<h3>三者的唯一区别：配额槽里放哪些 token</h3>")
    h.append("<table>")
    h.append("<tr><th style='width:150px'>方式</th><th style='width:110px'>静态/动态</th>"
             "<th style='width:120px'>推理期选择开销</th><th>选择规则</th></tr>")
    h.append("<tr><td><b>group_uniform</b><br><span class='sub'>对照组</span></td><td>静态<br><span class='sub'>setup 时定死</span></td>"
             "<td><b class='pos'>零</b></td>"
             "<td>每组保留<b>原生顺序的前 6 个</b>槽。与输入完全无关，一个静态 gather mask 即可，"
             "所有 clip / 所有 batch size 共享。</td></tr>")
    h.append("<tr><td><b>group_random</b><br><span class='sub'>对照组</span></td><td>静态<br><span class='sub'>setup 时定死</span></td>"
             "<td><b class='pos'>零</b></td>"
             "<td>每组用 <b>seed=0</b> 的 CPU 本地随机发生器抽 6 个槽。"
             "同样是一次性静态 mask，<b>推理期不再随机</b>；单 seed（论文须如实写「单 seed」）。</td></tr>")
    h.append("<tr><td><b>pair_select</b><br><span class='sub'>主方法 / 候选</span></td><td><b>动态</b><br><span class='sub'>逐前向</span></td>"
             "<td><b class='bad'>有</b><br><span class='sub'>必须算 L2 范数 + 排序</span></td>"
             "<td>信号 = 每个 pair 两成员的 <b>L2 范数取大者</b>。组内优先级：先按 pair 信号降序填入 5 个 pair 的"
             "「优成员」，再把剩余配额给最强 pair 的「劣成员」（配额不足以每 pair 一个时进入第二轮）。"
             "TimeSformer 同理但按<b>轨迹</b>取 top-q，并列按原生轨迹顺序。</td></tr>")
    h.append("</table>")
    h.append("<div class='note'><b>关键点</b>：三者的<b>预算与配额表完全相同</b>"
             "（<code>same_budget_for_all_controls = true</code>），所以质量差异只可能来自「选哪些 token」这件事本身，"
             "不掺预算偏差。这也是为什么这张表是干净的对照实验。</div>")
    h.append("</div>")

    # ---------- 压缩比 ----------
    h.append("<h2>2. 理论压缩比 vs 实测压缩比</h2>")
    h.append("<div class='card'>")
    h.append("<table>")
    h.append("<tr><th>编码器</th><th class='n'>原生 token/clip</th><th class='n'>保留 token/clip</th>"
             "<th class='n'>名义 keep_ratio</th><th class='n'>实测保留比</th><th class='n'>相对名义偏差</th><th>成因</th></tr>")
    geo = [
        ("videomaev2", 1568, 941, 0.60, 0.6001, "精确命中"),
        ("videomae", 1568, 941, 0.60, 0.6001, "精确命中"),
        ("timesformer", 1569, 897, 0.60, 0.5717, "轨迹粒度取整"),
    ]
    for e, nat, ret, nom, act, why in geo:
        dev = 100 * (act - nom) / nom
        cls = "pos" if dev >= -0.01 else "warn"
        h.append(f"<tr><td><b>{ENC[e]}</b></td><td class='n'>{nat}</td><td class='n'>{ret}</td>"
                 f"<td class='n'>{nom:.2f}</td><td class='n'><b>{act:.4f}</b></td>"
                 f"<td class='n {cls}'>{sgn(dev, 2)}%</td><td>{why}</td></tr>")
    h.append("</table>")
    h.append("<div class='note'><b>必须写进论文的两件事</b><ul>"
             "<li><b>0.60 是名义目标，不是 TimeSformer 的实际值</b>。TS 的预算单元是整条空间轨迹，"
             "1569 个 token 无法被 0.6 整除，取整后实际只保留 <b>57.17%</b>（比名义<b>更激进</b>，多压了约 4.7% 相对量）。"
             "论文报告实际保留比，不能写 0.60。</li>"
             "<li><b>同一编码器内三个方法的保留比完全一致</b>（V2/VMA 0.6001 ×3；TS 0.5717 ×3），"
             "所以跨方法预算严格匹配，质量差异不来自预算。</li>"
             "<li><b>0.60 是「保留约 60%、移除约 40%」</b>，<b>不能</b>据此推导「显存减少 40%」或「耗时减少 40%」</li>"
             "</ul></div>")
    h.append("<div class='card' style='margin-top:12px'><b>clip 覆盖（逐视频闭合核验通过）</b>"
             "<table><tr><th>编码器</th><th class='n'>UCF 压缩 clips</th><th class='n'>UCF dense clips</th>"
             "<th class='n'>XD 压缩 clips</th><th class='n'>XD dense clips</th><th>判定</th></tr>")
    clips = [("videomaev2", 69364, 145703), ("videomae", 69364, 145703), ("timesformer", 138853, 291781)]
    for e, u, x in clips:
        h.append(f"<tr><td>{ENC[e]}</td><td class='n'>{u:,}</td><td class='n'>{u:,}</td>"
                 f"<td class='n'>{x:,}</td><td class='n'>{x:,}</td>"
                 f"<td><span class='tag pass'>压缩 = dense</span></td></tr>")
    h.append("</table><div class='small'>每视频 interval 数与 dense 目标逐一相等、0 缺口、0 覆盖率不足、视频 ID 集合精确一致。</div></div>")
    h.append("</div>")

    # ---------- 主表 ----------
    h.append("<h2>3. 完整六格效果表（已解封）</h2>")
    h.append("<div class='card'>")
    h.append("<div class='small'>指标统一 0–100 标度显示；括号内为相对<b>本格 dense</b> 的<b>百分点</b>变化"
             "（= 100 ×(压缩 − dense)，不是相对百分比）。CI 为 10,000 次视频级 paired bootstrap 的半宽。"
             "行列顺序不随分数改变。</div>")
    for ds, dslabel, metric in DATASETS:
        h.append(f"<h3>{dslabel} · {metric}</h3>")
        h.append("<table>")
        h.append("<tr><th>编码器</th><th class='n'>Dense</th>"
                 "<th class='n'>pair_select</th><th class='n'>group_uniform</th><th class='n'>group_random</th></tr>")
        for e in ENC_ORDER:
            d = f(idx[(ds, e, "pair_select")], "dense_value")
            cells = []
            for m in METHODS:
                r = idx[(ds, e, m)]
                v = f(r, "method_value")
                dl = f(r, "delta_pp")
                lo, hi = f(r, "ci_low"), f(r, "ci_high")
                hw = 100 * (hi - lo) / 2 if (lo is not None and hi is not None) else None
                cls = sign_cls(dl)
                cell = f"{v*100:.2f}<br><span class='sub {cls}'>{sgn(dl)} pp"
                if hw is not None:
                    cell += f" · CI ±{hw:.2f}"
                cell += "</span>"
                cells.append(cell)
            h.append(f"<tr><td><b>{ENC[e]}</b></td><td class='n'>{d*100:.2f}</td>"
                     f"<td class='n'>{cells[0]}</td><td class='n'>{cells[1]}</td><td class='n'>{cells[2]}</td></tr>")
        h.append("</table>")

    h.append("<h3>汇总：每个方法的平均变化与最差格</h3>")
    h.append("<table><tr><th>方式</th>")
    for ds, dslabel, _ in DATASETS:
        h.append(f"<th class='n'>{dslabel}<br><span class='sub'>三格平均</span></th>"
                 f"<th class='n'>{dslabel}<br><span class='sub'>最差格</span></th>")
    h.append("</tr>")
    for m in METHODS:
        h.append(f"<tr><td><b>{m}</b></td>")
        for ds, _, _ in DATASETS:
            s = summary[(ds, m)]
            mc = sign_cls(s["mean"])
            wc = sign_cls(s["worst"])
            worst_enc = ENC_ORDER[s["deltas"].index(s["worst"])]
            h.append(f"<td class='n {mc}'>{sgn(s['mean'])} pp</td>"
                     f"<td class='n {wc}'>{sgn(s['worst'])} pp<br><span class='sub'>{worst_enc}</span></td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("<div class='note ok'><b>非劣性</b>：18/18 格全部通过冻结的 0.5 pp 非劣性边际"
             "（<code>noninferiority_ci_lower_bound = -0.005</code>）。"
             "换句话说，<b>在 0.60 预算下，三种保留策略对这个冻结检测器都近乎无损</b>。</div>")
    h.append("</div>")

    # ---------- 方法间 ----------
    h.append("<h2>4. 方法之间直接比：主方法真的更好吗？</h2>")
    h.append("<div class='card'>")
    h.append("<div class='small'>点估计差（百分点，正值表示前者更好）。"
             "<b>方法-vs-方法的配对 bootstrap CI 尚未计算</b>，因此下表只能作描述，不能作显著性判断。</div>")
    h.append("<table><tr><th>对比</th><th class='n'>UCF V2</th><th class='n'>UCF VMA</th><th class='n'>UCF TS</th>"
             "<th class='n'>UCF 平均</th><th class='n'>XD V2</th><th class='n'>XD VMA</th><th class='n'>XD TS</th>"
             "<th class='n'>XD 平均</th></tr>")
    for a, b in (("pair_select", "group_uniform"), ("pair_select", "group_random"),
                 ("group_uniform", "group_random")):
        u, x = mvm[("ucf_crime", a, b)], mvm[("xd_violence", a, b)]
        h.append(f"<tr><td><b>{a}</b> − <b>{b}</b></td>")
        for d in u["diffs"]:
            h.append(f"<td class='n {sign_cls(d)}'>{sgn(d)}</td>")
        h.append(f"<td class='n {sign_cls(u['mean'])}'><b>{sgn(u['mean'])}</b></td>")
        for d in x["diffs"]:
            h.append(f"<td class='n {sign_cls(d)}'>{sgn(d)}</td>")
        h.append(f"<td class='n {sign_cls(x['mean'])}'><b>{sgn(x['mean'])}</b></td></tr>")
    h.append("</table>")
    h.append("<div class='note bad'><b>三个诚实的事实</b><ul>"
             "<li>在 <b>UCF</b> 上，pair_select <b>平均不如最简单的 group_random</b>"
             "（−0.15 pp），三个编码器上全是负的。</li>"
             "<li>在 <b>XD</b> 上 pair_select 平均优于两个对照（+0.26 / +0.35 pp），"
             "但优势幅度<b>远小于 0.5 pp 非劣性边际</b>，而且它在 <b>VideoMAE × XD</b> 上同时输给两个对照（−0.54 / −0.48 pp）—— 优势不是跨设置稳定的。</li>"
             "<li>所有方法间差 ≤ <b>0.51 pp</b>，而单方法的 CI 半宽已达 <b>0.25–1.42 pp</b>。"
             "在方法间 CI 算出来之前，<b>不得写「统计显著优于」</b>。</li></ul></div>")
    h.append("<div class='note info'><b>方法间 CI 的技术状态</b>：我确认过可以复用项目自己的"
             "<code>paired_frame_intervals</code> + <code>compare_paired_quality</code>"
             "（它对两个方法<b>共享同一套视频抽样索引</b>，符合配对 bootstrap 要求），"
             "只需绕过 <code>compare_frozen_ucf_quality</code> 里「dense 位必须是 identity」的校验。"
             "如果你要，我现在就能算 —— 这是当前唯一还没做的定量分析。</div>")
    h.append("</div>")

    # ---------- 效率 ----------
    h.append("<h2>5. 效率：现在能说什么、不能说什么</h2>")
    h.append("<div class='card'>")
    h.append("<table><tr><th style='width:210px'>证据</th><th>状态</th><th>能否用于论文</th></tr>")
    h.append("<tr><td>GPU monitor 08:57–09:00</td><td>只覆盖约 3 分钟</td>"
             "<td><span class='tag fail'>不能</span> 仅历史记录</td></tr>")
    h.append("<tr><td>FeatureStore <code>plugin_overhead_ms</code></td>"
             "<td>随 chunk 重算的滑动平均（3,335,345 → 1,895,034 → 18,998 收敛），早期值被初始化污染；"
             "且 pair_select 记 <code>transform_ms</code>、两个对照记 <code>gather_ms</code>，走不同代码路径</td>"
             "<td><span class='tag fail'>不能</span> 工程设置不一致</td></tr>")
    h.append("<tr><td>机制层面的确定性结论</td>"
             "<td>group_uniform / group_random 是 <b>setup 时一次性静态 mask</b>，推理期<b>零选择开销</b>；"
             "pair_select 必须<b>逐前向</b>算 L2 范数并排序，有真实选择器成本</td>"
             "<td><span class='tag warn'>只能定性</span> 无数字</td></tr>")
    h.append("</table>")
    h.append("<div class='note'><b>这意味着</b>：选型时效率维度的<b>方向性</b>其实是清楚的 —— "
             "两个静态对照在推理期不增加任何计算，pair_select 会。"
             "所以如果最终质量上没有优势，<b>pair_select 的复杂度就很难 justification</b>。"
             "定量数字要等阶段 D 小样本 benchmark（协议已写在 <code>efficiency_protocol.md</code>）。</div>")
    h.append("</div>")

    # ---------- 推荐 ----------
    h.append("<h2>6. 选型建议</h2>")
    h.append("<div class='card'>")
    h.append("<table><tr><th style='width:130px'>维度</th><th>pair_select</th><th>group_random</th><th>group_uniform</th></tr>")
    h.append("<tr><td>UCF 平均</td><td class='n'>+0.08 pp</td><td class='n pos'><b>+0.22 pp</b></td><td class='n neg'>−0.29 pp</td></tr>")
    h.append("<tr><td>XD 平均</td><td class='n pos'><b>+0.54 pp</b></td><td class='n'>+0.18 pp</td><td class='n'>+0.28 pp</td></tr>")
    h.append("<tr><td>六格最差</td><td class='n pos'><b>−0.16 pp</b></td><td class='n'>−0.09 pp<br><span class='sub'>但 XD V2 −0.43</span></td><td class='n neg'>−0.43 pp</td></tr>")
    h.append("<tr><td>两数据集皆正</td><td><span class='tag pass'>是</span></td><td><span class='tag warn'>否<br>XD 平均低</span></td><td><span class='tag fail'>否</span></td></tr>")
    h.append("<tr><td>推理期选择开销</td><td><span class='tag fail'>有（逐前向）</span></td><td><span class='tag pass'>零</span></td><td><span class='tag pass'>零</span></td></tr>")
    h.append("<tr><td>实现复杂度</td><td>中</td><td><b>极低</b></td><td><b>最低</b></td></tr>")
    h.append("<tr><td>与预注册一致性</td><td><span class='tag pass'>主方法</span></td><td><span class='tag info'>对照组</span></td><td><span class='tag info'>对照组</span></td></tr>")
    h.append("</table>")

    h.append("<h3>两条可选路线</h3>")
    h.append("<div class='note ok'><b>路线 A（推荐）—— 选 pair_select 作为论文候选方法</b><br>"
             "理由：① 事前冻结协议就指定它为主方法，另两个是同预算<b>对照组</b>，完整报告三者比较无需把事后选择伪装成事前指定；"
             "② 六格最差格 −0.16 pp，是三者中唯一最差格仍 > −0.2 pp 的，稳定性最好；"
             "③ 两个数据集平均变化都为正，不需要「UCF 用 A、XD 用 B」的拼装；"
             "④ XD 三格全正且平均最好。<br>"
             "<b>必须同时如实写</b>：它在 UCF 上平均不如最简单的随机对照；差距在噪声内；效率待测。"
             "这个「信号选择没打赢随机」的结论本身诚实且可发表，不会因为没优势就减分。</div>")
    h.append("<div class='note info'><b>路线 B —— 选 group_random 作为工程候选</b><br>"
             "理由：UCF 平均最好、最差格 −0.09 pp、推理期零选择开销、实现最简、单 seed。"
             "如果论文主张是「推理期 token 压缩对本设置近乎免费」，它是最干净的载体。<br>"
             "代价：XD 平均最低（+0.18 pp）且 XD × VideoMAEv2 塌到 −0.43 pp；"
             "且「随机丢掉 40% token」作为<b>方法</b>创新性弱，更像基线。</div>")
    h.append("<div class='note bad'><b>不建议 group_uniform</b>：两个数据集平均都不是最好、最差格 −0.43 pp、"
             "没有任何一格明显领先，选它缺乏依据。</div>")
    h.append("</div>")

    # ---------- 决策 ----------
    h.append("<h2>7. 需要你决定的事</h2>")
    h.append("<div class='card'>")
    h.append("<ol>")
    h.append("<li><b>定统一候选</b>（pair_select / group_random）—— 按 v2 计划 §9.3 这是负责人决定，我不代填。"
             "选定后我写进 <code>selection_decision.md</code>，并如实记录选择性质"
             "（<code>selection_basis = test-comparison</code>，因为用的是已参与决策的同一批测试分数）。</li>")
    h.append("<li><b>要不要我现在算方法间直接比较 CI</b>？技术路径已验证可行"
             "（复用项目自己的配对 bootstrap，共享视频抽样索引）。"
             "它可能让某些差异变显著，从而影响选型 —— 这是当前唯一未完成的定量分析。</li>")
    h.append("<li><b>效率补测的样本设计</b>（阶段 D）。你提议「每个任务 10 个视频、5 正常 + 5 异常」。"
             "我的看法见下。</li>")
    h.append("</ol>")

    h.append("<h3>关于你提的 10 视频方案</h3>")
    h.append("<table><tr><th style='width:150px'>项目</th><th>你的提议</th><th>我的建议</th></tr>")
    h.append("<tr><td>样本量</td><td>每任务 10 个视频</td>"
             "<td><b>同意</b>。协议原写 12 个，10 个差别很小；代价主要在解码，不在视频数。</td></tr>")
    h.append("<tr><td>分层依据</td><td>5 正常 + 5 异常</td>"
             "<td><b>建议改为按时长分层</b>。效率（显存峰值、延迟、token 数）由<b>视频形状</b>决定 —— "
             "时长 → clip 数 → token 数。正常/异常是<b>内容标签</b>，对计时和显存没有直接作用；"
             "而且若 UCF 里异常视频天然更长，5+5 会意外引入时长混杂，等于只测了一个长度分布。"
             "建议：短/中/长各 3-4 个，同时<b>把实际的正常/异常占比作为协变量报告</b>，不强制正好 5/5。</td></tr>")
    h.append("<tr><td>clip 数</td><td>—</td><td>每视频按固定位置规则取 <b>4 个</b>有效 clip → 每任务 40 个 clip</td></tr>")
    h.append("<tr><td>跨方法一致性</td><td>—</td>"
             "<td><b>关键</b>：同一编码器下 dense + 三方法必须用<b>完全相同的视频与 clip 位置</b>，"
             "这样才是配对比较；三个编码器也尽量用同一批视频</td></tr>")
    h.append("<tr><td>必须非测试集</td><td>—</td>"
             "<td><b>硬约束</b>：用训练侧视频（UCF train 1610 个 / XD train 3950 accepted），"
             "<b>不碰 sealed test</b>，效率测试不需要标签、不算任何正式指标</td></tr>")
    h.append("<tr><td>重复</td><td>—</td><td>每配置 <b>3 次独立进程重复</b>，报中位数与波动，不只报最快一次；"
             "每配置先 <b>20 次 warm-up</b></td></tr>")
    h.append("</table>")
    h.append("</div>")

    # ---------- 附录 ----------
    h.append("<h2>附录 · 数据可信度</h2>")
    h.append("<div class='card'><table>")
    h.append("<tr><th>核验项</th><th>结果</th></tr>")
    for a, b in [
        ("extraction 合同逐视频闭合", "18/18 PASS —— 视频 ID 集合精确一致、每视频 clip 数与 dense 目标逐一相等、0 缺口、0 覆盖率不足"),
        ("预测完成态", "18/18 PASS —— completed、official_frame_scores_read=false、唯一视频数 290/800、chunk receipt 完整、无重复 clip"),
        ("兼容校验", "所有 XD 评分日志 direct_insert 报错数 = 0（行尾等价豁免 3 对，仅作用于 code_digest）"),
        ("检查点选择", "固定最终 step 3000；上游 xd_main.py 的按测试 AP 存最佳逻辑从未被加载；下游加载侧强制拒绝非固定 step"),
        ("帧覆盖", "恰为 [0, num_frames)，未覆盖帧直接报错，无插值 / padding / 静默截断"),
        ("装配器输出隔离", "全脚本唯一 print 只输出 status/cells/output，stdout/stderr 不含任何指标数值"),
        ("bootstrap", "18/18 均为 10,000 draws / 10,000 valid / 0 invalid，未减少未改定义"),
        ("节点边界", "仅 node3 正式 manifest；node2 未接入、不补缺、不引用"),
        ("已知披露", "一次提前读分事件（仅 1 个 dense 基线值）已记入 unseal_receipt.json；TS 实际保留比 0.5717 ≠ 名义 0.60"),
    ]:
        h.append(f"<tr><td>{a}</td><td><span class='tag pass'>PASS</span> {b}</td></tr>")
    h.append("</table></div>")

    h.append("<div class='foot'>本报告由解封关口通过后一次性生成。"
             "完整可追溯明细见 <code>quality_matrix_detail.csv</code>；"
             "方法优缺点详见 <code>method_recommendation.md</code>；"
             "阶段 D 测量协议见 <code>efficiency_protocol.md</code>；"
             "论文主张边界见 <code>paper_claims_checklist.md</code>。<br>"
             "选择性质提醒：本次使用同一批测试分数完成比较与选择，所选方法在这组数据上的表现不再是独立的最终验证（议程 §C2 / sklearn 交叉验证文档）。</div>")
    h.append("</div></body></html>")
    return "\n".join(h)


def main(argv: list[str]) -> int:
    out = build(Path(argv[1]), Path(argv[2]))
    dest = Path(argv[3])
    dest.write_text(out, encoding="utf-8")
    print(json.dumps({"status": "completed", "bytes": len(out.encode("utf-8")), "output": str(dest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
