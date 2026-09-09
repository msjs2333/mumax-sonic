"""Compact selection evidence, independent of whether a device is open."""


def _percent(value):
    return '—' if value is None else f'{100*value:.1f}%'


def selection_summary(report):
    budget = report['budget']
    if report['validity'] != 'valid':
        return f"声源预算 {budget} · 数据未就绪，贡献覆盖不计算（{report['validity']}）"
    groups = report['groups']
    if not groups:
        aggregation = report.get('spatial_aggregation', {}).get('channels', {}).get('absolute', {})
        if (aggregation.get('omitted') or 0) > 0:
            return f"声源预算 {budget} · 无可选声源 · 聚合退化省略 {aggregation['omitted']:.3g} · 原始观察量未归零"
        return f'声源预算 {budget} · 无正强度候选 · 覆盖比例不适用'
    group = groups[0]
    values = group['absolute']
    ratio = report.get('selected_fraction_observer_input', {}).get('absolute', values['selected_fraction_total'])
    scope = '观察量' if 'spatial_aggregation' in report else '候选贡献'
    text = (f"预算 {budget} · 选入 {len(report['selected_ids'])} 路 / 非零输出候选 {len(report['output_ids'])} 路"
            f" · {scope}覆盖 {_percent(ratio)} · 预算省略 {values['omitted_budget']:.3g}"
            f" · 符号筛除 {values['excluded_sign']:.3g} · 关注静音 {values['excluded_attention']:.3g} [{group['unit']}]")
    if group['quantity'].startswith('Q_tile_'):
        fractions = report.get('selected_fraction_observer_input', {})
        text += f"\nQ+ 覆盖 {_percent(fractions.get('positive',group['positive']['selected_fraction_total']))} · Q− 覆盖 {_percent(fractions.get('negative',group['negative']['selected_fraction_total']))} · 分母为各自非负贡献，非 Qnet"
    elif 'spatial_aggregation' in report:
        omitted = report['spatial_aggregation']['channels']['absolute']['omitted']
        text += f'\n空间聚合省略 {omitted:.3g} · 覆盖为原始强度占比，不代表耳机可辨或设备已发声'
    if len(groups) > 1:
        text += f" · 另 {len(groups)-1} 组量独立统计，见导出"
    return text
