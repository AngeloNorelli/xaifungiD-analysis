"""
Korelacja jednego uczestnika z całą resztą + wizualizacja 3D.

Idea:
  Każdy uczestnik to szereg czasowy (po slajdach) trzech metryk:
  engagement_level, confidence_level, communication_style.
  Dla wybranego uczestnika docelowego liczymy korelację Pearsona
  KAŻDEJ metryki osobno z każdym innym uczestnikiem (po wspólnych slajdach).
  Daje to 3 wartości na parę -> 3 osie wykresu.

Osie wykresu 3D:
  x = korelacja engagement_level
  y = korelacja confidence_level
  z = korelacja communication_style
Uczestnik docelowy = punkt (1, 1, 1).
"""

import json
import numpy as np
import plotly.graph_objects as go

MIN_SLIDES = 3        # minimalna liczba slajdów aby uczestnik był brany pod uwagę
MIN_SHARED = 4        # minimalna liczba wspólnych slajdów dla sensownej korelacji
METRICS = ["engagement_level", "confidence_level", "communication_style"]

GROUP_COLORS = {"DE": "#e4572e", "IT": "#2e86ab", "SSH": "#8a4fff"}
GROUP_NAMES = {"DE": "Eksperci (DE)", "IT": "Informatycy (IT)", "SSH": "Humaniści (SSH)"}


def load_profiles(path):
    """Wczytuje plik i zwraca {participant_id: {slide_id: (eng, conf, comm)}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}
    for pid, slides in data.items():
        series = {}
        for slide_id, records in slides.items():
            if not records:
                continue
            try:
                entry = json.loads(records[0])
                prof = entry["profile"]
                vals = tuple(prof.get(m) for m in METRICS)
                if None not in vals:
                    series[slide_id] = tuple(float(v) for v in vals)
            except Exception:
                continue
        if len(series) >= MIN_SLIDES:
            profiles[pid] = series
    return profiles


def correlate_one_vs_rest(profiles, target_id):
    """Zwraca listę rekordów korelacji uczestnika docelowego z resztą."""
    target = profiles[target_id]
    results = []
    for pid, series in profiles.items():
        if pid == target_id:
            continue
        shared = sorted(set(target) & set(series))
        if len(shared) < MIN_SHARED:
            continue

        tv = np.array([target[s] for s in shared])
        pv = np.array([series[s] for s in shared])

        corrs = []
        for i in range(3):
            a, b = tv[:, i], pv[:, i]
            if a.std() == 0 or b.std() == 0:
                corrs.append(0.0)          # brak zmienności -> korelacja nieokreślona
            else:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))

        results.append({
            "participant_id": pid,
            "group": pid.split("_")[1] if len(pid.split("_")) > 1 else "?",
            "shared": len(shared),
            "r_engagement": corrs[0],
            "r_confidence": corrs[1],
            "r_communication": corrs[2],
            "mean_r": float(np.mean(corrs)),
        })
    return results


def build_3d_figure(results, target_id):
    fig = go.Figure()

    # punkty uczestników, pogrupowane wg grupy
    for grp in ["DE", "IT", "SSH"]:
        pts = [r for r in results if r["group"] == grp]
        if not pts:
            continue
        fig.add_trace(go.Scatter3d(
            x=[r["r_engagement"] for r in pts],
            y=[r["r_confidence"] for r in pts],
            z=[r["r_communication"] for r in pts],
            mode="markers",
            name=GROUP_NAMES.get(grp, grp),
            marker=dict(size=6, color=GROUP_COLORS.get(grp, "#888"), opacity=0.85,
                        line=dict(width=0.5, color="#222")),
            text=[f"{r['participant_id']}<br>wspólnych slajdów: {r['shared']}"
                  f"<br>r_eng={r['r_engagement']:.2f}, r_conf={r['r_confidence']:.2f},"
                  f" r_comm={r['r_communication']:.2f}" for r in pts],
            hovertemplate="%{text}<extra></extra>",
        ))

    # uczestnik docelowy w (1,1,1)
    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1], mode="markers+text",
        name=f"{target_id} (cel)",
        marker=dict(size=10, color="#f4c20d", symbol="diamond",
                    line=dict(width=1.5, color="#222")),
        text=[target_id], textposition="top center",
        hovertemplate=f"{target_id} (uczestnik docelowy)<extra></extra>",
    ))

    fig.update_layout(
        title=f"Korelacja dynamiki metryk: {target_id} vs pozostali uczestnicy",
        scene=dict(
            xaxis=dict(title="r — engagement_level", range=[-1.05, 1.05], zeroline=True),
            yaxis=dict(title="r — confidence_level", range=[-1.05, 1.05], zeroline=True),
            zaxis=dict(title="r — communication_style", range=[-1.05, 1.05], zeroline=True),
        ),
        legend=dict(title="Grupa", x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    import sys
    path = "../llm_responses.json"
    target = sys.argv[1] if len(sys.argv) > 1 else "MW_IT_06"

    profiles = load_profiles(path)
    print(f"Uczestników nadających się do analizy: {len(profiles)}")

    results = correlate_one_vs_rest(profiles, target)
    results.sort(key=lambda r: r["mean_r"], reverse=True)

    print(f"\nNajbardziej skorelowani z {target} (wg średniej z 3 metryk):")
    for r in results[:5]:
        print(f"  {r['participant_id']:<12} {r['group']:<4} "
              f"mean_r={r['mean_r']:.2f}  "
              f"(eng={r['r_engagement']:.2f}, conf={r['r_confidence']:.2f}, comm={r['r_communication']:.2f})")

    fig = build_3d_figure(results, target)
    out = f"../correlation_3d_{target}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nZapisano wykres: {out}")