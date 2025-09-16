# ===== app.py (fixed end-to-end) =====
import json, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules.pdb_utils import (
    parse_header_remark,
    parse_plddt_series,
    render_overlay_by_residue,
    render_pdb_bfactor,
)
from modules.vqe_runner import parse_pauli_text, run_custom_pauli, run_spin_chain
from modules.metrics import convergence_metrics

st.set_page_config(page_title="Protein VQE Lab", layout="wide")

P_RUNS = Path("runs")
P_RUNS.mkdir(exist_ok=True)
P_CACHE = Path(".cache")
P_CACHE.mkdir(exist_ok=True)

st.sidebar.title("Protein VQE Lab")
page = st.sidebar.radio(
    "Pages", ["① Protein (AI)", "② VQE (Quantum)", "③ Results / Cross-Check"]
)

# ---------------- Common helpers ----------------
@st.cache_data
def _parse_plddt_cached(pdb_text: str):
    return parse_plddt_series(pdb_text)

def save_run(label: str, energy_series, meta: dict) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = P_RUNS / f"{stamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    pd.DataFrame(
        {"step": range(len(energy_series)), "energy": energy_series}
    ).to_csv(run_dir / "energy_curve.csv", index=False)
    # JSON (numpy 등 직렬화 안전)
    meta_safe = json.loads(json.dumps(meta, default=float))
    (run_dir / "summary.json").write_text(json.dumps(meta_safe, indent=2), "utf-8")
    return run_dir

import ast, re

def load_summary_safe(js_path: Path):
    try:
        text = js_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("empty file")
        # 1) 정상 JSON 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 2) 홑따옴표 → 쌍따옴표 치환 (간단 복구)
            fixed = re.sub(r"'([^']*)'", r'"\1"', text)
            try:
                return json.loads(fixed)
            except Exception:
                # 3) Python literal_eval fallback
                obj = ast.literal_eval(text)
                return json.loads(json.dumps(obj, default=float))
    except Exception as e:
        st.warning(f"무시됨: {js_path.name} ({e})")
        return None

st.session_state.setdefault("pdb_text", None)

# ================= Page 1: Protein (AI) =================
if page == "① Protein (AI)":
    st.header("① Protein (AI: AlphaFold) — pLDDT 시각화")

    up = st.file_uploader("PDB 업로드 (AlphaFold, B-factor = pLDDT)", type=["pdb"])
    if up:
        st.session_state["pdb_text"] = up.getvalue().decode("utf-8", errors="ignore")

    if st.session_state.get("pdb_text"):
        pdb_text = st.session_state["pdb_text"]

        # HEADER/REMARK 요약
        st.subheader("PDB 헤더 요약")
        st.text_area(
            "HEADER / TITLE / REMARK 1", parse_header_remark(pdb_text), height=220
        )

        # pLDDT 분포 + 3D
        st.subheader("pLDDT 분포 & 3D")
        plddt_arr, per_res = _parse_plddt_cached(pdb_text)
        if plddt_arr is None:
            st.warning("이 PDB에는 pLDDT(b-factor)가 없을 수 있어요.")
        else:
            c1, c2 = st.columns([1, 1.3])
            with c1:
                fig = plt.figure(figsize=(4, 3))
                plt.hist(plddt_arr, bins=30)
                plt.xlabel("pLDDT")
                plt.ylabel("Count")
                st.pyplot(fig)
                st.caption(
                    f"N={len(plddt_arr)}, mean={np.mean(plddt_arr):.1f}, median={np.median(plddt_arr):.1f}"
                )
            with c2:
                st.write("3D 구조 (pLDDT 컬러)")
                render_pdb_bfactor(pdb_text)

# ================= Page 2: VQE (Quantum) =================
elif page == "② VQE (Quantum)":
    st.header("② VQE 설정/실행")

    mode = st.radio("모드 선택", ["Spin-chain (toy)", "Custom Hamiltonian (text)"])
    c = st.columns(3)
    steps = int(c[0].number_input("반복(steps)", 10, 1000, 80, 10))
    noise = float(c[1].number_input("노이즈 σ", 0.0, 0.5, 0.02, 0.01))
    seed = int(c[2].number_input("Seed", 0, 99999, 0))

    terms = None
    if mode.startswith("Custom"):
        txt = st.text_area("Pauli terms (예: `-1.0 ZI`, `+0.5 XX`)", height=150)
        terms = parse_pauli_text(txt)

    if st.button("VQE 실행"):
        progress = st.progress(0)
        placeholder = st.empty()
        energies = []

        # 모의 실행(실제 VQE 연결 시 이 부분만 교체)
        if mode.startswith("Spin"):
            res = run_spin_chain(steps=steps, noise=noise, seed=seed)
        else:
            res = run_custom_pauli(terms=terms or [], steps=steps, noise=noise, seed=seed)

        for i, e in enumerate(res.series):
            energies.append(e)
            progress.progress(min((i + 1) / len(res.series), 1.0))
            placeholder.line_chart(pd.Series(energies, name="Energy"))

        cm = convergence_metrics(energies)
        meta = {
            "backend": res.backend,
            "E_min": float(np.min(energies)) if energies else None,
            "steps": len(energies),
            "delta_e": cm.delta_e,
            "r2_exp": cm.r2_exp,
            "var_last": cm.var_last,
            "mode": "spin" if mode.startswith("Spin") else "custom",
        }
        run_dir = save_run("vqe", energies, meta)
        st.success(f"완료: {run_dir.name}")
        st.json(meta)

# =========== Page 3: Results / Cross-Check (★ ArrowTypeError fix) ===========
else:
    st.header("③ 교차검증 (AI pLDDT ↔ VQE 잔기 점수)")

    if not st.session_state.get("pdb_text"):
        st.info("먼저 ① Protein 탭에서 PDB를 불러오세요.")
    else:
        pdb_text = st.session_state["pdb_text"]
        plddt_arr, per_res = _parse_plddt_cached(pdb_text)

        if per_res is None:
            st.warning("이 PDB에는 pLDDT(b-factor)가 없을 수 있어요.")
        else:
            st.subheader("잔기 점수 CSV 업로드")
            up = st.file_uploader("residue_scores (chain,resi,score)", type=["csv"], key="rescsv")
            if up:
                df = pd.read_csv(up)
            else:
                if st.checkbox("데모 점수 사용", value=False):
                    df = pd.read_csv("sample_data/residue_scores_demo.csv")
                else:
                    df = None

            if df is not None and not df.empty:
                # ---- (1) per_res(index=tuple) → 원시 컬럼으로 변환
                plddt_df = pd.DataFrame({
                    "chain": [c for (c, r) in per_res.index],
                    "resi":  [int(r) for (c, r) in per_res.index],
                    "plddt": per_res.values,
                })

                # ---- (2) 타입 확정 (원시 타입만 남기기)
                df["chain"] = df["chain"].astype(str)
                df["resi"]  = pd.to_numeric(df["resi"], errors="coerce").astype("Int64")
                df["score"] = pd.to_numeric(df["score"], errors="coerce").astype(float)
                plddt_df["plddt"] = pd.to_numeric(plddt_df["plddt"], errors="coerce").astype(float)

                # ---- (3) merge (tuple key 사용 X)
                merged = plddt_df.merge(
                    df[["chain", "resi", "score"]].dropna(),
                    on=["chain", "resi"],
                    how="inner",
                )

                if merged.empty:
                    st.error("매칭되는 (chain,resi)가 없습니다. 체인/번호를 확인하세요.")
                else:
                    # ---- (4) 파생 수치 (모두 float)
                    merged["instability"] = 1.0 - merged["plddt"].clip(0, 100) / 100.0
                    merged["resi"] = merged["resi"].astype(int)  # Int64 → int

                    # 피어슨 r
                    corr = float(
                        pd.Series(merged["instability"]).corr(merged["score"])
                    )

                    # ---- (5) 시각화 + 3D 오버레이
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        st.metric("Pearson r (1-pLDDT vs score)", f"{corr:.3f}")
                        fig = plt.figure(figsize=(4, 3))
                        plt.scatter(merged["instability"], merged["score"], s=10)
                        plt.xlabel("Instability (1 - pLDDT)")
                        plt.ylabel("Residue Score")
                        st.pyplot(fig)
                    with c2:
                        st.write("3D 오버레이 (잔기 점수)")
                        mapping = {
                            f"{row.chain}:{int(row.resi)}": float(row.score)
                            for row in merged.itertuples()
                        }
                        render_overlay_by_residue(pdb_text, mapping)

                    # ---- (6) 표 (원시 타입만 보여주기 → Arrow OK)
                    st.dataframe(
                        merged[["chain", "resi", "plddt", "instability", "score"]]
                        .sort_values("score", ascending=False)
                        .head(50)
                    )

    # -------- Run comparison (safe JSON) --------
    st.subheader("런 비교")
    run_dirs = sorted([p for p in P_RUNS.glob("*") if p.is_dir()])
    rows = []
    for d in run_dirs[-50:]:
        js = d / "summary.json"
        if not js.exists():
            continue
        meta = load_summary_safe(js)
        if not meta:
            continue
        rows.append(
            {
                "run": d.name,
                "E_min": meta.get("E_min"),
                "delta_e": meta.get("delta_e"),
                "r2_exp": meta.get("r2_exp"),
                "var_last": meta.get("var_last"),
                "backend": meta.get("backend"),
                "mode": meta.get("mode"),
            }
        )

    if rows:
        df_runs = pd.DataFrame(rows).sort_values("E_min")
        st.dataframe(df_runs)
        st.download_button(
            "runs_summary.csv 다운로드",
            data=df_runs.to_csv(index=False).encode(),
            file_name="runs_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("아직 유효한 VQE 런이 없습니다.")
