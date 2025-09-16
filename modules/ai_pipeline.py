from __future__ import annotations
import os, re, io, json, shutil, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import requests

ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"

@dataclass
class AIPredictResult:
    pdb_path: Path
    uniprot_id: Optional[str] = None
    plddt_available: bool = False
    meta: dict = None

def is_tool(name: str) -> bool:
    return shutil.which(name) is not None

def fetch_uniprot_fasta(uniprot_id: str) -> Optional[str]:
    url = UNIPROT_FASTA_URL.format(uniprot=uniprot_id)
    try:
        r = requests.get(url, timeout=30)
        if r.ok and r.text.startswith(">"):
            return r.text
    except Exception:
        pass
    return None

def fetch_alphafold_pdb(uniprot_id: str, outdir: Path) -> Optional[Path]:
    url = ALPHAFOLD_PDB_URL.format(uniprot=uniprot_id)
    try:
        r = requests.get(url, timeout=60)
        if r.ok and "ATOM" in r.text:
            out = outdir / f"AF-{uniprot_id}-F1-model_v4.pdb"
            out.write_text(r.text, encoding="utf-8")
            return out
    except Exception:
        pass
    return None

def parse_fasta_text_to_seq(fasta_text: str) -> str:
    seq = []
    for line in fasta_text.splitlines():
        if not line or line.startswith(">"):
            continue
        seq.append(line.strip())
    return "".join(seq)

def predict_with_colabfold(seq_or_fasta: str, outdir: Path) -> Optional[Path]:
    """
    로컬에 colabfold_batch 가 있으면 사용합니다.
    - 입력이 서열(AA만) 또는 FASTA-text여도 OK (임시 FASTA 생성)
    - 산출물: outdir/*.pdb 중 가장 큰 파일 선택
    """
    if not is_tool("colabfold_batch"):
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    fasta_path = outdir / "input.fasta"
    if seq_or_fasta.strip().startswith(">"):
        fasta_path.write_text(seq_or_fasta, encoding="utf-8")
    else:
        fasta_path.write_text(">query\n" + seq_or_fasta.strip() + "\n", encoding="utf-8")

    cmd = [
        "colabfold_batch",
        str(fasta_path),
        str(outdir)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None

    pdbs = list(outdir.glob("*.pdb"))
    if not pdbs:
        return None
    # 가장 큰 파일(모델 본체일 확률이 높음)
    pdbs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return pdbs[0]

def ensure_plddt_coloring(pdb_text: str) -> Tuple[str, bool]:
    """
    pLDDT가 이미 B-factor 컬럼에 들어있는 AlphaFold PDB면 True 반환.
    아니면 False 반환(필요시 기본값 50.0으로 채워 넣는 로직을 구현할 수도 있음).
    """
    has_b = False
    for ln in pdb_text.splitlines():
        if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 66:
            try:
                float(ln[60:66])  # B-factor field
                has_b = True
                break
            except Exception:
                continue
    return pdb_text, has_b

def predict_structure(
    sequence_text: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    prefer_online: bool = True,
    outdir: Path = Path("runs/ai_predict")
) -> AIPredictResult:
    """
    우선순위:
      1) uniprot_id 있으면 → AlphaFold DB에서 PDB 다운로드 시도
      2) (또는) sequence_text 있으면 → ColabFold 로컬 추론 시도
      3) 둘 다 실패 시 예외 발생
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) UniProt ID → AlphaFold DB PDB
    if uniprot_id and prefer_online:
        pdb_path = fetch_alphafold_pdb(uniprot_id, outdir)
        if pdb_path and pdb_path.exists():
            text = pdb_path.read_text(encoding="utf-8", errors="ignore")
            text, has_b = ensure_plddt_coloring(text)
            pdb_path.write_text(text, encoding="utf-8")
            return AIPredictResult(pdb_path=pdb_path, uniprot_id=uniprot_id, plddt_available=has_b, meta={"source": "AlphaFoldDB"})

    # 2) Sequence/FASTA → ColabFold
    if sequence_text:
        # 서열만 들어왔으면 FASTA로 변환
        fasta = sequence_text if sequence_text.strip().startswith(">") else (">query\n" + sequence_text.strip() + "\n")
        pdb_path = predict_with_colabfold(fasta, outdir)
        if pdb_path and pdb_path.exists():
            text = pdb_path.read_text(encoding="utf-8", errors="ignore")
            text, has_b = ensure_plddt_coloring(text)
            pdb_path.write_text(text, encoding="utf-8")
            return AIPredictResult(pdb_path=pdb_path, uniprot_id=uniprot_id, plddt_available=has_b, meta={"source": "ColabFold"})

    raise RuntimeError("구조 예측 실패: 인터넷/AlphaFold 또는 로컬 ColabFold 둘 다 사용할 수 없습니다.")
