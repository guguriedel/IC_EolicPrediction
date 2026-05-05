"""
Fase 1.6: Coleta features macro (climaticas e socioeconomicas).

Fontes (todas publicas, sem chave):
  - Nino 3.4 anomaly (NOAA PSL):
      https://psl.noaa.gov/data/correlation/nina34.anom.data
  - AMM SST (NOAA PSL):
      https://psl.noaa.gov/data/timeseries/monthly/AMM/ammsst.data
  - IBC-Br (proxy mensal de PIB, BCB SGS codigo 24364):
      https://api.bcb.gov.br/dados/serie/bcdata.sgs.24364/dados?formato=json

Saida:
    Dados/features_macro.csv com colunas:
        data, nino34_anom, amm_sst, ibc_br

Cada coluna pode estar com NaN no inicio/fim conforme cobertura da fonte.
A integracao no dataset principal e feita por gerar_dataset_limpo.py
(merge left por 'data').

Uso:
    python scripts/coletar_features_macro.py
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
OUT_CSV = DADOS / "features_macro.csv"
OUT_RELATORIO = DADOS / "features_macro_relatorio.json"

URL_NINO34 = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
URL_AMM = "https://psl.noaa.gov/data/timeseries/monthly/AMM/ammsst.data"
URL_IBCBR = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.24364/dados?formato=json"

ANO_MIN = 2010
TIMEOUT = 30


def _fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 TCC-eolico"})
    with urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def parse_psl_table(raw: bytes, missing: float = -99.99) -> pd.Series:
    """Le formato NOAA PSL: primeira linha 'AAAA AAAA' (range de anos), depois
    linhas 'AAAA m1 m2 ... m12'. Sentinela = -99.99."""
    txt = raw.decode("utf-8", errors="ignore").splitlines()
    # Achar primeira linha com 13 numeros (ano + 12 meses)
    registros: list[tuple[pd.Timestamp, float]] = []
    for ln in txt:
        toks = ln.split()
        if len(toks) != 13:
            continue
        try:
            ano = int(toks[0])
            valores = [float(t) for t in toks[1:]]
        except ValueError:
            continue
        if ano < 1900 or ano > 2100:
            continue
        for m, v in enumerate(valores, start=1):
            if abs(v - missing) < 1e-3:
                v = np.nan
            registros.append((pd.Timestamp(ano, m, 1), v))
    s = pd.Series(
        {d: v for d, v in registros}, name="valor"
    ).sort_index()
    s.index.name = "data"
    return s


def coletar_nino34() -> pd.Series:
    raw = _fetch(URL_NINO34)
    s = parse_psl_table(raw)
    s.name = "nino34_anom"
    return s


def coletar_amm() -> pd.Series:
    raw = _fetch(URL_AMM)
    s = parse_psl_table(raw)
    s.name = "amm_sst"
    return s


def coletar_ibcbr() -> pd.Series:
    raw = _fetch(URL_IBCBR)
    obj = json.loads(raw.decode("utf-8"))
    df = pd.DataFrame(obj)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    s = df.set_index("data")["valor"].sort_index()
    s.name = "ibc_br"
    # BCB ja e mensal; normalizar para inicio do mes
    s.index = s.index.to_period("M").to_timestamp()
    return s


def main() -> int:
    DADOS.mkdir(exist_ok=True)
    relatorio: dict = {"fontes": {}, "linhas": {}, "periodo": {}}
    series: dict[str, pd.Series] = {}

    for nome, fn, url in [
        ("nino34_anom", coletar_nino34, URL_NINO34),
        ("amm_sst", coletar_amm, URL_AMM),
        ("ibc_br", coletar_ibcbr, URL_IBCBR),
    ]:
        try:
            s = fn()
            s = s[s.index.year >= ANO_MIN]
            series[nome] = s
            relatorio["fontes"][nome] = {"url": url, "ok": True, "n": int(s.notna().sum())}
            relatorio["periodo"][nome] = [str(s.index.min().date()), str(s.index.max().date())]
            print(f"OK  {nome}: {s.notna().sum()} obs ({s.index.min().date()} -> {s.index.max().date()})")
        except (HTTPError, URLError, ValueError, KeyError) as e:
            relatorio["fontes"][nome] = {"url": url, "ok": False, "erro": repr(e)}
            print(f"ERRO {nome}: {e}", file=sys.stderr)

    if not series:
        print("Nenhuma fonte coletada. Abortando.", file=sys.stderr)
        OUT_RELATORIO.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
        return 1

    df = pd.concat(series.values(), axis=1).sort_index()
    df.index.name = "data"
    df = df.reset_index()
    df.to_csv(OUT_CSV, index=False)

    relatorio["linhas"] = len(df)
    relatorio["colunas"] = list(df.columns)
    relatorio["out_csv"] = str(OUT_CSV)
    OUT_RELATORIO.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    print(f"\nOK: {OUT_CSV} ({len(df)} linhas, {len(df.columns)} colunas)")
    print(f"OK: {OUT_RELATORIO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
