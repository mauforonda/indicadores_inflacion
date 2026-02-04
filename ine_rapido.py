#!/usr/bin/env python3

import requests
import pandas as pd
import argparse
from upload import upload_dataset

URL = "https://servicioswm.ine.gob.bo/canastita/dashboard/reporte2"
FILENAME = "ine_ipc_rapido"


def update(url):

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(raw).rename(columns={"id": "id_producto"})

    df["fecha"] = pd.to_datetime(
        df["dia"].astype(str).str.strip() + " " + df["gestion"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    df["precio"] = pd.to_numeric(df["precio_mercado"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["unidad"] = df["unidad_madre"].astype(str).str.lower()
    df["producto"] = df["producto"].astype(str).str.lower()

    price_series = (
        df.loc[
            df["precio"] > 0,
            [
                "fecha",
                "departamento",
                "producto",
                "id_producto",
                "precio",
                "unidad",
                "cantidad",
            ],
        ]
        .dropna(subset=["fecha"])
        .sort_values(["departamento", "producto", "fecha"])
        .reset_index(drop=True)
    )

    return price_series


def do_upload():
    parser = argparse.ArgumentParser(
        description="Descarga y guarda la serie de precios rapida de la canastita del INE."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Sube los datos a Supabase (por defecto solo guarda CSV).",
    )
    args = parser.parse_args()
    return args.upload


upload = do_upload()

df = update(URL)
df.to_csv(f"{FILENAME}.csv", index=False)
df.to_excel(f"{FILENAME}.xlsx", index=False)

if upload:
    upload_dataset(
        FILENAME, df, ["fecha", "departamento", "producto", "unidad", "cantidad"]
    )
