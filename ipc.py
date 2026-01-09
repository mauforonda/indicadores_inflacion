#!/usr/bin/env python3

import pandas as pd
import datetime as dt
import requests
from bs4 import BeautifulSoup
from supabase import create_client
import os
from time import sleep
import argparse

SB_URL = os.environ["SUPABASE_URL"]
SB_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def descargar_excel(url, link_texto, filename):
    def encontrar_link(url, link_texto):
        r = requests.get(url)
        html = BeautifulSoup(r.text, "html.parser")
        return [a for a in html.select("#main a") if a.get_text() == link_texto][0][
            "href"
        ]

    link = encontrar_link(url, link_texto)
    with open(filename, "wb") as f:
        f.write(requests.get(link).content)


def extraer_indice(xl, sheet_query, name):
    meses = {
        "Enero": 1,
        "Febrero": 2,
        "Marzo": 3,
        "Abril": 4,
        "Mayo": 5,
        "Junio": 6,
        "Julio": 7,
        "Agosto": 8,
        "Septiembre": 9,
        "Octubre": 10,
        "Noviembre": 11,
        "Diciembre": 12,
    }

    # Select and read the sheet
    sheet = [s for s in xl.sheet_names if sheet_query in s.lower()][0]
    df = pd.read_excel(xl, sheet, skiprows=4)

    # Select only rows with month names
    df = df[df.MES.isin(meses.keys())]

    # Simplify the table
    df = df.set_index("MES").stack().reset_index()
    df.columns = ["mes", "año", name]

    # Format dates
    df["fecha"] = pd.to_datetime(
        df.apply(lambda _: dt.date(int(_["año"]), meses[_["mes"]], 1), axis=1)
    )
    df = df[["fecha", name]]

    # Filter and type valid indice values
    df = df[~df[name].apply(lambda _: type(_) == str)]
    df[name] = df[name].astype(float)

    return df.set_index("fecha")[name].sort_index()


def extract_ciudad(xl, ciudad):
    # Read the sheet
    df = pd.read_excel(xl, ciudad, skiprows=4)

    # Select valid rows
    df = df[df["CÓDIGO"].fillna("").str.strip().str.isdigit()]

    # Simplify the table
    df = df.set_index(["CÓDIGO", "DESCRIPCIÓN"]).stack().reset_index()
    df.columns = ["codigo", "producto", "fecha", "indice"]

    # Type each column right
    df.fecha = pd.to_datetime(df.fecha)
    df.indice = df.indice.astype(float)
    df.insert(0, "ciudad", ciudad)
    df.codigo = df.codigo.str.strip()

    return df


def indice_nacional():
    fn = "indice.xlsx"
    descargar_excel(
        "https://www.ine.gob.bo/index.php/nacional/",
        "Índice General, Variación Mensual, Acumulada y a 12 Meses",
        fn,
    )
    xl = pd.ExcelFile(fn)
    df = pd.concat(
        [
            extraer_indice(xl, sheet, name)
            for sheet, name in zip(
                ["ndice mensual", "var mensual", "var acumulada", "12 meses"],
                [
                    "indice_mensual",
                    "variacion_mensual",
                    "variacion_acumulada",
                    "variacion_12_meses",
                ],
            )
        ],
        axis=1,
    ).reset_index()
    print(f"nacional: {df.shape[0]} filas")
    return df


def indice_producto_ciudad():
    fn = "indice_producto_ciudad.xlsx"
    descargar_excel(
        "https://www.ine.gob.bo/index.php/ciudades-y-conurbaciones/",
        "Índices a nivel producto",
        fn,
    )
    xl = pd.ExcelFile(fn)
    ciudades = [s for s in xl.sheet_names if s != "Inicio"]
    df = pd.concat([extract_ciudad(xl, ciudad) for ciudad in ciudades])
    print(f"producto_ciudad: {df.shape[0]} filas")
    return df


def guardar_supabase(df, tabla, fake=False):
    def last_updated(supabase, tabla):
        r = (
            supabase.table(tabla)
            .select("fecha")
            .order("fecha", desc=True)
            .limit(1)
            .execute()
        )

        rows = getattr(r, "data", None) or []
        if not rows:
            return None

        return pd.Timestamp(rows[0]["fecha"])

    print(f"guardar {tabla}")
    if fake:
        df.to_csv(f"{tabla}.csv", index=False)
    else:
        chunk_size = 5000
        sleep_s = 0.2
        supabase = create_client(SB_URL, SB_KEY)
        updated = last_updated(supabase, tabla)
        if updated:
            print(f"última actualización: {updated.strftime('%Y-%m-%d')}")
            df = df[df.fecha > updated].copy()

        n = len(df)
        if n == 0:
            print("no existen filas nuevas")
            return

        print(f"existen {n} filas nuevas")
        df.fecha = df.fecha.dt.strftime("%Y-%m-%d")
        for i in range(0, n, chunk_size):
            print(f"{n if i + chunk_size > n else i + chunk_size} filas")
            chunk = df.iloc[i : i + chunk_size]
            supabase.table(tabla).insert(chunk.to_dict(orient="records")).execute()
            sleep(sleep_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y guarda datos del IPC")
    parser.add_argument("--fake", action="store_true", help="Save to CSV instead of Supabase")
    args = parser.parse_args()

    nacional = indice_nacional()

guardar_supabase(nacional, "ine_ipc_nacional", fake=args.fake)
producto_ciudad = indice_producto_ciudad()
guardar_supabase(producto_ciudad, "ine_ipc_producto_ciudad", fake=args.fake)

