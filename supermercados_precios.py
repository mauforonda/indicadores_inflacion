#!/usr/bin/env python

import glob
import os

import numpy as np
import pandas as pd

import sklearn
import unidecode


SKIP_SUBCATS = {
    "mascotas",
    "churrasqueria",
    "limpieza de zapatos",
    "cotillon",
    "mascotas general",
    "libreria",
    "living y dormitorio",
    "menaje plastico",
    "bijouteria/cuidado personal",
}


def load_products(products_csv):
    df = pd.read_csv(products_csv)
    df["subcategoria_"] = df["subcategoria"].apply(
        lambda s: unidecode.unidecode(s).lower()
    )
    return df[~df["subcategoria_"].isin(SKIP_SUBCATS)].copy()


def load_department_price_frames(base_dir):
    out = {}
    for dept_path in sorted(glob.glob(os.path.join(base_dir, "*/"))):
        files = sorted(glob.glob(os.path.join(dept_path, "*")))
        df = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)

        dept_name = os.path.basename(os.path.normpath(dept_path))
        out[dept_name] = df

    return out


# exact 1m rolling window
def rolling_1m(s):
    return [s.loc[t - pd.DateOffset(months=1) : t].sum() for t in s.index]


def get_inflation(dept_df, products_df):
    # pivot prices so rows are products and columns are dates
    df = dept_df.groupby(["fecha", "id_producto"]) ["precio"].mean()
    df = df.unstack(level=0)
    df.columns = pd.to_datetime(df.columns)

    # product filtering: exclude products with many missing values
    denom = (df.isna().T != df.isna().T.shift()).iloc[1:].cumsum().iloc[-1]
    denom = denom.replace(0, df.shape[1])
    mask = (df.isna().T.sum() / denom) < 10
    mask = mask & (df.isna().T.sum() < df.shape[1] * 0.1)
    df_filtered = df[mask]

    # map products to normalized subcategory and keep large subcategories
    prod_to_subcat = (
        products_df.set_index("id_producto").reindex(df_filtered.index)["subcategoria_"]
    )
    subcat_counts = prod_to_subcat.value_counts()
    large_subcats = subcat_counts[subcat_counts > subcat_counts.quantile(0.33)].index
    prod_to_subcat = prod_to_subcat[prod_to_subcat.isin(large_subcats)]

    # compute daily log differences, forward-fill inside gaps, then mean by subcat
    daily = df.loc[prod_to_subcat.index].T.asfreq("D").ffill(limit_area="inside")
    daily_logs = np.log(daily)
    daily_diff = daily_logs.diff()
    daily_diff = daily_diff.T.fillna(0)
    rates = daily_diff.groupby(prod_to_subcat).mean()

    # map subcategories to categories (choose most common category per subcat)
    grp = products_df.groupby("subcategoria_")["categoria"]
    vc = grp.value_counts()
    most_common = vc.groupby(level=0).apply(lambda x: x.sort_values().index[-1][1])
    cat_map = most_common.loc[rates.index]

    cat_infl = np.exp(rates).groupby(cat_map).mean()
    cat_infl = cat_infl.T.iloc[1:]

    # remove noisy categories and restrict to recent dates
    noisy = ["Juguetería", "Juguetería Importación", "Bazar Importación"]
    rates_clean = rates.drop(noisy, errors="ignore").T
    rates_clean = rates_clean.loc["2024-08-01":]

    # apply 1-month window on already-windowed series
    rates_windowed = rates_clean.apply(rolling_1m).loc["2024-09-01":]
    rates_windowed = np.exp(rates_windowed)

    # PCA on standardized series, take first component
    scaler = sklearn.preprocessing.StandardScaler()

    X = scaler.fit_transform(rates_windowed)
    pca = sklearn.decomposition.PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    infl = pd.Series(X_pca[:, 0], index=rates_windowed.index)

    return cat_infl, infl


def run_all():
    # compute inflation for all departments and return (infl_df, cat_infl_df)
    data_dir = "../data/hipermaxi"
    products_csv = os.path.join(data_dir, "productos.csv")

    products_df = load_products(products_csv)
    dept_price_frames = load_department_price_frames(data_dir)

    cat_infl_map = {}
    infl_map = {}

    for dept, df in dept_price_frames.items():
        cat_infl, infl = get_inflation(df, products_df)
        cat_infl_map[dept] = cat_infl
        infl_map[dept] = infl

    # global inflation: average across departments
    infl_df = pd.DataFrame(infl_map).mean(axis=1)

    # category inflation: average across departments, 28-day cumprod
    cat_infl_df = np.log(
        pd.concat(cat_infl_map).groupby(level=1).mean()
    ).rolling(window=28).sum().loc["2024-09-01":]

    return infl_df, np.exp(cat_infl_df)


if __name__ == "__main__":
    infl_df, cat_infl_df = run_all()
