import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

st.title("📊 Heatmap (Correlation + Significance Anots)")

uploaded_file = st.file_uploader("Korelasyon matrisi içeren bir CSV dosyası yükleyin", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, sep=';', index_col=0)

    st.subheader("📁 Yüklenen Veri")
    st.dataframe(df_raw)

    df_corr = df_raw.copy()
    df_annot = df_raw.copy()

    def split_corr_and_stars(cell):
        match = re.match(r"(-?\d*\.\d+)(\*+)?", str(cell).strip())
        if match:
            corr = float(match.group(1))
            stars = match.group(2) if match.group(2) else ''
        else:
            corr = np.nan
            stars = ''
        return corr, stars

    for i in df_raw.index:
        for j in df_raw.columns:
            corr, stars = split_corr_and_stars(df_raw.loc[i, j])
            df_corr.loc[i, j] = corr
            df_annot.loc[i, j] = stars

    np.fill_diagonal(df_corr.values, 1.0)
    np.fill_diagonal(df_annot.values, '')

    # Maskeleme işlemi (üst üçgeni gizle)
    mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)

    df_corr = df_corr.astype(float)

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 12))

    im = sns.heatmap(df_corr.astype(float),
                          mask=mask,
                          cmap="coolwarm",
                          vmin=-1, vmax=1,
                          square=True,
                          linewidths=0.5,
                          annot=df_annot,
                          fmt="",
                          xticklabels=False, yticklabels=True,
                          cbar=False,
                          annot_kws={"size": 9})

    #im = sns.heatmap(df_corr, annot=df_annot, fmt="", cmap="coolwarm", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_yticklabels(df_corr.index, rotation=0, fontsize=9)

    ax.set_xticks(np.arange(len(df_corr.columns)) + 0.5)
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.8)

    cbar = plt.colorbar(im.collections[0],
                        cax=cax,
                        orientation='horizontal',
                        ticks=np.arange(-1, 1.1, 0.2))

    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)

    # Aralık çizgileri
    for tick in np.arange(-1, 1.1, 0.2):
        cbar.ax.axvline(tick, color='white', lw=0.5, linestyle='--')
        
    st.pyplot(fig)
