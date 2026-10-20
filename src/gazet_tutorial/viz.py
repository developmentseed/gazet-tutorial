"""Plots and maps for the notebooks."""

from __future__ import annotations

import json

import pandas as pd


def plot_loss(log_history: list[dict], ax=None):
    """Training (and eval) loss from trainer.state.log_history."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(log_history)
    ax = ax or plt.gca()
    if "loss" in df:
        df.dropna(subset=["loss"]).plot(x="step", y="loss", ax=ax, label="train loss")
    if "eval_loss" in df:
        df.dropna(subset=["eval_loss"]).plot(x="step", y="eval_loss", ax=ax, marker="o", label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    return ax


def plot_family_bars(results: dict[str, pd.DataFrame], metric: str = "exec_match", ax=None):
    """Grouped bars of `metric` per task_family for several models."""
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {label: df.groupby("task_family")[metric].mean() for label, df in results.items()}
    )
    ax = ax or plt.gca()
    table.plot.bar(ax=ax, rot=30, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="")
    return ax


def to_geodataframe(df: pd.DataFrame, geometry_col: str = "geometry"):
    """Result DataFrame with GeoJSON-string geometry → GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd
    from shapely.geometry import shape

    geoms = [shape(json.loads(g)) if isinstance(g, str) else None for g in df[geometry_col]]
    props = df.drop(columns=[geometry_col])
    for col in props.columns:
        if props[col].map(lambda v: isinstance(v, (list, dict))).any():
            props[col] = props[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")


# Basemap for every map in the tutorial. OpenStreetMap's tile server refuses requests
# from Colab, and the CartoDB entry in xyzservices now asks for an API key; Esri's
# light-gray canvas needs neither and works in Colab and locally.
DEFAULT_TILES = "Esri.WorldGrayCanvas"


def map_features(result_df: pd.DataFrame, anchors=None, tooltip: str | list | None = None, simplify: float = 0.002,
                 tiles: str = DEFAULT_TILES):
    """Interactive folium map of a query result, with optional anchor outlines.

    `anchors` is a GeoDataFrame or result DataFrame of the places the user asked
    about, drawn as dashed outlines so you can see what the result is relative to.
    `tiles` is any xyzservices provider name or folium tile layer; the default is a
    key-less basemap that also works inside Colab (see DEFAULT_TILES).
    """
    import geopandas as gpd

    gdf = result_df if isinstance(result_df, gpd.GeoDataFrame) else to_geodataframe(result_df)
    if simplify:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
    if tooltip is None:
        tooltip = [c for c in ("name", "subtype", "region", "area", "rn") if c in gdf.columns] or True
    m = gdf.explore(tooltip=tooltip, color="#1f77b4", style_kwds={"fillOpacity": 0.35, "weight": 1}, tiles=tiles)
    if anchors is not None and len(anchors):
        agdf = anchors if isinstance(anchors, gpd.GeoDataFrame) else to_geodataframe(anchors)
        agdf.explore(m=m, color="#d62728", style_kwds={"fill": False, "weight": 2.5, "dashArray": "6 4"},
                     tooltip=[c for c in ("name", "subtype") if c in agdf.columns] or True)
    return m
