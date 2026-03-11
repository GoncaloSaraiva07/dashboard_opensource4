import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px

st.set_page_config(layout="wide")

st.title("Wine Export Strategy – UK Market")

st.markdown(
"""
This dashboard supports the strategic decision of exporting wine to the UK market
based on data-driven segmentation.
"""
)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("dataset__Wine_Trabalho_Segmentação.csv")

df = df.drop(columns=["Id","quality"])

# create variable
df["bound_sulfur_dioxide"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]

df = df.drop(columns=["free sulfur dioxide","total sulfur dioxide"])

# ==============================
# NORMALIZE DATA 0–1
# ==============================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

X_scaled = pd.DataFrame(X_scaled, columns=df.columns)

# ==============================
# KMEANS WITH 3 CLUSTERS
# ==============================

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

X_scaled["cluster"] = clusters

# ==============================
# EXECUTIVE KPIs
# ==============================

st.header("Executive Summary")

col1,col2,col3 = st.columns(3)

col1.metric("Wines analysed", len(df))
col2.metric("Clusters identified",3)
col3.metric("Variables analysed",len(df.columns))

# ==============================
# CLUSTER VISUALIZATION
# ==============================

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled.drop(columns=["cluster"]))

X_scaled["pca1"] = pca_result[:,0]
X_scaled["pca2"] = pca_result[:,1]

fig = px.scatter(
X_scaled,
x="pca1",
y="pca2",
color="cluster",
title="Wine Clusters"
)

st.plotly_chart(fig,use_container_width=True)

# ==============================
# CLUSTER PROFILES
# ==============================

st.header("Cluster Profiles")

cluster_profile = X_scaled.groupby("cluster").mean()

st.dataframe(cluster_profile)

# ==============================
# UK MARKET TARGET PROFILE
# ==============================

uk_target = pd.Series({
"alcohol":0.35,
"fixed acidity":0.65,
"citric acid":0.60,
"sulphates":0.55
})

scores = {}

for cluster in cluster_profile.index:

    profile = cluster_profile.loc[cluster]

    score = (
        abs(profile["alcohol"]-uk_target["alcohol"])
        + abs(profile["fixed acidity"]-uk_target["fixed acidity"])
        + abs(profile["citric acid"]-uk_target["citric acid"])
        + abs(profile["sulphates"]-uk_target["sulphates"])
    )

    scores[cluster] = score

best_cluster = min(scores, key=scores.get)

st.header("Recommended Cluster for UK Export")

st.success(f"Cluster {best_cluster} shows the best alignment with UK market preferences.")

st.write(cluster_profile.loc[best_cluster])

# ==============================
# COMMERCIAL STRATEGY
# ==============================

st.header("Commercial Strategy")

st.markdown(
"""
Recommended actions:

• Focus on wines in the selected cluster  
• Position product as fresh and modern  
• Target UK importers and wine bars  
• Maintain competitive pricing  

This profile aligns with UK consumer trends favouring
fresh wines with moderate alcohol levels.
"""
)
