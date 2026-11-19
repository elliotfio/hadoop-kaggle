import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession

st.set_page_config(page_title="Olist — Logistique & Satisfaction", layout="wide")
st.title("Olist — Performance logistique & satisfaction client")
st.caption(
    "Marketplace brésilienne (2016-2018). Lecture exclusive des tables pré-calculées "
    "de la couche **Gold** (`/lakehouse/gold/`) — jamais les CSV bruts."
)


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(hdfs_path: str):
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


# ------------------------------------------------------------------ chargement
cat = load_gold_table("/lakehouse/gold/revenue_by_category")
monthly = load_gold_table("/lakehouse/gold/monthly_performance").sort_values("purchase_month")
states = load_gold_table("/lakehouse/gold/state_logistics")
segments = load_gold_table("/lakehouse/gold/customer_segments")

# --------------------------------------------------------------------- filtres
st.sidebar.header("Filtres")
top_n = st.sidebar.slider("Nombre de catégories affichées (Pareto)", 5, 40, 15)
state_opts = sorted(states["customer_state"].tolist())
sel_states = st.sidebar.multiselect(
    "États du client", state_opts, default=state_opts
)
states_f = states[states["customer_state"].isin(sel_states)]

# ------------------------------------------------------------------------- KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Chiffre d'affaires", f"{cat['revenue_total'].sum() / 1e6:.1f} M R$")
k1.caption("Somme sur toutes les catégories")
k2.metric("Commandes livrées", f"{int(cat['nb_orders'].sum()):,}".replace(",", " "))
k3.metric("Note moyenne", f"{segments['avg_review_score'].mean():.2f} / 5")
k4.metric("Clients segmentés", f"{len(segments):,}".replace(",", " "))

st.divider()

# -------------------------------------------------- 1. Pareto par catégorie
st.subheader("1. Concentration du chiffre d'affaires par catégorie (Pareto)")
cat_top = cat.sort_values("revenue_rank").head(top_n)
fig1 = px.bar(
    cat_top, x="main_category", y="revenue_total",
    color="late_rate_pct", color_continuous_scale="Reds",
    labels={"main_category": "Catégorie", "revenue_total": "CA (R$)",
            "late_rate_pct": "% retard"},
)
fig1.add_scatter(
    x=cat_top["main_category"], y=cat_top["cumul_pct"],
    mode="lines+markers", name="% cumulé", yaxis="y2",
)
fig1.update_layout(
    yaxis2=dict(title="% cumulé", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig1, width="stretch")
with st.expander("Requête SQL & justification statistique"):
    st.code(
        """SELECT main_category,
       COUNT(*) AS nb_orders,
       ROUND(SUM(order_total_value), 2) AS revenue_total,
       PERCENTILE_APPROX(order_total_value, 0.50) AS median_order_value,
       DENSE_RANK() OVER (ORDER BY revenue_total DESC) AS revenue_rank,
       ROUND(100 * SUM(revenue_total) OVER (
             ORDER BY revenue_total DESC
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) / SUM(revenue_total) OVER (), 2) AS cumul_pct
FROM v_silver GROUP BY main_category;""",
        language="sql",
    )
    st.markdown(
        "**Loi de Pareto :** ~18 catégories sur 74 réalisent 80 % du CA. La couleur "
        "signale le taux de retard : elle révèle les catégories à fort CA ET à risque logistique."
    )

# ------------------------------------------------- 2. Dynamique mensuelle
st.subheader("2. Dynamique mensuelle : chiffre d'affaires vs taux de retard")
fig2 = px.line(
    monthly, x="purchase_month", y="revenue_total", markers=True,
    labels={"purchase_month": "Mois", "revenue_total": "CA (R$)"},
)
fig2.add_bar(
    x=monthly["purchase_month"], y=monthly["late_rate_pct"],
    name="% retard", yaxis="y2", opacity=0.4, marker_color="crimson",
)
fig2.update_layout(
    yaxis2=dict(title="% retard", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig2, width="stretch")
with st.expander("Requête SQL & justification statistique"):
    st.code(
        """SELECT purchase_month,
       COUNT(*) AS nb_orders,
       ROUND(SUM(order_total_value), 2) AS revenue_total,
       ROUND(100 * AVG(CAST(is_late AS INT)), 2) AS late_rate_pct,
       LAG(revenue_total, 1) OVER (ORDER BY purchase_month) AS prev_month_revenue
FROM v_silver WHERE purchase_month >= '2017-01' GROUP BY purchase_month;""",
        language="sql",
    )
    st.markdown(
        "**LAG() :** mesure la croissance mois sur mois. Pic de retards en nov. 2017 "
        "(Black Friday, 12,4 %) corrélé à une chute de la note moyenne."
    )

# ------------------------------------------------- 3. Retards par État
st.subheader("3. Taux de retard par État du client")
states_s = states_f.sort_values("late_rate_pct", ascending=False)
fig3 = px.bar(
    states_s, x="customer_state", y="late_rate_pct",
    color="median_delivery_days", color_continuous_scale="OrRd",
    labels={"customer_state": "État", "late_rate_pct": "% retard",
            "median_delivery_days": "Délai médian (j)"},
)
st.plotly_chart(fig3, width="stretch")
with st.expander("Requête SQL & justification statistique"):
    st.code(
        """SELECT customer_state,
       COUNT(*) AS nb_orders,
       PERCENTILE_APPROX(delivery_time_days, 0.50) AS median_delivery_days,
       ROUND(100 * AVG(CAST(is_late AS INT)), 2) AS late_rate_pct
FROM v_silver GROUP BY customer_state;""",
        language="sql",
    )
    st.markdown(
        "**Médiane** du délai (plutôt que moyenne) car la distribution est fortement "
        "asymétrique à droite (skewness ≈ 3,8). Les États du Nord/Nord-Est cumulent délais et retards."
    )

# ------------------------------------------------- 4. Segmentation clients
st.subheader("4. Segmentation des clients (K-Means)")
seg_counts = segments.groupby("segment").size().reset_index(name="nb_clients")
c_left, c_right = st.columns([1, 1])
fig4 = px.pie(seg_counts, names="segment", values="nb_clients", hole=0.4)
c_left.plotly_chart(fig4, width="stretch")
seg_profile = (
    segments.groupby("segment")[["avg_review_score", "late_rate", "total_spend"]]
    .median().reset_index()
)
fig5 = px.bar(
    seg_profile, x="segment", y="avg_review_score", color="late_rate",
    color_continuous_scale="RdYlGn_r",
    labels={"segment": "Segment", "avg_review_score": "Note médiane",
            "late_rate": "Taux retard"},
)
fig5.update_xaxes(showticklabels=False)
c_right.plotly_chart(fig5, width="stretch")
with st.expander("Requête SQL & justification statistique"):
    st.code(
        """-- Features agrégées par client puis K-Means (k=4) sur variables log1p + standardisées
SELECT customer_unique_id,
       SUM(order_total_value) AS total_spend,
       AVG(delivery_time_days) AS avg_delivery_days,
       AVG(review_score) AS avg_review_score,
       AVG(CAST(is_late AS INT)) AS late_rate
FROM v_silver GROUP BY customer_unique_id;""",
        language="sql",
    )
    st.markdown(
        "**log1p + StandardScaler** avant K-Means : on corrige l'asymétrie des montants "
        "et on met les features à la même échelle pour que la distance euclidienne soit équitable. "
        "Le segment *Mécontents — livraison en retard* confirme le lien retard → insatisfaction "
        "(test de Mann-Whitney, note médiane 1 vs 5)."
    )
