import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# -------------------------------
# Load default dataset
# -------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")


# -------------------------------
# Title
# -------------------------------
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("Segment mall customers using **K-Means Clustering** and explore tailored marketing strategies.")

# -------------------------------
# Upload or use default dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    df = load_default_data()
    st.info("ℹ️ Using default **Mall_Customers.csv** dataset")

# -------------------------------
# Data Preprocessing
# -------------------------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(int)
elif "Genre" in df.columns:
    df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1}).astype(int)

# Select numeric features
X = df.select_dtypes(include=["int64", "float64"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("⚙️ Clustering Options")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
features = st.sidebar.multiselect("Select features for clustering", X.columns.tolist(), default=X.columns.tolist())

if features:
    X_scaled = scaler.fit_transform(df[features])

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# -------------------------------
# Tabs Layout
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🔍 EDA", "📈 Clustering", "💡 Insights"])

# --- Dataset Tab ---
with tab1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Shape:", df.shape)

    # Download buttons
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download as CSV", csv, "segmented_customers.csv", "text/csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Customers")
    st.download_button("📊 Download as Excel", buffer.getvalue(), "segmented_customers.xlsx", "application/vnd.ms-excel")

# --- EDA Tab ---
with tab2:
    st.subheader("🔍 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**📊 Summary Statistics**")
        st.dataframe(df.describe(), use_container_width=True)
        
        if len(features) > 1:
            st.write("**🔗 Feature Correlations**")
            corr_matrix = df[features].corr()
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Feature Correlation Matrix",
                               color_continuous_scale="RdBu_r")
            fig_corr.update_layout(height=400, width=500)
            st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.write("**📈 Feature Distributions**")
        for i, feature in enumerate(features[:4]):  # Show up to 4 features
            fig = px.histogram(df, x=feature, 
                             nbins=25, 
                             title=f"Distribution of {feature}",
                             marginal="box",  # Add box plot on top
                             color_discrete_sequence=['#636EFA'])
            fig.update_layout(
                height=300,
                title_font_size=14,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

# --- Clustering Tab ---
with tab3:
    st.subheader("📈 Cluster Visualization (PCA 2D)")
    fig = px.scatter(df, x="PCA1", y="PCA2", color="Cluster",
                     title="Customer Clusters (PCA Projection)",
                     hover_data=df.columns,
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     size_max=10)
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='white')))
    fig.update_layout(
        height=500,
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Cluster Distribution")
    col5, col6 = st.columns(2)
    
    with col5:
        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_bar = px.bar(cluster_counts,
                        x="Cluster", y="Count",
                        title="Customer Count by Cluster",
                        color="Count",
                        color_continuous_scale="viridis",
                        text="Count")
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col6:
        fig_pie = px.pie(cluster_counts, values="Count", names="Cluster",
                        title="Cluster Proportions",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📌 Cluster Profiles (Means)")
    cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_profiles.style.background_gradient(cmap='RdYlBu_r'), use_container_width=True)
    
    if len(features) >= 3:
        st.subheader("🎯 Cluster Comparison (Radar Chart)")
        selected_clusters = st.multiselect("Select clusters to compare:", 
                                         sorted(df["Cluster"].unique()), 
                                         default=sorted(df["Cluster"].unique())[:3],
                                         key="clustering_tab_multiselect")
        
        if selected_clusters:
            import plotly.graph_objects as go
            
            fig_radar = go.Figure()
            colors = px.colors.qualitative.Set3[:len(selected_clusters)]
            
            for i, cluster in enumerate(selected_clusters):
                cluster_data = df[df["Cluster"] == cluster][features].mean()
                radar_values = cluster_data.tolist() + [cluster_data.tolist()[0]]  # Close the radar
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=features + [features[0]],
                    fill='toself',
                    fillcolor=colors[i],
                    line=dict(color=colors[i], width=3),
                    opacity=0.6,
                    name=f'Cluster {cluster}'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max([df[f].max() for f in features])]
                    )),
                showlegend=True,
                title="Cluster Feature Comparison (Multi-Cluster Radar)",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# --- Insights Tab ---
with tab4:
    st.subheader("💡 Marketing Strategies per Cluster")
    
    st.write("**📊 Cluster Overview**")
    cluster_summary = df.groupby("Cluster").agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean' if 'Annual Income (k$)' in df.columns else 'count',
        'Spending Score (1-100)': 'mean' if 'Spending Score (1-100)' in df.columns else 'count'
    }).round(2)
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    st.write("**🎯 Detailed Marketing Strategies**")
    for cluster_id, profile in df.groupby("Cluster").mean(numeric_only=True).iterrows():
        with st.expander(f"📌 Cluster {cluster_id} Strategy (Click to expand)"):
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                st.write("**Cluster Profile:**")
                st.dataframe(profile.to_frame(name="Average Value"), use_container_width=True)
            
            with col_b:
                st.write("**Recommended Strategy:**")
                if "Spending Score (1-100)" in profile.index:
                    score = profile["Spending Score (1-100)"]
                    income = profile.get("Annual Income (k$)", 50)
                    age = profile.get("Age", 35)
                    
                    if score > 60:
                        st.success("🟢 **High Spenders** — Premium product lines, VIP membership programs, exclusive early access to new products, personalized luxury experiences.")
                    elif score < 40:
                        st.warning("🔵 **Budget Conscious** — Value deals, seasonal discounts, loyalty point systems, budget-friendly product recommendations.")
                    else:
                        st.info("🟡 **Moderate Spenders** — Balanced promotional offers, product bundles, targeted seasonal campaigns.")
                    
                    # Additional demographic insights
                    if age > 45:
                        st.write("👥 **Age Factor:** Focus on quality, reliability, and traditional marketing channels.")
                    elif age < 30:
                        st.write("👥 **Age Factor:** Emphasize trends, social media marketing, and digital engagement.")
                    
                    if income > 70:
                        st.write("💰 **Income Factor:** Premium positioning and high-value propositions work well.")
                    elif income < 40:
                        st.write("💰 **Income Factor:** Cost-effective solutions and payment flexibility options.")
                        
                else:
                    st.info("ℹ️ **General Strategy:** Analyze customer behavior patterns and preferences for targeted marketing approaches.")
