import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Set page config
st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Mall_Customers.csv")
    data = data.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'SpendingScore'})
    return data

data = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Clustering"])

# Main content
if page == "Data Overview":
    st.title("Customer Segmentation Analysis")
    st.header("Data Overview")
    
    st.subheader("First 5 rows of the dataset")
    st.write(data.head())
    
    st.subheader("Dataset Shape")
    st.write(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")
    
    st.subheader("Descriptive Statistics")
    st.write(data.describe())
    
    st.subheader("Data Types")
    st.write(data.dtypes)
    
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    st.write("Percentage of missing values:")
    st.write((data.isnull().sum()/len(data))*100)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Age'], bins=25, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Income Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Income'], bins=35, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Spending Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['SpendingScore'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Gender', data=data, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Income vs Spending Score by Gender")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Income', y='SpendingScore', hue='Gender', data=data, ax=ax)
    st.pyplot(fig)

elif page == "Clustering":
    st.title("Customer Segmentation using Clustering")
    
    # Select features for clustering
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose features to cluster on", 
                            ['Age', 'Income', 'SpendingScore'],
                            default=['Income', 'SpendingScore'])
    
    # Select clustering method
    method = st.selectbox("Select Clustering Method", 
                         ["K-Means", "Hierarchical"])
    
    # Number of clusters
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    if st.button("Run Clustering"):
        if len(features) < 2:
            st.warning("Please select at least 2 features for clustering")
        else:
            # Prepare data
            X = data[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if method == "K-Means":
                # K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate metrics
                silhouette = silhouette_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                
                # Plot clusters
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster_labels, 
                               palette='viridis', ax=ax)
                plt.title(f"K-Means Clustering (n_clusters={n_clusters})")
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                
                # Plot centroids
                centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                plt.scatter(centroids[:, 0], centroids[:, 1], 
                            marker='X', s=100, c='red', label='Centroids')
                plt.legend()
                st.pyplot(fig)
                
            else:
                # Hierarchical clustering
                linkage_matrix = linkage(X_scaled, method='ward')
                cluster_labels = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity='euclidean', linkage='ward'
                ).fit_predict(X_scaled)
                
                # Calculate metrics
                silhouette = silhouette_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                
                # Plot clusters
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Dendrogram
                dendrogram(linkage_matrix, ax=ax1)
                ax1.set_title("Dendrogram")
                
                # Cluster plot
                sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster_labels, 
                               palette='viridis', ax=ax2)
                ax2.set_title(f"Hierarchical Clustering (n_clusters={n_clusters})")
                ax2.set_xlabel(features[0])
                ax2.set_ylabel(features[1])
                st.pyplot(fig)
            
            # Display metrics
            st.subheader("Clustering Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{silhouette:.3f}")
            col2.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}")
            
            # Add cluster labels to data
            data['Cluster'] = cluster_labels
            
            # Show cluster statistics
            st.subheader("Cluster Statistics")
            st.write(data.groupby('Cluster')[features].mean())
            
            # Download clustered data
            st.download_button(
                label="Download Clustered Data",
                data=data.to_csv(index=False),
                file_name="clustered_customers.csv",
                mime="text/csv"
            )
