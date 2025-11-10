import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# Load the data
# Make sure to adjust the path to where you extracted the zip file
df = pd.read_csv('C:\\Users\\pampa\\Downloads\\SWCPQ-Features-Aggregated-Dataset-January2025\\data files\\characters-aggregated-scores.csv', sep='\t')

# First look at the data
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum(), "total missing values")



# Let's identify trait columns (usually numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nFound {len(numeric_cols)} numeric columns (traits)")

# Remove ID columns if present
trait_cols = [col for col in numeric_cols if not col.lower() in ['id', 'char_display_name', 'fictional_work']]

print(f"Using {len(trait_cols)} trait columns for clustering")

# Extract trait data
X = df[trait_cols].copy()

# Check for missing values
print(f"\nMissing values in traits: {X.isnull().sum().sum()}")

# Handle missing values if any (drop rows with missing data)
if X.isnull().sum().sum() > 0:
    X = X.dropna()
    print(f"After removing missing values: {X.shape[0]} samples")

# Keep a cleaned copy of the original dataframe aligned with X (drop rows that were removed)
df_clean = df.loc[X.index].copy()

# Standardize the data (CRITICAL for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFinal dataset for clustering: {X_scaled.shape}")
print(f"  - {X_scaled.shape[0]} characters")
print(f"  - {X_scaled.shape[1]} personality traits")





# ============================================
# MODEL SELECTION: FIND OPTIMAL NUMBER OF CLUSTERS

print("Starting model selection...")
print("Testing K from 2 to 15 clusters")
print("This will take a few minutes...\n")

# Test different numbers of clusters
n_components_range = range(2, 16)

# Store results
aic_scores = []
bic_scores = []

for n in n_components_range:
    print(f"Testing K={n}...", end=" ")
    
    # Fit GMM with K clusters
    gmm = GaussianMixture(
        n_components=n,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    
    gmm.fit(X_scaled)
    
    # Calculate AIC and BIC
    aic = gmm.aic(X_scaled)
    bic = gmm.bic(X_scaled)
    
    aic_scores.append(aic)
    bic_scores.append(bic)
    
    print(f"AIC={aic:.1f}, BIC={bic:.1f}, Converged={gmm.converged_}")

print("\n✓ Model selection complete!")

# ============================================
# VISUALIZE RESULTS (Jake VDP style)
# ============================================
plt.figure(figsize=(14, 6))

# Plot AIC
plt.subplot(1, 2, 1)
plt.plot(n_components_range, aic_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('AIC', fontsize=12)
plt.title('AIC Score (lower is better)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot BIC
plt.subplot(1, 2, 2)
plt.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8, color='red')
plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('BIC', fontsize=12)
plt.title('BIC Score (lower is better)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_selection_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved as 'model_selection_results.png'")

# ============================================
# FIND BEST K
# ============================================
best_k_aic = n_components_range[np.argmin(aic_scores)]
best_k_bic = n_components_range[np.argmin(bic_scores)]

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print(f"Best K by AIC: {best_k_aic}")
print(f"Best K by BIC: {best_k_bic}")
print("\nBIC is typically more conservative (favors simpler models)")
print("Look for the 'elbow' in the curve where it flattens out")
print("="*60)



# ============================================
# FIT FINAL MODEL WITH K=7
# ============================================
print("Fitting final GMM model with K=7...")
print("This may take a minute...\n")

OPTIMAL_K = 7

gmm_final = GaussianMixture(
    n_components=OPTIMAL_K,
    covariance_type='full',
    random_state=42,
    n_init=50,  # Many initializations for best fit
    max_iter=300
)

# Fit the model
gmm_final.fit(X_scaled)

# Get cluster assignments
labels = gmm_final.predict(X_scaled)
proba = gmm_final.predict_proba(X_scaled)

print(f"✓ Model converged: {gmm_final.converged_}")
print(f"✓ Final BIC: {gmm_final.bic(X_scaled):.1f}")
print(f"✓ Final AIC: {gmm_final.aic(X_scaled):.1f}")

# ============================================
# ANALYZE CLUSTER SIZES
# ============================================
print("\n" + "="*60)
print("CLUSTER SIZES")
print("="*60)

for i in range(OPTIMAL_K):
    count = np.sum(labels == i)
    pct = 100 * count / len(labels)
    print(f"Cluster {i}: {count:4d} characters ({pct:5.1f}%)")

# ============================================
# ADD CLUSTER LABELS TO DATAFRAME
# ============================================
df_clean['cluster'] = labels
df_clean['max_cluster_probability'] = proba.max(axis=1)

print("\n✓ Cluster labels added to dataframe")

# ============================================
# SAVE RESULTS
# ============================================
df_clean.to_csv('characters_with_clusters.csv', index=False)
print("✓ Saved to 'characters_with_clusters.csv'")

print("\n" + "="*60)
print("NEXT STEP: Find exemplars for each cluster")
print("="*60)


# ============================================
# FIND EXEMPLAR FOR EACH CLUSTER
# ============================================
print("Finding exemplars (most representative character per cluster)...\n")

def find_exemplars(X, labels, gmm, df_clean):
    """
    Find the character closest to each cluster's center
    Using Mahalanobis distance (accounts for covariance)
    """
    exemplars = {}
    
    for cluster_id in range(gmm.n_components):
        # Get all characters in this cluster
        cluster_mask = (labels == cluster_id)
        X_cluster = X[cluster_mask]
        
        # Get the ACTUAL indices from df_clean (not just 0,1,2,...)
        cluster_indices = df_clean[cluster_mask].index.values
        
        # Cluster mean (center)
        mean = gmm.means_[cluster_id]
        
        # Cluster covariance
        cov = gmm.covariances_[cluster_id]
        
        # Calculate Mahalanobis distance for each character to center
        try:
            inv_cov = np.linalg.inv(cov)
            distances = []
            for point in X_cluster:
                diff = point - mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                distances.append(mahal_dist)
            
            # Find closest character (minimum distance = exemplar)
            exemplar_idx_local = np.argmin(distances)
            exemplar_idx_global = cluster_indices[exemplar_idx_local]
            
            exemplars[cluster_id] = {
                'index': exemplar_idx_global,  # This is now the correct df index
                'distance': distances[exemplar_idx_local],
                'cluster_size': len(cluster_indices)
            }
            
        except np.linalg.LinAlgError:
            # Fallback: use Euclidean distance if covariance matrix is singular
            distances = np.linalg.norm(X_cluster - mean, axis=1)
            exemplar_idx_local = np.argmin(distances)
            exemplar_idx_global = cluster_indices[exemplar_idx_local]
            
            exemplars[cluster_id] = {
                'index': exemplar_idx_global,
                'distance': distances[exemplar_idx_local],
                'cluster_size': len(cluster_indices)
            }
    
    return exemplars

# Find exemplars (pass df_clean now)
exemplars = find_exemplars(X_scaled, labels, gmm_final, df_clean)

# ============================================
# DISPLAY EXEMPLARS
# ============================================
print("="*60)
print("EXEMPLARS (Most Representative Character per Cluster)")
print("="*60)

for cluster_id, info in exemplars.items():
    idx = info['index']
    
    # Use .loc instead of .iloc to access by actual index
    char_code = df_clean.loc[idx, 'Unnamed: 0']
    cluster_size = info['cluster_size']
    
    print(f"\nCluster {cluster_id}: {cluster_size} characters")
    print(f"  Exemplar: {char_code}")
    print(f"  Distance to center: {info['distance']:.3f}")


# ============================================
# SAVE EXEMPLAR INFO
# ============================================
exemplar_data = []
for cluster_id, info in exemplars.items():
    idx = info['index']
    exemplar_data.append({
        'cluster': cluster_id,
        'exemplar_code': df_clean.loc[idx, 'Unnamed: 0'],  # Use .loc
        'exemplar_index': idx,
        'cluster_size': info['cluster_size'],
        'distance_to_center': info['distance']
    })

exemplar_df = pd.DataFrame(exemplar_data)
exemplar_df.to_csv('cluster_exemplars.csv', index=False)
print("\n✓ Exemplars saved to 'cluster_exemplars.csv'")


# ============================================
# EXTRACT EXEMPLAR TRAIT PROFILES
# ============================================
print("\nExtracting trait profiles for exemplars...")

exemplar_profiles = []
for cluster_id, info in exemplars.items():
    idx = info['index']
    
    # Get the trait values using .loc and the trait_cols list
    traits = df_clean.loc[idx, trait_cols].values
    
    exemplar_profiles.append({
        'cluster': cluster_id,
        'exemplar_code': df_clean.loc[idx, 'Unnamed: 0'],  # Use .loc
        'trait_scores': traits
    })

# Save trait profiles
np.save('exemplar_trait_profiles.npy', exemplar_profiles, allow_pickle=True)
print("✓ Trait profiles saved to 'exemplar_trait_profiles.npy'")

print("\n" + "="*60)
print("STEP 1 COMPLETE!")
print("="*60)
print(f"✓ Found {OPTIMAL_K} personality clusters")
print(f"✓ Identified exemplar for each cluster")
print(f"✓ Saved cluster assignments and exemplars")
print("\nNext: Run identify_exemplars.py to see character names")
print("="*60)



