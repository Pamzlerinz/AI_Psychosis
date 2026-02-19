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

# ============================================
# LOAD THE IPIP-BFFM DATA
# ============================================
print("Loading IPIP-BFFM dataset...")

# Load the data (tab-separated)
df = pd.read_csv('data-final.csv', sep='\t')

print("Dataset shape:", df.shape)
print("\nFirst few columns:")
print(df.columns.tolist()[:20])

# ============================================
# DEFINE BIG FIVE ITEMS
# ============================================

# Each dimension has 10 items
# Responses are on a 1-5 scale (1=Disagree, 3=Neutral, 5=Agree)

big_five_items = {
    'Extraversion': [f'EXT{i}' for i in range(1, 11)],
    'Neuroticism': [f'EST{i}' for i in range(1, 11)],  # EST = Emotional STability (reversed scoring)
    'Agreeableness': [f'AGR{i}' for i in range(1, 11)],
    'Conscientiousness': [f'CSN{i}' for i in range(1, 11)],
    'Openness': [f'OPN{i}' for i in range(1, 11)]
}

# Verify all columns exist
print("\n" + "="*60)
print("VERIFYING BIG FIVE ITEMS")
print("="*60)

all_items = []
for dimension, items in big_five_items.items():
    all_items.extend(items)
    missing = [item for item in items if item not in df.columns]
    if missing:
        print(f"❌ {dimension}: Missing {missing}")
    else:
        print(f"✓ {dimension}: All 10 items found")

# ============================================
# CLEAN THE DATA
# ============================================

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Extract only Big Five items
X = df[all_items].copy()

print(f"Initial data: {X.shape[0]:,} responses × {X.shape[1]} items")

# Check for missing values
missing_count = X.isnull().sum().sum()
print(f"Missing values: {missing_count:,}")

# Remove rows with missing values
if missing_count > 0:
    X = X.dropna()
    df_clean = df.loc[X.index].copy()
    print(f"After removing missing: {X.shape[0]:,} responses")
else:
    df_clean = df.copy()

# Filter by IPC=1 (single submission per IP - cleaner data)
if 'IPC' in df_clean.columns:
    ipc_before = len(df_clean)
    df_clean = df_clean[df_clean['IPC'] == 1]
    X = X.loc[df_clean.index]
    print(f"After IPC=1 filter: {len(df_clean):,} responses (removed {ipc_before - len(df_clean):,})")

# ============================================
# STANDARDIZE DATA
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✓ Data standardized")
print(f"Final dataset: {X_scaled.shape[0]:,} people × {X_scaled.shape[1]} Big Five items")

# ============================================
# OPTION: CLUSTER BY DIMENSION OR ALL TOGETHER?
# ============================================

print("\n" + "="*60)
print("CLUSTERING APPROACH")
print("="*60)
print("Option A: Cluster all 50 items together")
print("Option B: Cluster each Big Five dimension separately (10 items each)")
print("\nUsing Option A for now (all 50 items)")
print("="*60)

# Use X_scaled with all 50 items
# If you want Option B, you'd extract just one dimension's items

# ============================================
# MODEL SELECTION: FIND OPTIMAL K
# ============================================

print("\nStarting model selection...")
print("Testing K from 2 to 15 clusters")
print("This may take several minutes with 1M+ samples...\n")

# For speed with large dataset, use a sample for K selection
# Then fit final model on full data
SAMPLE_SIZE = 50000  # Use 50k random samples for K selection

if len(X_scaled) > SAMPLE_SIZE:
    print(f"Using random sample of {SAMPLE_SIZE:,} for K selection (faster)")
    sample_indices = np.random.choice(len(X_scaled), SAMPLE_SIZE, replace=False)
    X_sample = X_scaled[sample_indices]
else:
    X_sample = X_scaled

n_components_range = range(2, 16)
aic_scores = []
bic_scores = []

for n in n_components_range:
    print(f"Testing K={n}...", end=" ")
    
    gmm = GaussianMixture(
        n_components=n,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    
    gmm.fit(X_sample)
    
    aic = gmm.aic(X_sample)
    bic = gmm.bic(X_sample)
    
    aic_scores.append(aic)
    bic_scores.append(bic)
    
    print(f"AIC={aic:.1f}, BIC={bic:.1f}, Converged={gmm.converged_}")

print("\n✓ Model selection complete!")

# ============================================
# VISUALIZE RESULTS
# ============================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(n_components_range, aic_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('AIC', fontsize=12)
plt.title('AIC Score (lower is better)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8, color='red')
plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('BIC', fontsize=12)
plt.title('BIC Score (lower is better)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_selection_results.png', dpi=150, bbox_inches='tight')
plt.show()

best_k_aic = n_components_range[np.argmin(aic_scores)]
best_k_bic = n_components_range[np.argmin(bic_scores)]

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print(f"Best K by AIC: {best_k_aic}")
print(f"Best K by BIC: {best_k_bic}")
print("="*60)

# ============================================
# FIT FINAL MODEL ON FULL DATA
# ============================================

OPTIMAL_K = best_k_bic  # Use BIC recommendation

print(f"\nFitting final GMM model with K={OPTIMAL_K} on FULL dataset...")
print("This may take a few minutes...\n")

gmm_final = GaussianMixture(
    n_components=OPTIMAL_K,
    covariance_type='full',
    random_state=42,
    n_init=3,
    max_iter=300
)

gmm_final.fit(X_scaled)

labels = gmm_final.predict(X_scaled)
proba = gmm_final.predict_proba(X_scaled)

print(f"✓ Model converged: {gmm_final.converged_}")
print(f"✓ Final BIC: {gmm_final.bic(X_scaled):.1f}")

# ============================================
# ANALYZE CLUSTERS
# ============================================
print("\n" + "="*60)
print("CLUSTER SIZES")
print("="*60)

for i in range(OPTIMAL_K):
    count = np.sum(labels == i)
    pct = 100 * count / len(labels)
    print(f"Cluster {i}: {count:,} people ({pct:.1f}%)")

# ============================================
# SAVE RESULTS
# ============================================

df_clean['cluster'] = labels
df_clean['max_cluster_probability'] = proba.max(axis=1)

df_clean.to_csv('ipip_with_clusters.csv', index=False)
print("\n✓ Saved to 'ipip_with_clusters.csv'")

# ============================================
# FIND EXEMPLARS
# ============================================

def find_exemplars(X, labels, gmm, df_clean):
    exemplars = {}
    
    for cluster_id in range(gmm.n_components):
        cluster_mask = (labels == cluster_id)
        X_cluster = X[cluster_mask]
        cluster_indices = df_clean[cluster_mask].index.values
        
        mean = gmm.means_[cluster_id]
        cov = gmm.covariances_[cluster_id]
        
        try:
            inv_cov = np.linalg.inv(cov)
            distances = []
            for point in X_cluster:
                diff = point - mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                distances.append(mahal_dist)
            
            exemplar_idx_local = np.argmin(distances)
            exemplar_idx_global = cluster_indices[exemplar_idx_local]
            
            exemplars[cluster_id] = {
                'index': exemplar_idx_global,
                'distance': distances[exemplar_idx_local],
                'cluster_size': len(cluster_indices)
            }
            
        except np.linalg.LinAlgError:
            distances = np.linalg.norm(X_cluster - mean, axis=1)
            exemplar_idx_local = np.argmin(distances)
            exemplar_idx_global = cluster_indices[exemplar_idx_local]
            
            exemplars[cluster_id] = {
                'index': exemplar_idx_global,
                'distance': distances[exemplar_idx_local],
                'cluster_size': len(cluster_indices)
            }
    
    return exemplars

print("\nFinding exemplars...")
exemplars = find_exemplars(X_scaled, labels, gmm_final, df_clean)

# ============================================
# DISPLAY EXEMPLAR PROFILES
# ============================================
print("\n" + "="*60)
print("EXEMPLAR PERSONALITY PROFILES")
print("="*60)

for cluster_id, info in exemplars.items():
    idx = info['index']
    
    print(f"\nCluster {cluster_id}: {info['cluster_size']:,} people")
    print(f"  Exemplar profile:")
    
    # Get exemplar's Big Five scores
    for dimension, items in big_five_items.items():
        scores = df_clean.loc[idx, items].values
        avg_score = scores.mean()
        print(f"    {dimension}: {avg_score:.2f}")

# Save exemplar data
exemplar_data = []
for cluster_id, info in exemplars.items():
    idx = info['index']
    
    profile = {'cluster': cluster_id, 'cluster_size': info['cluster_size']}
    
    # Add Big Five scores
    for dimension, items in big_five_items.items():
        scores = df_clean.loc[idx, items].values
        profile[dimension] = scores.mean()
    
    exemplar_data.append(profile)

exemplar_df = pd.DataFrame(exemplar_data)
exemplar_df.to_csv('cluster_exemplars.csv', index=False)
print("\n✓ Exemplars saved to 'cluster_exemplars.csv'")

print("\n" + "="*60)
print("CLUSTERING COMPLETE!")
print("="*60)
