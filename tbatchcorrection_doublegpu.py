import os
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from combat.pycombat import pycombat
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score

# ============================================================
#  Data Preparation & Leakage-Proof ComBat
# ============================================================

def load_and_prepare_data(base_path, datasets):
    cell_expression = {name: pd.read_csv(os.path.join(base_path, f"{name}_excitatory.csv")) for name in datasets}
    Female = cell_expression["Zhuang-ABCA-1"].copy()
    Male = cell_expression["Zhuang-ABCA-2"].copy()
    MERFISH = cell_expression["MERFISH-C57BL6J-638850"].copy()
    
    for df, s, b in zip([Female, Male, MERFISH], ["F", "M", "M"], ["female", "male", "merfish"]):
        df["sex"], df["batch"] = s, b
    
    combined = pd.concat([Female, Male, MERFISH], axis=0)
    # file_map = {
    #     "F1": "merfish_F1_log.csv",
    #     "F2": "merfish_F2_log.csv",
    #     "M1": "merfish_M1_log.csv",
    #     "M2": "MERFISH-C57BL6J-638850_excitatory.csv"
    # }
    
    # processed_dfs = []
    
    # for batch_name, filename in file_map.items():
    #     print(f"Processing {batch_name} in chunks...")
    #     path = os.path.join(base_path, filename)
        
    #     # We read in chunks of 100k rows to avoid 'out of memory' during tokenization
    #     chunk_list = []
    #     for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
    #         # 1. Downcast floats immediately
    #         float_cols = chunk.select_dtypes(include=['float64']).columns
    #         chunk[float_cols] = chunk[float_cols].astype('float32')
            
    #         # 2. Standardize columns
    #         chunk = chunk.rename(columns={
    #             'CCF_level2': 'parcellation_structure', 
    #             'acronym': 'parcellation_substructure'
    #         })
            
    #         # 3. Add metadata
    #         chunk["sex"] = "F" if "F" in batch_name else "M"
    #         chunk["batch"] = batch_name
            
    #         chunk_list.append(chunk)
        
    #     df_batch = pd.concat(chunk_list, axis=0, ignore_index=True)
    #     processed_dfs.append(df_batch)
    #     print(f"Finished {batch_name}. Sub-total rows: {len(df_batch)}")

    # combined = pd.concat(processed_dfs, axis=0, ignore_index=True)
    # del processed_dfs 
   
    mid_x = 3.23
    bins = [combined["x_ccf"].min(), mid_x, combined["x_ccf"].max()]
    combined = combined.assign(
        axes = pd.cut(combined["x_ccf"], bins=bins, labels=["ros", "caud"]).fillna("ros").astype(str),
        sex_region = lambda x: x["sex"] + "_" + x["axes"],
        layer = lambda x: x["parcellation_substructure"].replace({
            'PL6a': 'PL6', 'PL6b': 'PL6', 'PL2': 'PL23', 'PL3': 'PL23'
        })
    )

    return combined.copy()

def apply_pycombat(X_cells_genes, batch_labels, gene_names=None):
    genes = gene_names if gene_names is not None else [f"g{i}" for i in range(X_cells_genes.shape[1])]
    cell_ids = [f"cell{i}" for i in range(X_cells_genes.shape[0])]
    combat_input = pd.DataFrame(X_cells_genes.T, index=genes, columns=cell_ids)
    return pycombat(combat_input, batch_labels).T.values.astype(np.float32)

def extract_training_matrices(df, common_genes_csv, apply_combat=False, min_pc_cells=0.01):
    gene_list_df = pd.read_csv(common_genes_csv)
    common_genes = list(gene_list_df["gene_symbol"])
    gene_cols = [g for g in common_genes if g in df.columns]
    
    X_raw, batch_labels = df[gene_cols].values.astype(np.float32), df["batch"].astype(str).values
    
    if apply_combat:
        print(f"Applying ComBat on {len(df)} Non-PL cells...")
        X_raw = apply_pycombat(X_raw, batch_labels, gene_names=gene_cols)
        
    keep_mask = (X_raw > 0).mean(axis=0) >= float(min_pc_cells)
    selected_genes = [gene_cols[i] for i in np.where(keep_mask)[0]]
    print(f"Selected {len(selected_genes)} genes after filtering for >{min_pc_cells*100:.1f}% expression")
    X_sel = X_raw[:, keep_mask]
    
    mu, sigma = X_sel.mean(axis=0, keepdims=True), X_sel.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_sel - mu) / sigma
    
    batch_cat = df["batch"].astype("category")
    return X_norm, batch_cat.cat.codes.values.astype(np.int64), batch_cat.cat.categories.tolist(), selected_genes, mu, sigma

def process_pl_with_combat(df_pl, selected_genes, mu, sigma):
    X_pl_raw = df_pl[selected_genes].values.astype(np.float32)
    batch_labels_pl = df_pl["batch"].astype(str).values
    #X_pl_combat = apply_pycombat(X_pl_raw, batch_labels_pl, gene_names=selected_genes)
    return (X_pl_raw - mu) / sigma

# ============================================================
#  Model Architectures
# ============================================================

class GeneDataset(Dataset):
    def __init__(self, X, batch_ids, sex_labels=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.batch = torch.tensor(batch_ids, dtype=torch.long)
        self.sex = torch.tensor(sex_labels, dtype=torch.long) if sex_labels is not None else None
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        if self.sex is not None: return self.X[idx], self.batch[idx], self.sex[idx]
        return self.X[idx], self.batch[idx]

class ConditionalAE(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim + n_batches, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))
        self.n_batches = n_batches
    def forward(self, x, batch_ids):
        z = self.encoder(x)
        b_oh = F.one_hot(batch_ids, num_classes=self.n_batches).float()
        return self.decoder(torch.cat([z, b_oh], dim=1)), z

class AEWithDiscriminator(nn.Module):
    def __init__(self, ae, latent_dim=64, n_batches=3, disc_hidden=128):
        super().__init__()
        self.ae = ae
        self.disc = nn.Sequential(nn.Linear(latent_dim, disc_hidden), nn.ReLU(), nn.Linear(disc_hidden, n_batches))
    def forward(self, x, batch_ids):
        x_hat, z = self.ae(x, batch_ids)
        return x_hat, z, self.disc(z)

class SexRegionClassifier(nn.Module):
    def __init__(self, latent_dim=64, n_classes=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, n_classes))
    def forward(self, z): return self.net(z)

# ============================================================
# Helpers
# ============================================================

@torch.no_grad()
def evaluate_frozen_metrics(model, loader, device):
    model.eval()
    r_ls, all_preds, all_labels = [], [], []
    for x, b, *rest in loader:
        x, b = x.to(device), b.to(device)
        x_hat, _, logits = model(x, b)
        r_ls.append(F.mse_loss(x_hat, x).item())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(b.cpu().numpy())
    p, l = np.concatenate(all_preds), np.concatenate(all_labels)
    return np.mean(r_ls), balanced_accuracy_score(l, p)

def split_stratified(y):
    rng, tr, va = np.random.default_rng(42), [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        split_point = int(0.7 * len(idx))
        tr.append(idx[:split_point])
        va.append(idx[split_point:])
    return np.concatenate(tr), np.concatenate(va)

#-----Batch Distance Calculation----#
def calculate_batch_distances(z_mat, batch_labels, b_names):
    centroids = []
    for i in range(len(b_names)):
        mask = batch_labels == i
        if np.any(mask):
            centroids.append(z_mat[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(z_mat.shape[1]))
    
    centroids = np.stack(centroids)
    n_batches = len(b_names)
    mse_matrix = np.zeros((n_batches, n_batches))
    
    for i in range(n_batches):
        for j in range(n_batches):
            # MSE between the average cell of Batch I and Batch J
            mse_matrix[i, j] = np.mean((centroids[i] - centroids[j])**2)
            
    return mse_matrix
#-----Layer Preservation Calculation----#
def plot_layer_preservation(z_mat, layers, batches, b_names, title_suffix):
    unique_layers = sorted(np.unique(layers))
    
    sil = silhouette_score(z_mat, layers)
    print(f"Layer Silhouette Score ({title_suffix}): {sil:.4f}")
    
    reducer = umap.UMAP(random_state=42)
    z_2d = reducer.fit_transform(z_mat)
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    for layer in unique_layers:
        mask = layers == layer
        ax[0].scatter(z_2d[mask, 0], z_2d[mask, 1], label=layer, s=5, alpha=0.6)
    ax[0].set_title(f"Layers in Latent Space ({title_suffix})")
    ax[0].legend(markerscale=3, title="Anatomical Layer")

    for i, b_name in enumerate(b_names):
        mask = batches == i
        ax[1].scatter(z_2d[mask, 0], z_2d[mask, 1], label=b_name, s=5, alpha=0.3)
    ax[1].set_title(f"Batch Mixing in Latent Space ({title_suffix})")
    ax[1].legend(markerscale=3, title="Dataset Source")
    
    plt.tight_layout()
    plt.show()
    
    return sil
def plot_tsne_comparison(data_mat, labels, label_name, title):
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(data_mat)
    
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], label=str(lbl), s=10, alpha=0.7)
    
    plt.title(f"t-SNE: {title} colored by {label_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Log to WandB
    wandb.log({f"tsne/{title}_{label_name}": wandb.Image(plt)})
    plt.show()
# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\StujenskeLab\\Documents\\NAN151_workspace\\microarray"
    datasets = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "MERFISH-C57BL6J-638850"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_ID = "OldRun_001_withoutCombat"

    df_all = load_and_prepare_data(BASE_PATH, datasets)
    df_tr = df_all[df_all["parcellation_structure"] != "PL"].copy()
    df_pl = df_all[df_all["parcellation_structure"] == "PL"].copy()

    X, b_ids, b_names, genes, mu, sigma = extract_training_matrices(df_tr, os.path.join(BASE_PATH, "common_genes.csv"))
    X_pl = process_pl_with_combat(df_pl, genes, mu, sigma)
    
    b_ids_pl = df_pl["batch"].astype(pd.CategoricalDtype(categories=b_names)).cat.codes.values.astype(np.int64)
    sex_cat = df_pl["sex_region"].astype("category")
    sex_labels_pl, sex_names = sex_cat.cat.codes.values, sex_cat.cat.categories.tolist()
    layer_labels_pl = df_pl["layer"].values
    tr_i, va_i = split_stratified(sex_labels_pl)
    
 
    pretrain_ldr = DataLoader(GeneDataset(X, b_ids), batch_size=512, shuffle=True)
    pl_tr_ldr = DataLoader(GeneDataset(X_pl[tr_i], b_ids_pl[tr_i], sex_labels_pl[tr_i]), batch_size=512, shuffle=True)
    pl_va_ldr = DataLoader(GeneDataset(X_pl[va_i], b_ids_pl[va_i], sex_labels_pl[va_i]), batch_size=512)
    
    weights = torch.tensor((1.0/np.bincount(b_ids)) / (1.0/np.bincount(b_ids)).sum() * len(b_names), dtype=torch.float32).to(device)

    # PHASE 1: PRETRAIN AE
    ae = ConditionalAE(X.shape[1], len(b_names)).to(device)
    wandb.init(project="Runs_on_DoubleGPU", name="Phase_1_Pretrain", id=f"{RUN_ID}_pre")
    opt_pre = torch.optim.Adam(ae.parameters(), lr=5e-4)
    
    for epoch in range(100):
        ae.train()
        losses = []
        for x, b in pretrain_ldr:
            x, b = x.to(device), b.to(device)
            x_hat, _ = ae(x, b)
            loss = F.mse_loss(x_hat, x)
            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
            losses.append(loss.item())
        wandb.log({"pretrain/recon_mse": np.mean(losses)}, step=epoch+1)
    wandb.finish()

    # PHASE 2: MIN-MAX ADVERSARIAL
    full_model = AEWithDiscriminator(ae, 64, len(b_names)).to(device)
    wandb.init(project="Runs_on_DoubleGPU", name="Phase_2_MinMax", id=f"{RUN_ID}_mm")
    opt_ae = torch.optim.Adam(full_model.ae.parameters(), lr=1e-4)
    opt_disc = torch.optim.Adam(full_model.disc.parameters(), lr=5e-5)
    
    lam_max = 0.4
    for epoch in range(200):
        full_model.train()
        lam = (epoch/200) * lam_max
        
        for x, b in pretrain_ldr:
            x, b = x.to(device), b.to(device)
            
            # Update Discriminator
            full_model.ae.requires_grad_(False)
            full_model.disc.requires_grad_(True)
            _, z = full_model.ae(x, b)
            l_disc = F.cross_entropy(full_model.disc(z.detach()), b, weight=weights)
            opt_disc.zero_grad(); l_disc.backward(); opt_disc.step()
            with torch.no_grad():
                disc_probs = F.softmax(full_model.disc(z.detach()), dim=1).cpu().numpy()
                # For multi-class AUC (ovr = one-vs-rest)
                batch_auc = roc_auc_score(b.cpu().numpy(), disc_probs, multi_class='ovr')
                
            wandb.log({
                "minmax/disc_loss": l_disc.item(),
                "minmax/batch_auc": batch_auc  # Aim for ~0.5 as training progresses
            }, step=epoch+1)
            # Update AE
            full_model.ae.requires_grad_(True)
            full_model.disc.requires_grad_(False)
            x_hat, z = full_model.ae(x, b)
            l_recon = F.mse_loss(x_hat, x)
            l_adv = F.cross_entropy(full_model.disc(z), b, weight=weights)
            
            total_loss = l_recon - (lam * l_adv)
            opt_ae.zero_grad(); total_loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.ae.parameters(), max_norm=1.0)
            opt_ae.step()
            # for _ in range(3):
            #     x_hat, z = full_model.ae(x, b)
            #     l_recon = F.mse_loss(x_hat, x)
                
            #     # Adversarial loss: we want to MAXIMIZE the discriminator's entropy 
            #     # (make it confused), so we minimize the negative cross entropy
            #     l_adv = F.cross_entropy(full_model.disc(z), b, weight=weights)
                
            #     # The total loss for the AE
            #     total_loss = l_recon - (lam * l_adv)
                
            #     opt_ae.zero_grad()
            #     total_loss.backward()
            #     opt_ae.step()

        if (epoch+1) % 20 == 0:
            _, va_bal = evaluate_frozen_metrics(full_model, pl_va_ldr, device)
            print(f"Epoch {epoch+1} | Batch Val Bal Acc: {va_bal:.4f}")
            wandb.log({"minmax/batch_bal_acc": va_bal}, step=epoch+1)
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                full_model.ae.eval()
                full_model.disc.eval()
                all_disc_probs, all_b = [], []
                for x_v, b_v in pl_va_ldr:
                    _, z_v = full_model.ae(x_v.to(device), b_v.to(device))
                    probs = F.softmax(full_model.disc(z_v), dim=1)
                    all_disc_probs.append(probs.cpu().numpy())
                    all_b.append(b_v.numpy())
                
                y_b_true = np.concatenate(all_b)
                y_b_probs = np.concatenate(all_disc_probs)
  
                wandb.log({
                    "minmax/discriminator_roc": wandb.plot.roc_curve(y_b_true, y_b_probs, labels=b_names)
                }, step=epoch+1)
    wandb.finish()

    # PHASE 3: BIOLOGY (SEX/REGION)
    classifier = SexRegionClassifier(64, len(sex_names)).to(device)
    sex_weights = torch.tensor((1.0/(np.bincount(sex_labels_pl[tr_i], minlength=len(sex_names))+1e-6)), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=sex_weights/sex_weights.sum()*len(sex_names))
    
    wandb.init(project="Runs_on_DoubleGPU", name="Phase_3_Biology", id=f"{RUN_ID}_bio")
    
    for epoch in range(200):
        classifier.train()
        for x, b, y_s in pl_tr_ldr: # REAL batch IDs 'b'
            x, b, y_s = x.to(device), b.to(device), y_s.to(device)
            with torch.no_grad():
                _, z = full_model.ae(x, b) 
            
            optimizer.zero_grad()
            criterion(classifier(z), y_s).backward()
            optimizer.step()
            
        # Evaluation
        classifier.eval()
        all_probs, all_y, all_preds = [], [], []
        with torch.no_grad():
            for x, b, y_s in pl_va_ldr:
                x, b = x.to(device), b.to(device)
                _, z = full_model.ae(x, b)
                logits = classifier(z)
                # Get probabilities using Softmax for ROC
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_y.append(y_s.numpy())
        
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        
        bio_bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Log to WandB
        wandb.log({
            "bio/val_bal_acc": bio_bal_acc,
            "bio/roc": wandb.plot.roc_curve(y_true, y_probs, labels=sex_names)
        }, step=epoch+1)
        
        if (epoch+1) % 50 == 0:
            print(f"Phase 3 Epoch {epoch+1} | Bio Bal Acc: {bio_bal_acc:.4f}")

    # ============================================================
    # FINAL VISUALIZATIONS & BATCH BIAS ANALYSIS
    # ============================================================
    
    full_model.ae.eval()
    all_raw_x, all_z, all_batches = [], [], []
    with torch.no_grad():
        for x, b, y_s in pl_va_ldr:
            x_dev, b_dev = x.to(device), b.to(device)
            _, z = full_model.ae(x_dev, b_dev)
            all_raw_x.append(x.numpy())
            all_z.append(z.cpu().numpy())
            all_batches.append(b.numpy())
    
    raw_x_mat = np.concatenate(all_raw_x)
    z_mat = np.concatenate(all_z)
    batch_labels = np.concatenate(all_batches)

    pca_raw = PCA(n_components=2).fit_transform(raw_x_mat)
    pca_z = PCA(n_components=2).fit_transform(z_mat)

    # # 2. PCA Plotting
    # fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    # for i, name in enumerate(b_names):
    #     mask = batch_labels == i
    #     axes[0].scatter(pca_raw[mask, 0], pca_raw[mask, 1], label=f'Batch: {name}', s=6, alpha=0.5, c=colors[i])
    # axes[0].set_title("PCA Before Adversarial Training (Raw Standardized)")
    # axes[0].legend()

    # # Plot After (Batch Corrected Latent Space)
    # for i, name in enumerate(b_names):
    #     mask = batch_labels == i
    #     axes[1].scatter(pca_z[mask, 0], pca_z[mask, 1], label=f'Batch: {name}', s=8, alpha=0.5, c=colors[i])
    # axes[1].set_title("PCA After Adversarial Training (Corrected Latent z)")
    # axes[1].legend()
    
    # plt.tight_layout()
    # plt.show()

    # Extract metadata for the PL test set (va_i)
    pl_test_sex = df_pl.iloc[va_i]["sex"].values
    pl_test_batch = df_pl.iloc[va_i]["batch"].values
    pl_test_layer = df_pl.iloc[va_i]["layer"].values

    # BEFORE CORRECTION (Raw Matrix)
    print("Generating t-SNE for Raw Data...")
    plot_tsne_comparison(raw_x_mat, pl_test_batch, "Batch", "Raw_PL_Test")
    plot_tsne_comparison(raw_x_mat, pl_test_sex, "Sex", "Raw_PL_Test")
    plot_tsne_comparison(raw_x_mat, pl_test_layer, "Anatomical_Layer", "Raw_PL_Test")
    # AFTER CORRECTION (Latent Space z)
    print("Generating t-SNE for Corrected Latent Space...")
    plot_tsne_comparison(z_mat, pl_test_batch, "Batch", "Corrected_PL_Test")
    plot_tsne_comparison(z_mat, pl_test_sex, "Sex", "Corrected_PL_Test")
    plot_tsne_comparison(z_mat, pl_test_layer, "Anatomical_Layer", "Corrected_PL_Test")

    # FINAL VISUALIZATION
    full_model.ae.eval(); all_z, all_l = [], []
    with torch.no_grad():
        for x, b, y_s in pl_va_ldr:
            _, z = full_model.ae(x.to(device), b.to(device))
            all_z.append(z.cpu().numpy()); all_l.append(y_s.numpy())
    
    z_2d = umap.UMAP().fit_transform(np.concatenate(all_z))
    l_flat = np.concatenate(all_l)
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(sex_names):
        m = l_flat == i
        plt.scatter(z_2d[m, 0], z_2d[m, 1], label=name, s=6, alpha=0.6)
    plt.title("Adversarially Cleaned Latent Space (PL Val Set)")
    plt.legend(); plt.show()

    # Multi-class ROC Plotting
    plt.figure(figsize=(8, 6))
    for i in range(len(sex_names)):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{sex_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Sex-Region Classification (PL Val)')
    plt.legend(loc='lower right')
    plt.show()

    # ============================================================
    # BATCH BIAS ANALYSIS (Centroid MSE)
    # ============================================================
    mse_mat = calculate_batch_distances(z_mat, batch_labels, b_names)
    
    print("\n" + "="*40)
    print("BATCH DISTANCE MATRIX (MSE between Centroids)")
    print("="*40)
    mse_df = pd.DataFrame(mse_mat, index=b_names, columns=b_names)
    print(mse_df)

    # Plotting the Bias Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(mse_df, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Inter-Batch Latent Distance (MSE)\nLower = Better Batch Integration")
    plt.show()

    # Log to WandB
    wandb.log({"eval/batch_mse_matrix": wandb.Table(dataframe=mse_df.reset_index())})
    # Calculate scores for the Latent Space (After Correction)
    batch_sil = silhouette_score(z_mat, batch_labels)
    layer_sil = silhouette_score(z_mat, pl_test_layer)
    print(f"Batch Silhouette (Lower is better): {batch_sil:.4f}")
    print(f"Layer Silhouette (Higher is better): {layer_sil:.4f}")

    # Log to WandB
    wandb.log({"eval/batch_silhouette": batch_sil, "eval/layer_silhouette": layer_sil})
    wandb.finish()