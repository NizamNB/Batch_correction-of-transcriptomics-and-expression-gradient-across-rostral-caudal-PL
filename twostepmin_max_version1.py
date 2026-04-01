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
from sklearn.metrics import balanced_accuracy_score
import umap
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.data import WeightedRandomSampler

# ============================================================
#  Hyperparameters — edit here, auto-logged to every WandB run
# ============================================================
CONFIG = {
    # Architecture
    "latent_dim":      64,
    "hidden_dim":      128,
    "disc_hidden_dim": 128,

    # Training
    "batch_size":      512,
    "lr_pretrain":     5e-4,
    "lr_ae":           1e-4,
    "lr_disc":         5e-5,
    "lr_bio":          1e-3,
    "epochs_phase1":   100,
    "epochs_phase2":   300,
    "epochs_phase3":   200,

    # Adversarial
    "lam_max":         0.8, 
    "lam_ramp_epochs": 200,   # max adversarial weight (ramps up linearly)

    # Data
    "apply_combat":    False,
    "min_pc_cells":    0.01,   # min fraction of cells expressing a gene
}

# ============================================================
#  Data Preparation & Leakage-Proof ComBat
# ============================================================

def load_and_prepare_data(base_path, datasets):
    cell_expression = {name: pd.read_csv(os.path.join(base_path, f"{name}_excitatory.csv")) for name in datasets}
    Female  = cell_expression["Zhuang-ABCA-1"].copy()
    Male    = cell_expression["Zhuang-ABCA-2"].copy()
    MERFISH = cell_expression["MERFISH-C57BL6J-638850"].copy()

    for df, s, b in zip([Female, Male, MERFISH], ["F", "M", "M"], ["female", "male", "merfish"]):
        df["sex"], df["batch"] = s, b

    combined = pd.concat([Female, Male, MERFISH], axis=0, ignore_index=True)

    mid_x = 3.23
    bins  = [combined["x_ccf"].min(), mid_x, combined["x_ccf"].max()]
    combined = combined.assign(
        axes       = pd.cut(combined["x_ccf"], bins=bins, labels=["ros", "caud"]).fillna("ros").astype(str),
        sex_region = lambda x: x["sex"] + "_" + x["axes"],
        layer      = lambda x: x["parcellation_substructure"].replace({
            'PL6a': 'PL6', 'PL6b': 'PL6', 'PL2': 'PL23', 'PL3': 'PL23'
        })
    )
    return combined


def apply_pycombat(X_cells_genes, batch_labels, gene_names=None):
    genes    = gene_names if gene_names is not None else [f"g{i}" for i in range(X_cells_genes.shape[1])]
    cell_ids = [f"cell{i}" for i in range(X_cells_genes.shape[0])]
    combat_input = pd.DataFrame(X_cells_genes.T, index=genes, columns=cell_ids)
    return pycombat(combat_input, batch_labels).T.values.astype(np.float32)


def extract_training_matrices(df, common_genes_csv, apply_combat=False, min_pc_cells=0.01):
    gene_list_df = pd.read_csv(common_genes_csv)
    common_genes = list(gene_list_df["gene_symbol"])
    gene_cols    = [g for g in common_genes if g in df.columns]

    X_raw        = df[gene_cols].values.astype(np.float32)
    batch_labels = df["batch"].astype(str).values

    if apply_combat:
        print(f"Applying ComBat on {len(df)} Non-PL cells...")
        X_raw = apply_pycombat(X_raw, batch_labels, gene_names=gene_cols)

    keep_mask      = (X_raw > 0).mean(axis=0) >= float(min_pc_cells)
    selected_genes = [gene_cols[i] for i in np.where(keep_mask)[0]]
    print(f"Selected {len(selected_genes)} genes after filtering for >{min_pc_cells*100:.1f}% expression")
    X_sel = X_raw[:, keep_mask]

    mu, sigma = X_sel.mean(axis=0, keepdims=True), X_sel.std(axis=0, keepdims=True) + 1e-6
    X_norm    = (X_sel - mu) / sigma

    batch_cat = df["batch"].astype("category")
    return X_norm, batch_cat.cat.codes.values.astype(np.int64), batch_cat.cat.categories.tolist(), selected_genes, mu, sigma


def process_pl_with_combat(df_pl, selected_genes, mu, sigma):
    X_pl_raw = df_pl[selected_genes].values.astype(np.float32)
    return (X_pl_raw - mu) / sigma


# ============================================================
#  Model Architectures
# ============================================================

class GeneDataset(Dataset):
    def __init__(self, X, batch_ids, sex_labels=None):
        self.X     = torch.tensor(X, dtype=torch.float32)
        self.batch = torch.tensor(batch_ids, dtype=torch.long)
        self.sex   = torch.tensor(sex_labels, dtype=torch.long) if sex_labels is not None else None

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        if self.sex is not None: return self.X[idx], self.batch[idx], self.sex[idx]
        return self.X[idx], self.batch[idx]


class ConditionalAE(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder   = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder   = nn.Sequential(nn.Linear(latent_dim + n_batches, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))
        self.n_batches = n_batches

    def forward(self, x, batch_ids):
        z    = self.encoder(x)
        b_oh = F.one_hot(batch_ids, num_classes=self.n_batches).float()
        return self.decoder(torch.cat([z, b_oh], dim=1)), z


class AEWithDiscriminator(nn.Module):
    def __init__(self, ae, latent_dim=64, n_batches=3, disc_hidden=128):
        super().__init__()
        self.ae   = ae
        self.disc = nn.Sequential(nn.Linear(latent_dim, disc_hidden), nn.ReLU(), nn.Linear(disc_hidden, n_batches))

    def forward(self, x, batch_ids):
        x_hat, z = self.ae(x, batch_ids)
        return x_hat, z, self.disc(z)


class SexRegionClassifier(nn.Module):
    def __init__(self, latent_dim=64, n_classes=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, z): return self.net(z)


# ============================================================
#  Helpers
# ============================================================

def split_stratified(y):
    rng, tr, va = np.random.default_rng(42), [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        split_point = int(0.7 * len(idx))
        tr.append(idx[:split_point])
        va.append(idx[split_point:])
    return np.concatenate(tr), np.concatenate(va)


def calculate_batch_distances(z_mat, batch_labels, b_names):
    centroids  = np.array([z_mat[batch_labels == i].mean(axis=0) for i in range(len(b_names))])
    mse_matrix = np.zeros((len(b_names), len(b_names)))
    for i in range(len(b_names)):
        for j in range(len(b_names)):
            mse_matrix[i, j] = np.mean((centroids[i] - centroids[j])**2)
    return mse_matrix


def make_tsne_fig(data_mat, labels, label_name, title):
    """Run t-SNE and return a matplotlib figure (caller decides when to close)."""
    tsne   = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(data_mat)
    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1], label=str(lbl), s=10, alpha=0.7)
    ax.set_title(f"t-SNE: {title} — colored by {label_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def make_umap_fig(data_mat, labels, label_name, title):
    """Run UMAP and return a matplotlib figure."""
    reducer = umap.UMAP(random_state=42)
    z_2d    = reducer.fit_transform(data_mat)
    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], label=str(lbl), s=6, alpha=0.6)
    ax.set_title(f"UMAP: {title} — colored by {label_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def plot_roc_fig(y_true, y_probs, class_names, title):
    """Return a matplotlib ROC figure. Skips classes absent from y_true."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            print(f"  [plot_roc] Skipping '{name}' — only one class present in y_true.")
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


# ============================================================
#  Main Execution
# ============================================================
if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\StujenskeLab\\Documents\\NAN151_workspace\\microarray"
    SAVE_DIR  = r"C:\Users\StujenskeLab\Documents\NAN151_workspace\microarray\saved_weights"
    datasets  = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "MERFISH-C57BL6J-638850"]
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_ID    = "Batchcorrection_003_withoutCombat"
    prev_ID   = "Batchcorrection_001_withCombat"  # source of Phase 1 weights when SKIP_PHASE1=True
 
    # ── Checkpoint control ───────────────────────────────────────────────────
    # Set SKIP_PHASE* = True to load saved weights and skip that phase's training.
    # The vis/artifact blocks always run regardless of these flags.
    #
    # Typical workflows:
    #   First run       → all False (train everything, save all checkpoints)
    #   Retrain P2+P3   → SKIP_PHASE1=True, others False
    #   Retrain P3 only → SKIP_PHASE1=True, SKIP_PHASE2=True, SKIP_PHASE3=False
    #   Plots only      → all True (loads all weights, jumps straight to vis)
    # ────────────────────────────────────────────────────────────────────────
    SKIP_PHASE1 = True
    SKIP_PHASE2 = False
    SKIP_PHASE3 = False
 
    phase1_ae_path   = os.path.join(SAVE_DIR, f"{prev_ID}_ae_phase1.pt")
    phase2_ae_path   = os.path.join(SAVE_DIR, f"{RUN_ID}_ae_phase2.pt")
    phase2_disc_path = os.path.join(SAVE_DIR, f"{RUN_ID}_disc_phase2.pt")
    phase3_bio_path  = os.path.join(SAVE_DIR, f"{RUN_ID}_bio_classifier_phase3.pt")
 
    os.makedirs(SAVE_DIR, exist_ok=True)
 
    # ── Data ─────────────────────────────────────────────────────────────────
    df_all = load_and_prepare_data(BASE_PATH, datasets)
    df_tr  = df_all[df_all["parcellation_structure"] != "PL"].copy()
    df_pl  = df_all[df_all["parcellation_structure"] == "PL"].copy()
 
    X, b_ids, b_names, genes, mu, sigma = extract_training_matrices(
        df_tr,
        os.path.join(BASE_PATH, "common_genes.csv"),
        apply_combat=CONFIG["apply_combat"],
        min_pc_cells=CONFIG["min_pc_cells"],
    )
    X_pl = process_pl_with_combat(df_pl, genes, mu, sigma)
 
    b_ids_pl       = df_pl["batch"].astype(pd.CategoricalDtype(categories=b_names)).cat.codes.values.astype(np.int64)
    sex_cat        = df_pl["sex_region"].astype("category")
    sex_labels_pl  = sex_cat.cat.codes.values
    sex_names      = sex_cat.cat.categories.tolist()
    tr_i, va_i     = split_stratified(sex_labels_pl)
 
    nonpl_layer_labels = df_tr["layer"].values
    nonpl_batch_labels = df_tr["batch"].astype(pd.CategoricalDtype(categories=b_names)).cat.codes.values.astype(np.int64)
 
    pl_test_sex   = df_pl.iloc[va_i]["sex"].values
    pl_test_batch = df_pl.iloc[va_i]["batch"].values
    pl_test_layer = df_pl.iloc[va_i]["layer"].values
 
    all_batch_labels = list(range(len(b_names)))
    all_sex_labels   = list(range(len(sex_names)))
 
    # ── Samplers & DataLoaders ───────────────────────────────────────────────
    batch_counts  = np.bincount(b_ids)
    weights_batch = 1. / torch.tensor(batch_counts, dtype=torch.float)
    sampler_mm    = WeightedRandomSampler(weights_batch[b_ids], len(b_ids))
    pretrain_ldr  = DataLoader(GeneDataset(X, b_ids), batch_size=CONFIG["batch_size"], sampler=sampler_mm)
 
    bio_counts  = np.bincount(sex_labels_pl[tr_i])
    weights_bio = 1. / torch.tensor(bio_counts, dtype=torch.float)
    sampler_bio = WeightedRandomSampler(weights_bio[sex_labels_pl[tr_i]], len(tr_i))
    pl_tr_ldr   = DataLoader(GeneDataset(X_pl[tr_i], b_ids_pl[tr_i], sex_labels_pl[tr_i]), batch_size=CONFIG["batch_size"], sampler=sampler_bio)
    pl_va_ldr   = DataLoader(GeneDataset(X_pl[va_i], b_ids_pl[va_i], sex_labels_pl[va_i]), batch_size=CONFIG["batch_size"])
 
    disc_class_weights = torch.tensor(
        (1.0 / np.bincount(b_ids)) / (1.0 / np.bincount(b_ids)).sum() * len(b_names),
        dtype=torch.float32
    ).to(device)
 
    # ============================================================
    # PHASE 1 — PRETRAIN AE
    # ============================================================
    ae = ConditionalAE(
        X.shape[1], len(b_names),
        latent_dim=CONFIG["latent_dim"],
        hidden_dim=CONFIG["hidden_dim"],
    ).to(device)
 
    if SKIP_PHASE1 and os.path.exists(phase1_ae_path):
        print(f"[Phase 1] Loading AE weights from:\n  {phase1_ae_path}")
        ae.load_state_dict(torch.load(phase1_ae_path, map_location=device))
        print("[Phase 1] Weights loaded — skipping training.\n")
    else:
        print("[Phase 1] Training AE from scratch...")
        wandb.init(project="Runs_on_DoubleGPU", name="Phase_1_Pretrain",
                   id=f"{RUN_ID}_pre", config=CONFIG)
        opt_pre = torch.optim.Adam(ae.parameters(), lr=CONFIG["lr_pretrain"])
 
        for epoch in range(CONFIG["epochs_phase1"]):
            ae.train()
            losses = []
            for x, b in pretrain_ldr:
                x, b     = x.to(device), b.to(device)
                x_hat, _ = ae(x, b)
                loss     = F.mse_loss(x_hat, x)
                opt_pre.zero_grad(); loss.backward(); opt_pre.step()
                losses.append(loss.item())
            wandb.log({"pretrain/recon_mse": np.mean(losses)}, step=epoch+1)
 
        wandb.finish()
        torch.save(ae.state_dict(), phase1_ae_path)
        print(f"[Phase 1] Weights saved to:\n  {phase1_ae_path}\n")
 
    # ============================================================
    # PHASE 2 — MIN-MAX ADVERSARIAL
    # ============================================================
    full_model = AEWithDiscriminator(
        ae,
        latent_dim=CONFIG["latent_dim"],
        n_batches=len(b_names),
        disc_hidden=CONFIG["disc_hidden_dim"],
    ).to(device)
 
    if SKIP_PHASE2 and os.path.exists(phase2_ae_path) and os.path.exists(phase2_disc_path):
        print(f"[Phase 2] Loading AE weights from:\n  {phase2_ae_path}")
        full_model.ae.load_state_dict(torch.load(phase2_ae_path, map_location=device))
        print(f"[Phase 2] Loading disc weights from:\n  {phase2_disc_path}")
        full_model.disc.load_state_dict(torch.load(phase2_disc_path, map_location=device))
        print("[Phase 2] Weights loaded — skipping training.\n")
 
        # Still need va_b_all / va_probs_all for the final ROC plot below,
        # so run one inference pass over the val set
        full_model.eval()
        va_disc_probs, va_b, va_preds = [], [], []
        with torch.no_grad():
            for x_v, b_v, *_ in pl_va_ldr:
                x_v, b_v = x_v.to(device), b_v.to(device)
                _, z_v   = full_model.ae(x_v, b_v)
                logits_v = full_model.disc(z_v)
                va_disc_probs.append(F.softmax(logits_v, dim=1).cpu().numpy())
                va_preds.append(logits_v.argmax(1).cpu().numpy())
                va_b.append(b_v.cpu().numpy())
        va_b_all     = np.concatenate(va_b)
        va_probs_all = np.concatenate(va_disc_probs)
 
    else:
        print("[Phase 2] Training adversarial model...")
        wandb.init(project="Runs_on_DoubleGPU", name="Phase_2_MinMax",
                   id=f"{RUN_ID}_mm", config=CONFIG)
 
        opt_ae   = torch.optim.Adam(full_model.ae.parameters(),   lr=CONFIG["lr_ae"])
        opt_disc = torch.optim.Adam(full_model.disc.parameters(), lr=CONFIG["lr_disc"])
        lam_max  = CONFIG["lam_max"]
        RAMP_EPOCHS = CONFIG["lam_ramp_epochs"]
 
        for epoch in range(CONFIG["epochs_phase2"]):
            full_model.train()
            # Ramp lambda linearly to lam_max over RAMP_EPOCHS, then hold fixed
            lam = min((epoch / RAMP_EPOCHS) * lam_max, lam_max)
 
            disc_losses, recon_losses = [], []
 
            for x, b in pretrain_ldr:
                x, b = x.to(device), b.to(device)
 
                # -- Update Discriminator --
                full_model.ae.requires_grad_(False)
                full_model.disc.requires_grad_(True)
                _, z   = full_model.ae(x, b)
                l_disc = F.cross_entropy(full_model.disc(z.detach()), b, weight=disc_class_weights)
                opt_disc.zero_grad(); l_disc.backward(); opt_disc.step()
                disc_losses.append(l_disc.item())
 
                # -- Update AE --
                full_model.ae.requires_grad_(True)
                full_model.disc.requires_grad_(False)
                x_hat, z   = full_model.ae(x, b)
                l_recon    = F.mse_loss(x_hat, x)
                l_adv      = F.cross_entropy(full_model.disc(z), b, weight=disc_class_weights)
                total_loss = l_recon - (lam * l_adv)
                opt_ae.zero_grad(); total_loss.backward()
                torch.nn.utils.clip_grad_norm_(full_model.ae.parameters(), max_norm=1.0)
                opt_ae.step()
                recon_losses.append(l_recon.item())
 
            # ── Epoch-level metrics ──────────────────────────────────────────
            full_model.eval()
            with torch.no_grad():
                tr_disc_probs, tr_b = [], []
                for x_e, b_e in pretrain_ldr:
                    x_e, b_e = x_e.to(device), b_e.to(device)
                    _, z_e   = full_model.ae(x_e, b_e)
                    tr_disc_probs.append(F.softmax(full_model.disc(z_e), dim=1).cpu().numpy())
                    tr_b.append(b_e.cpu().numpy())
                tr_b_all        = np.concatenate(tr_b)
                tr_probs_all    = np.concatenate(tr_disc_probs)
                epoch_batch_auc = roc_auc_score(
                    tr_b_all, tr_probs_all, multi_class='ovr', labels=all_batch_labels
                )
 
                va_disc_probs, va_b, va_preds = [], [], []
                for x_v, b_v, *_ in pl_va_ldr:
                    x_v, b_v = x_v.to(device), b_v.to(device)
                    _, z_v   = full_model.ae(x_v, b_v)
                    logits_v = full_model.disc(z_v)
                    va_disc_probs.append(F.softmax(logits_v, dim=1).cpu().numpy())
                    va_preds.append(logits_v.argmax(1).cpu().numpy())
                    va_b.append(b_v.cpu().numpy())
 
            va_b_all     = np.concatenate(va_b)
            va_probs_all = np.concatenate(va_disc_probs)
            va_preds_all = np.concatenate(va_preds)
            va_bal_acc   = balanced_accuracy_score(va_b_all, va_preds_all)
 
            wandb.log({
                "minmax/disc_loss":         np.mean(disc_losses),
                "minmax/recon_loss":        np.mean(recon_losses),
                "minmax/train_batch_auc":   epoch_batch_auc,
                "minmax/val_batch_bal_acc": va_bal_acc,
                "minmax/lambda":            lam,
                "minmax/val_disc_roc":      wandb.plot.roc_curve(va_b_all, va_probs_all, labels=b_names),
            }, step=epoch+1)
 
            if (epoch + 1) % 20 == 0:
                print(f"[P2] Epoch {epoch+1:3d} | recon={np.mean(recon_losses):.4f} "
                      f"| disc_loss={np.mean(disc_losses):.4f} "
                      f"| train_AUC={epoch_batch_auc:.4f} "
                      f"| val_bal_acc={va_bal_acc:.4f}")
 
            full_model.train()
 
        # ── Phase 2 final static ROC ─────────────────────────────────────────
        print(f"\n[Phase 2] Final discriminator ROC after {CONFIG['epochs_phase2']} epochs...")
        fig_p2_roc = plot_roc_fig(
            va_b_all, va_probs_all, b_names,
            title=(f"Final Discriminator ROC (After {CONFIG['epochs_phase2']} Epochs)\n"
                   "AUC ≈ 0.5 = good batch correction (3-class chance)")
        )
        wandb.log({"minmax/final_disc_roc_static": wandb.Image(fig_p2_roc)})
        plt.show(); plt.close(fig_p2_roc)
 
        wandb.finish()
 
        # Save Phase 2 checkpoints
        torch.save(full_model.ae.state_dict(),   phase2_ae_path)
        torch.save(full_model.disc.state_dict(), phase2_disc_path)
        print(f"[Phase 2] AE weights saved to:   {phase2_ae_path}")
        print(f"[Phase 2] Disc weights saved to: {phase2_disc_path}\n")
 
    # ============================================================
    # PHASE 3 — BIOLOGY (SEX/REGION)
    # ============================================================
    classifier = SexRegionClassifier(
        latent_dim=CONFIG["latent_dim"],
        n_classes=len(sex_names),
        hidden_dim=CONFIG["hidden_dim"],
    ).to(device)
 
    if SKIP_PHASE3 and os.path.exists(phase3_bio_path):
        print(f"[Phase 3] Loading classifier weights from:\n  {phase3_bio_path}")
        classifier.load_state_dict(torch.load(phase3_bio_path, map_location=device))
        print("[Phase 3] Weights loaded — skipping training.\n")
 
        # Run one inference pass to populate y_true / y_probs for final ROC
        classifier.eval()
        all_probs, all_y, all_preds = [], [], []
        with torch.no_grad():
            for x, b, y_s in pl_va_ldr:
                x, b  = x.to(device), b.to(device)
                _, z  = full_model.ae(x, b)
                logits = classifier(z)
                probs  = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_y.append(y_s.numpy())
        y_true  = np.concatenate(all_y)
        y_pred  = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
 
    else:
        print("[Phase 3] Training sex-region classifier...")
        sex_weights = torch.tensor(
            1.0 / (np.bincount(sex_labels_pl[tr_i], minlength=len(sex_names)) + 1e-6),
            dtype=torch.float32
        ).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=CONFIG["lr_bio"])
        criterion = nn.CrossEntropyLoss(weight=sex_weights / sex_weights.sum() * len(sex_names))
 
        wandb.init(project="Runs_on_DoubleGPU", name="Phase_3_Biology",
                   id=f"{RUN_ID}_bio", config=CONFIG)
 
        for epoch in range(CONFIG["epochs_phase3"]):
            classifier.train()
            for x, b, y_s in pl_tr_ldr:
                x, b, y_s = x.to(device), b.to(device), y_s.to(device)
                with torch.no_grad():
                    _, z = full_model.ae(x, b)
                optimizer.zero_grad()
                criterion(classifier(z), y_s).backward()
                optimizer.step()
 
            classifier.eval()
            all_probs, all_y, all_preds = [], [], []
            with torch.no_grad():
                for x, b, y_s in pl_va_ldr:
                    x, b    = x.to(device), b.to(device)
                    _, z    = full_model.ae(x, b)
                    logits  = classifier(z)
                    probs   = F.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_y.append(y_s.numpy())
 
            y_true  = np.concatenate(all_y)
            y_pred  = np.concatenate(all_preds)
            y_probs = np.concatenate(all_probs)
 
            bio_bal_acc = balanced_accuracy_score(y_true, y_pred)
            wandb.log({
                "bio/val_bal_acc": bio_bal_acc,
                "bio/roc":         wandb.plot.roc_curve(y_true, y_probs, labels=sex_names),
            }, step=epoch+1)
 
            if (epoch + 1) % 20 == 0:
                print(f"[P3] Epoch {epoch+1} | Bio Bal Acc: {bio_bal_acc:.4f}")
 
        # Phase 3 final static ROC
        print(f"\n[Phase 3] Final sex-region ROC after {CONFIG['epochs_phase3']} epochs...")
        fig_p3_roc = plot_roc_fig(
            y_true, y_probs, sex_names,
            title=f"Final Sex-Region Classifier ROC (After {CONFIG['epochs_phase3']} Epochs)"
        )
        wandb.log({"bio/final_bio_roc_static": wandb.Image(fig_p3_roc)})
        plt.show(); plt.close(fig_p3_roc)
 
        wandb.finish()
 
        # Save Phase 3 checkpoint
        torch.save(classifier.state_dict(), phase3_bio_path)
        print(f"[Phase 3] Classifier weights saved to: {phase3_bio_path}\n")
 
    # ============================================================
    # VISUALIZATIONS — all inside one WandB run
    # ============================================================
    wandb.init(project="Runs_on_DoubleGPU", name="Visualizations",
               id=f"{RUN_ID}_vis", config=CONFIG)
 
    # ── Collect PL val: raw gene space + corrected latent space ─────────────
    full_model.ae.eval()
    pl_raw_x, pl_z_corr, pl_batches_num = [], [], []
    with torch.no_grad():
        for x, b, _ in pl_va_ldr:
            _, z = full_model.ae(x.to(device), b.to(device))
            pl_raw_x.append(x.numpy())
            pl_z_corr.append(z.cpu().numpy())
            pl_batches_num.append(b.numpy())
 
    pl_raw_mat   = np.concatenate(pl_raw_x)
    pl_z_mat     = np.concatenate(pl_z_corr)
    pl_batch_num = np.concatenate(pl_batches_num)
 
    n_sub   = min(10000, len(pl_raw_mat))
    idx_sub = np.random.choice(len(pl_raw_mat), n_sub, replace=False)
    print(f"\n[Vis] t-SNE on {n_sub} PL val cells (subsampled)...")
 
    # ── PL val t-SNE: BEFORE batch correction ────────────────────────────────
    for label_arr, label_name in [
        (pl_test_batch[idx_sub], "Batch"),
        (pl_test_sex[idx_sub],   "Sex_Region"),
        (pl_test_layer[idx_sub], "Layer"),
    ]:
        fig = make_tsne_fig(pl_raw_mat[idx_sub], label_arr, label_name,
                            "PL_Val_Raw (Before Correction)")
        wandb.log({f"tsne_pl/before_{label_name}": wandb.Image(fig)})
        plt.show(); plt.close(fig)
 
    # ── PL val t-SNE: AFTER batch correction ─────────────────────────────────
    for label_arr, label_name in [
        (pl_test_batch[idx_sub], "Batch"),
        (pl_test_sex[idx_sub],   "Sex_Region"),
        (pl_test_layer[idx_sub], "Layer"),
    ]:
        fig = make_tsne_fig(pl_z_mat[idx_sub], label_arr, label_name,
                            "PL_Val_Latent (After Correction)")
        wandb.log({f"tsne_pl/after_{label_name}": wandb.Image(fig)})
        plt.show(); plt.close(fig)
 
    # ── Non-PL UMAP: BEFORE batch correction ─────────────────────────────────
    # n_sub_nonpl     = min(20000, len(X))
    # idx_sub_nonpl   = np.random.choice(len(X), n_sub_nonpl, replace=False)
    # nonpl_raw_sub   = X[idx_sub_nonpl]
    # nonpl_layer_sub = nonpl_layer_labels[idx_sub_nonpl]
    # print(f"[Vis] UMAP on {n_sub_nonpl} non-PL cells — BEFORE correction...")
    # fig = make_umap_fig(nonpl_raw_sub, nonpl_layer_sub, "Layer",
    #                     "Non-PL_Raw (Before Correction)")
    # wandb.log({"umap_nonpl/before_Layer": wandb.Image(fig)})
    # plt.show(); plt.close(fig)
 
    # ── Non-PL UMAP: AFTER batch correction ──────────────────────────────────
    # nonpl_b_sub_ids = nonpl_batch_labels[idx_sub_nonpl].astype(np.int64)
    # nonpl_ds        = GeneDataset(nonpl_raw_sub, nonpl_b_sub_ids)
    # nonpl_ldr       = DataLoader(nonpl_ds, batch_size=CONFIG["batch_size"])
    # nonpl_z_list    = []
    # full_model.ae.eval()
    # with torch.no_grad():
    #     for x_n, b_n in nonpl_ldr:
    #         _, z_n = full_model.ae(x_n.to(device), b_n.to(device))
    #         nonpl_z_list.append(z_n.cpu().numpy())
    # nonpl_z_mat = np.concatenate(nonpl_z_list)
    # print(f"[Vis] UMAP on {n_sub_nonpl} non-PL cells — AFTER correction...")
    # fig = make_umap_fig(nonpl_z_mat, nonpl_layer_sub, "Layer",
    #                     "Non-PL_Latent (After Correction)")
    # wandb.log({"umap_nonpl/after_Layer": wandb.Image(fig)})
    # plt.show(); plt.close(fig)
 
    # ── Batch bias heatmap ────────────────────────────────────────────────────
    mse_mat = calculate_batch_distances(pl_z_mat, pl_batch_num, b_names)
    mse_df  = pd.DataFrame(mse_mat, index=b_names, columns=b_names)
 
    print("\n" + "="*40)
    print("BATCH DISTANCE MATRIX (MSE between Centroids)")
    print("="*40)
    print(mse_df)
 
    fig_mse, ax_mse = plt.subplots(figsize=(8, 6))
    sns.heatmap(mse_df, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax_mse)
    ax_mse.set_title("Inter-Batch Latent Distance (MSE)\nLower = Better Batch Integration")
    fig_mse.tight_layout()
    wandb.log({
        "eval/batch_mse_matrix":     wandb.Table(dataframe=mse_df.reset_index()),
        "eval/batch_mse_heatmap":    wandb.Image(fig_mse),
        "eval/mean_inter_batch_mse": mse_mat[np.triu_indices(len(b_names), k=1)].mean(),
    })
    plt.show(); plt.close(fig_mse)
 
    wandb.finish()
 
    # ============================================================
    # MODEL SAVING & ARTIFACT LOGGING
    # ============================================================
    print("\nSaving final models and uploading artifacts to WandB...")
    wandb.init(project="Runs_on_DoubleGPU", name="Model_Artifacts",
               id=f"{RUN_ID}_artifacts", config=CONFIG)
 
    model_paths = {
        "ae_weights":     os.path.join(SAVE_DIR, f"{RUN_ID}_ae.pt"),
        "disc_weights":   os.path.join(SAVE_DIR, f"{RUN_ID}_disc.pt"),
        "bio_classifier": os.path.join(SAVE_DIR, f"{RUN_ID}_bio_classifier.pt"),
    }
 
    torch.save(full_model.ae.state_dict(),   model_paths["ae_weights"])
    torch.save(full_model.disc.state_dict(), model_paths["disc_weights"])
    torch.save(classifier.state_dict(),      model_paths["bio_classifier"])
 
    artifact = wandb.Artifact(name=f"model_bundle_{RUN_ID}", type="model")
    for filepath in model_paths.values():
        artifact.add_file(filepath)
    wandb.log_artifact(artifact)
 
    print("Model saving and WandB upload complete.")
    wandb.finish()