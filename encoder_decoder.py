import wandb
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from combat.pycombat import pycombat


# ============================================================
# 1) DATA PREPARATION
# ============================================================

def load_and_prepare_data(base_path: str, datasets):
    cell_expression = {
        name: pd.read_csv(os.path.join(base_path, f"{name}_excitatory.csv"))
        for name in datasets
    }
    for name, df in cell_expression.items():
        print(f"Loaded {name}: {df.shape}")

    Female  = cell_expression["Zhuang-ABCA-1"].copy()
    Male    = cell_expression["Zhuang-ABCA-2"].copy()
    MERFISH = cell_expression["MERFISH-C57BL6J-638850"].copy()

    Female["sex"] = "F"
    Male["sex"] = "M"
    MERFISH["sex"] = "M"

    Female["batch"] = "female"
    Male["batch"] = "male"
    MERFISH["batch"] = "merfish"

    combined = pd.concat([Female, Male, MERFISH], axis=0)

    min_x = combined["x_ccf"].min()
    max_x = combined["x_ccf"].max()
    mid_x = 3.23

    bins = [min_x, mid_x, max_x]
    labels = ["ros", "caud"]

    for df in [Female, Male, MERFISH]:
        df["axes"] = pd.cut(df["x_ccf"], bins=bins, labels=labels).fillna("ros").astype(str)
        df["sex_region"] = df["sex"] + "_" + df["axes"]

    mixed = pd.concat([Female, Male, MERFISH], ignore_index=True)

    print("\nMerged dataset distribution (sex_region):")
    print(mixed["sex_region"].value_counts())

    return mixed

def apply_pycombat(X_cells_genes: np.ndarray, batch_labels: np.ndarray, gene_names=None):
    """
    X_cells_genes: numpy array (cells x genes)
    batch_labels:  array-like of length n_cells (strings or ints)
    Returns: X_combat (cells x genes)
    """
    genes = gene_names if gene_names is not None else [f"g{i}" for i in range(X_cells_genes.shape[1])]
    cell_ids = [f"cell{i}" for i in range(X_cells_genes.shape[0])]

    combat_input = pd.DataFrame(
        X_cells_genes.T,    
        index=genes,
        columns=cell_ids
    )
    assert len(batch_labels) == combat_input.shape[1], "batch_labels length must match number of cells"
    corrected = pycombat(combat_input, batch_labels)
    X_combat = corrected.T.values.astype(np.float32)
    return X_combat
def extract_training_matrices(
    df,
    common_genes_csv,
    top_k=60,
    apply_combat=True,
    expr_threshold=0.00,
    min_pc_cells=0.60
):
    gene_list_df = pd.read_csv(common_genes_csv)
    common_genes = list(gene_list_df["gene_symbol"])

    gene_cols = [g for g in common_genes if g in df.columns]
    X_all =df[gene_cols].values.astype(np.float32)
    batch_cat=df["batch"].astype("category")
    batch_ids = batch_cat.cat.codes.values.astype(np.int64)
    batch_names = batch_cat.cat.categories.tolist()
    n_batches = len(batch_names)
    batch_labels_str = batch_cat.astype(str).values
    print("Batch names:", batch_names)
    sr_cat = df["sex_region"].astype("category")
    sex_region_ids = sr_cat.cat.codes.values.astype(np.int64)
    class_names = list(sr_cat.cat.categories)
    n_classes = len(class_names)
    print("sex_region classes:", class_names)

    #---------Apply Combat---------
    if apply_combat:
        print("Applying ComBat on ALL common genes ...")
        X_all = apply_pycombat(X_all, batch_labels_str, gene_names=gene_cols)
        print("ComBat done. Shape:", X_all.shape)

    expressed = (X_all > expr_threshold).mean(axis=0)  
    keep_mask = expressed >= float(min_pc_cells)
    kept_genes_idx = np.where(keep_mask)[0]

    print(
        f"Genes expressed in >= {int(min_pc_cells*100)}% cells: "
        f"{len(kept_genes_idx)} / {len(gene_cols)}"
    )

    if len(kept_genes_idx) == 0:
        print("Debug: expressed fraction summary:")
        print(
            f"  min={expressed.min():.3f}, "
            f"p10={np.quantile(expressed,0.10):.3f}, "
            f"median={np.median(expressed):.3f}, "
            f"p90={np.quantile(expressed,0.90):.3f}, "
            f"max={expressed.max():.3f}"
        )
        raise ValueError(
            f"No genes passed min_pc_cells={min_pc_cells}. "
            f"Try lowering it or adjusting expr_threshold."
        )

    kept_genes = [gene_cols[i] for i in kept_genes_idx]
    X_kept = X_all[:, kept_genes_idx]

    # ---- gene selection ----
    if top_k is None:
        X = X_kept
        selected_genes = kept_genes
        top_k=X_kept.shape[1]
    else:
        top_k = int(top_k)
        if top_k > X_kept.shape[1]:
            top_k = X_kept.shape[1]
        
        mean_expression_kept = X_kept.mean(axis=0)
        top_local_idx = np.argsort(mean_expression_kept)[-top_k:]
        X = X_kept[:, top_local_idx]
        selected_genes = [kept_genes[i] for i in top_local_idx]
    print(f"Selected {top_k} genes among all genes.")

  
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    return X, batch_ids, sex_region_ids, n_batches, n_classes, class_names, selected_genes

#plot making for latent space---------

def extract_latents(model, loader, device="cpu"):
    model=model.to(device)
    Z = []
    batches = []
    sex_regions = []

    with torch.no_grad():
        for x, batch_ids, sr_ids in loader:
            x = x.to(device)
            batch_ids = batch_ids.to(device)

            _, z = model.encode_decode(x, batch_ids)
            Z.append(z.cpu().numpy())
            batches.append(batch_ids.cpu().numpy())
            sex_regions.append(sr_ids.cpu().numpy())

    Z = np.concatenate(Z, axis=0)
    batches = np.concatenate(batches, axis=0)
    sex_regions = np.concatenate(sex_regions, axis=0)

    return Z, batches, sex_regions

# ============================================================
# 2) DATASET (PL-only)
# ============================================================

class GeneDataset(Dataset):
    def __init__(self, X, batch_ids, sex_region_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.batch = torch.tensor(batch_ids, dtype=torch.long)
        self.y = torch.tensor(sex_region_ids, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.batch[idx], self.y[idx]


# ============================================================
# 3) MODEL: AE + GRL + Heads
# ============================================================

class Encoder(nn.Module):
    def __init__(self, n_genes, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_batches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, z, b_oh):
        return self.net(torch.cat([z, b_oh], dim=1))


class ConditionalAE(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(n_genes, latent_dim, hidden_dim)
        self.decoder = Decoder(n_genes, n_batches, latent_dim, hidden_dim)
        self.n_batches = n_batches

    def forward(self, x, batch_ids):
        z = self.encoder(x)
        b_oh = F.one_hot(batch_ids, num_classes=self.n_batches).float()
        x_hat = self.decoder(z, b_oh)
        return x_hat, z


class GRLFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd=0.05):
        super().__init__()
        self.lambd = float(lambd)
    def set_lambda(self, lambd: float):
        self.lambd=float(lambd)
    def forward(self, x):
        return GRLFunction.apply(x, self.lambd)


class BatchPredictor(nn.Module):
    def __init__(self, latent_dim, n_batches, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_batches),
        )

    def forward(self, z):
        return self.net(z)


class SexRegionPredictor(nn.Module):
    def __init__(self, latent_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z):
        return self.net(z)


class FullModel3Loss(nn.Module):
    def __init__(self, n_genes, n_batches, n_classes, latent_dim=16, hidden_dim=128, head_dim=64,grl_lambda=0.05):
        super().__init__()
        self.ae = ConditionalAE(n_genes, n_batches, latent_dim, hidden_dim)
        self.grl = GRL(grl_lambda)
        self.batch_head = BatchPredictor(latent_dim, n_batches, hidden_dim=head_dim)
        self.sr_head = SexRegionPredictor(latent_dim, n_classes, hidden_dim=head_dim)

    # def forward(self, x, batch_ids):
    #     x_hat, z = self.ae(x, batch_ids)
    #     batch_logits = self.batch_head(self.grl(z))
    #     sr_logits = self.sr_head(z)
    #     return x_hat, z, batch_logits, sr_logits

    def encode_decode (self, x, batch_ids):
        x_hat,z= self.ae(x, batch_ids)
        return x_hat, z
    def batch_logits(self,z,use_grl:bool):
        if use_grl:
            z=self.grl(z)
        return self.batch_head(z)
    def sr_logits(self,z):
        return self.sr_head(z)
    def forward(self, x, batch_ids):
        x_hat, z = self.encode_decode(x, batch_ids)
        batch_logits = self.batch_logits(z, use_grl=True)
        sr_logits = self.sr_logits(z)
        return x_hat, z, batch_logits, sr_logits

# ============================================================
# 4) TRAINING
# ============================================================

def train_three_losses_max_norm(
    model,
    loader,
    batch_weights,
    n_epochs=300,
    lr=5e-4,
    device="cpu",
    eps=1e-8
):
    """
    PL-only training.
    Max-normalized losses:
      L1n = L1 / max(L1_so_far)
      L2n = L2 / max(L2_so_far)   (Loss2 uses weighted CE)
      L3n = L3 / max(L3_so_far)

    Total = L1n + L2n + L3n
    """

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    max_l1, max_l2, max_l3 = eps, eps, eps
    hist_l1n, hist_l2n, hist_l3n, hist_totn = [], [], [], []
    hist_l1_raw, hist_l2_raw, hist_l3_raw = [], [], []
    hist_batch_acc, hist_sr_acc = [], []

    batch_weights = batch_weights.to(device)

    for epoch in range(n_epochs):
        model.train()
        lam_max=0.4
        lam=lam_max*(epoch/(max(1,n_epochs -1)))
        model.grl.set_lambda(lam)
        l1ns, l2ns, l3ns, totns = [], [], [], []
        l1_raws, l2_raws, l3_raws = [], [], []
        baccs, saccs = [], []

        for x, batch_ids, sr_ids in loader:
            x = x.to(device)
            batch_ids = batch_ids.to(device)
            sr_ids = sr_ids.to(device)

            x_hat, z, batch_logits, sr_logits = model(x, batch_ids)

            # RAW losses
            loss1 = F.mse_loss(x_hat, x)
            loss2 = F.cross_entropy(batch_logits, batch_ids, weight=batch_weights)
            loss3 = F.cross_entropy(sr_logits, sr_ids)

            # update running maxima
            max_l1 = max(max_l1, float(loss1.detach().item()))
            max_l2 = max(max_l2, float(loss2.detach().item()))
            max_l3 = max(max_l3, float(loss3.detach().item()))

            # normalized losses
            loss1_n = loss1 / (max_l1 + eps)
            loss2_n = loss2 / (max_l2 + eps)
            loss3_n = loss3 / (max_l3 + eps)

            total_n = loss1_n + loss2_n + loss3_n

            opt.zero_grad()
            total_n.backward()
            opt.step()

            with torch.no_grad():
                bacc = (batch_logits.argmax(1) == batch_ids).float().mean().item()
                sacc = (sr_logits.argmax(1) == sr_ids).float().mean().item()

            # save batch stats
            l1_raws.append(loss1.item())
            l2_raws.append(loss2.item())
            l3_raws.append(loss3.item())

            l1ns.append(loss1_n.item())
            l2ns.append(loss2_n.item())
            l3ns.append(loss3_n.item())
            totns.append(total_n.item())

            baccs.append(bacc)
            saccs.append(sacc)

        # epoch means
        l1_raw = float(np.mean(l1_raws))
        l2_raw = float(np.mean(l2_raws))
        l3_raw = float(np.mean(l3_raws))

        l1n = float(np.mean(l1ns))
        l2n = float(np.mean(l2ns))
        l3n = float(np.mean(l3ns))
        tot_n = float(np.mean(totns))

        bacc_e = float(np.mean(baccs))
        sacc_e = float(np.mean(saccs))

        # store histories
        hist_l1_raw.append(l1_raw); hist_l2_raw.append(l2_raw); hist_l3_raw.append(l3_raw)
        hist_l1n.append(l1n); hist_l2n.append(l2n); hist_l3n.append(l3n); hist_totn.append(tot_n)
        hist_batch_acc.append(bacc_e); hist_sr_acc.append(sacc_e)

        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"RAW(L1={l1_raw:.4f}, L2={l2_raw:.4f}, L3={l3_raw:.4f}) | "
            f"MAXN(Total={tot_n:.4f}; L1n={l1n:.3f}, L2n={l2n:.3f}, L3n={l3n:.3f}) | "
            f"BatchAcc={bacc_e:.4f} | SexRegionAcc={sacc_e:.4f}"
        )


        #wandb log (one log per epoch)
        wandb.log({
            "epoch": epoch + 1,
            "loss_raw/L1_recon": l1_raw,
            "loss_raw/L2_batch_weighted": l2_raw,
            "loss_raw/L3_sex_region": l3_raw,
            "loss_norm/L1_recon": l1n,
            "loss_norm/L2_batch": l2n,
            "loss_norm/L3_sex_region": l3n,
            "loss_norm/total": tot_n,
            "accuracy/batch": bacc_e,
            "accuracy/sex_region": sacc_e,
            "max/L1": max_l1,
            "max/L2": max_l2,
            "max/L3": max_l3,
            "grl_lambda": float(getattr(model.grl, "lambd", 0.0)),
        })
    plt.figure(figsize=(10, 6))
    plt.plot(hist_l1n, label="L1/max(L1)", linewidth=2)
    plt.plot(hist_l2n, label="L2/max(L2) [weighted CE]", linewidth=2)
    plt.plot(hist_l3n, label="L3/max(L3)", linewidth=2)
    plt.plot(hist_totn, label="Total", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (max-normalized)")
    plt.title("Max-normalized Loss Curves (PL-only + weighted batch CE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist_batch_acc, label="Batch accuracy", linewidth=2)
    plt.plot(hist_sr_acc, label="SexRegion accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracies (PL-only + weighted batch CE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return model

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    BASE_PATH = "/Users/nawshadbintanizam/Documents/Documents/Research/data"
    COMMON_GENES_CSV = os.path.join(BASE_PATH, "common_genes.csv")
    datasets = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "MERFISH-C57BL6J-638850"]

    df_all = load_and_prepare_data(BASE_PATH, datasets)
    df_pl = df_all[df_all["parcellation_structure"].astype(str) == "PL"].copy()
    print(f"\nFiltered to PL only: {df_pl.shape[0]} cells")
    assert (df_pl["parcellation_structure"].astype(str) == "PL").all()
    X, batch_ids, sex_region_ids, n_batches, n_classes, class_names, selected_genes = extract_training_matrices(
        df_pl,
        common_genes_csv=COMMON_GENES_CSV,
        top_k=None,
        apply_combat=True,
        expr_threshold=0.00,
        min_pc_cells=0.80

    )

    TOP_K = selected_genes
    LR = 1e-4
    BATCH_SIZE = 256
    LATENT_DIM=32
    HIDDEN_DIM=128
    GRL_LAMBDA = 0.0
    HEAD_DIM=128
    APPLY_COMBAT = True
    LOSS_NORM = "max"
    WEIGHTED_BATCH_CE = True
    PL_ONLY = True

    counts = np.bincount(batch_ids, minlength=n_batches)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    print("\nBatch counts (PL-only):", counts)
    print("Batch weights:", weights)
    batch_weights = torch.tensor(weights, dtype=torch.float32)

    dataset = GeneDataset(X, batch_ids, sex_region_ids)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    model = FullModel3Loss(
        n_genes=X.shape[1],
        n_batches=n_batches,
        n_classes=n_classes,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        head_dim=HEAD_DIM,
        grl_lambda=GRL_LAMBDA
    )
    #latent space plot

    Z, batch_ids_all, sr_ids_all = extract_latents(
    model,
    loader,
    device=device

)
    pca = PCA(n_components=min(10, Z.shape[1]))
    Z_pca = pca.fit_transform(Z)

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        n_components=2,
        random_state=42
    )
    Z_umap = reducer.fit_transform(Z_pca)
    def plot_latent(Z_2d, labels, title, label_names=None):
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            Z_2d[:, 0],
            Z_2d[:, 1],
            c=labels,
            s=5,
            alpha=0.6,
            cmap="tab10"
        )
        plt.title(title)
        plt.xlabel("Latent 1")
        plt.ylabel("Latent 2")
        plt.grid(False)

        if label_names is not None:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, label_names, title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    # Batch-colored
    plot_latent(
        Z_umap,
        batch_ids_all,
        title="Latent space (colored by batch)",
        label_names=None
    )

    # Sex-region–colored
    plot_latent(
        Z_umap,
        sr_ids_all,
        title="Latent space (colored by sex_region)",
        label_names=None
)
    wandb.init(
        entity="nawshad-binta-university-of-pittsburgh",
        project="Rostral_caudal_transcriptomic_differences",
        name=f"PL_AE_GRL_lambda ramping, Batchsize {BATCH_SIZE}, Latentspace {LATENT_DIM}",
        config={
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "top_k_genes": TOP_K,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "grl_lambda": GRL_LAMBDA,
            "loss_normalization": LOSS_NORM,
            "weighted_batch_ce": WEIGHTED_BATCH_CE,
            "pl_only": PL_ONLY,
            "optimizer": "Adam",
            "device": device
        }
    )
    wandb.watch(model, log="gradients", log_freq=100)

    trained_model = train_three_losses_max_norm(
        model,
        loader,
        batch_weights=batch_weights,
        n_epochs=400,
        lr=LR,
        device=device,
        eps=1e-8
    )

    wandb.finish()
