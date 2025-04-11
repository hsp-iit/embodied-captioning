import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def mean_cosine_similarity(captions_list):
    """Compute mean cosine similarity between all pairs of captions."""
    if len(captions_list) < 2:
        return np.nan  # Not enough captions to compare
    captions_list = list(captions_list)
    embeddings = model.encode(captions_list, convert_to_numpy=True)
    n = len(embeddings)
    similarities = [
        1 - cosine(embeddings[i], embeddings[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    return np.mean(similarities) if similarities else np.nan


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarities from CSV files and generate a grouped bar plot."
    )
    parser.add_argument(
        "--csv_not_finetuned",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the non-finetuned model."
    )
    parser.add_argument(
        "--csv_random_finetuned_vanilla",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the vanilla finetuned model."
    )
    parser.add_argument(
        "--csv_random_finetuned_triplet",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the contrastive finetuned model."
    )
    parser.add_argument(
        "--csv_frontier_finetuned_vanilla",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the vanilla finetuned model."
    )
    parser.add_argument(
        "--csv_frontier_finetuned_triplet",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the contrastive finetuned model."
    )
    parser.add_argument(
        "--csv_policy_finetuned_vanilla",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the vanilla finetuned model."
    )
    parser.add_argument(
        "--csv_policy_finetuned_triplet",
        type=str,
        default="...",
        # required=True,
        help="Path to the CSV file for the contrastive finetuned model."
    )

    parser.add_argument(
        "--output_plot_path",
        type=str,
        default="a.png",
        # required=True,
        help="Path to save the output plot (e.g., plot.png)."
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        # required=True,
        help="Whether to visualize the plot or not."
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        # required=True,
        help="Whether to save the plot or not."
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Load args
    args = get_args()
    visualize = args.visualize
    save = args.save

    df = pd.read_csv(args.csv_not_finetuned)
    df_random_va = pd.read_csv(args.csv_random_finetuned_vanilla)
    df_random_tr = pd.read_csv(args.csv_random_finetuned_triplet)
    df_frontier_va = pd.read_csv(args.csv_frontier_finetuned_vanilla)
    df_frontier_tr = pd.read_csv(args.csv_frontier_finetuned_triplet)
    df_policy_va = pd.read_csv(args.csv_policy_finetuned_vanilla)
    df_policy_tr = pd.read_csv(args.csv_policy_finetuned_triplet)

    df.columns = df.columns.str.strip()
    df_random_va.columns = df_random_va.columns.str.strip()
    df_random_tr.columns = df_random_tr.columns.str.strip()
    df_frontier_va.columns = df_frontier_va.columns.str.strip()
    df_frontier_tr.columns = df_frontier_tr.columns.str.strip()
    df_policy_va.columns = df_policy_va.columns.str.strip()
    df_policy_tr.columns = df_policy_tr.columns.str.strip()

    # Ensure episode_id and object_id are strings
    df["episode_id"] = df["episode_id"].astype(str)
    df["object_id"] = df["object_id"].astype(str)
    df_random_va["episode_id"] = df_random_va["episode_id"].astype(str)
    df_random_va["object_id"] = df_random_va["object_id"].astype(str)
    df_random_tr["episode_id"] = df_random_tr["episode_id"].astype(str)
    df_random_tr["object_id"] = df_random_tr["object_id"].astype(str)
    df_frontier_va["episode_id"] = df_frontier_va["episode_id"].astype(str)
    df_frontier_va["object_id"] =  df_frontier_va["object_id"].astype(str)
    df_frontier_tr["episode_id"] = df_frontier_tr["episode_id"].astype(str)
    df_frontier_tr["object_id"] =  df_frontier_tr["object_id"].astype(str)
    df_policy_va["episode_id"] = df_policy_va["episode_id"].astype(str)
    df_policy_va["object_id"] =  df_policy_va["object_id"].astype(str)
    df_policy_tr["episode_id"] = df_policy_tr["episode_id"].astype(str)
    df_policy_tr["object_id"] =  df_policy_tr["object_id"].astype(str)

    # Drop rows with missing captions and convert captions to string
    df = df.dropna(subset=["proposed_caption"])
    df_random_va = df_random_va.dropna(subset=["proposed_caption"])
    df_random_tr = df_random_tr.dropna(subset=["proposed_caption"])
    df_frontier_va = df_frontier_va.dropna(subset=["proposed_caption"])
    df_frontier_tr = df_frontier_tr.dropna(subset=["proposed_caption"])
    df_policy_va = df_policy_va.dropna(subset=["proposed_caption"])
    df_policy_tr = df_policy_tr.dropna(subset=["proposed_caption"])
    df["proposed_caption"] = df["proposed_caption"].astype(str)
    df_random_va["proposed_caption"] = df_random_va["proposed_caption"].astype(str)
    df_random_tr["proposed_caption"] = df_random_tr["proposed_caption"].astype(str)
    df_frontier_va["proposed_caption"] = df_frontier_va["proposed_caption"].astype(str)
    df_frontier_tr["proposed_caption"] = df_frontier_tr["proposed_caption"].astype(str)
    df_policy_va["proposed_caption"] = df_policy_va["proposed_caption"].astype(str)
    df_policy_tr["proposed_caption"] = df_policy_tr["proposed_caption"].astype(str)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Group by episode_id and object_id and compute mean cosine similarity for each group
    result = df.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_random_va = df_random_va.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_random_tr = df_random_tr.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_frontier_va = df_frontier_va.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_frontier_tr = df_frontier_tr.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_policy_va = df_policy_va.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()
    result_policy_tr = df_policy_tr.groupby(["episode_id", "object_id"])["proposed_caption"].apply(
        lambda x: mean_cosine_similarity(list(x))
    ).reset_index()

    result = result.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_random_va = result_random_va.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_random_tr = result_random_tr.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_frontier_va = result_frontier_va.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_frontier_tr = result_frontier_tr.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_policy_va = result_policy_va.rename(columns={"proposed_caption": "mean_cosine_similarity"})
    result_policy_tr = result_policy_tr.rename(columns={"proposed_caption": "mean_cosine_similarity"})

    total_mean = result["mean_cosine_similarity"].mean()
    total_mean_random_va = result_random_va["mean_cosine_similarity"].mean()
    total_mean_random_tr = result_random_tr["mean_cosine_similarity"].mean()
    total_mean_frontier_va = result_frontier_va["mean_cosine_similarity"].mean()
    total_mean_frontier_tr = result_frontier_tr["mean_cosine_similarity"].mean()
    total_mean_policy_va = result_policy_va["mean_cosine_similarity"].mean()
    total_mean_policy_tr = result_policy_tr["mean_cosine_similarity"].mean()
    print(f"Mean CS (off-the-shelf): {total_mean:.6f}")
    print(f"Mean CS (random-vanilla): {total_mean_random_va:.6f}")
    print(f"Mean CS (random-triplet): {total_mean_random_tr:.6f}")
    print(f"Mean CS (frontier-vanilla): {total_mean_frontier_va:.6f}")
    print(f"Mean CS (frontier-triplet): {total_mean_frontier_tr:.6f}")
    print(f"Mean CS (policy-vanilla): {total_mean_policy_va:.6f}")
    print(f"Mean CS (policy-triplet): {total_mean_policy_tr:.6f}")



    merged = pd.merge(result_random_va, result_random_tr, on=["episode_id", "object_id"], how="inner", suffixes=("_", "_df_nf"))
    merged = pd.merge(merged, total_mean_random_tr, on=["episode_id", "object_id"], how="inner", suffixes=(""))
    print(f"{merged.keys()=}")

    merged["object_label"] = merged["episode""_id"] + "_" + merged["object_id"]

    x = np.arange(len(merged))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_df = ax.bar(x - width, merged["mean_cosine_similarity_df_nf"], width, label="Non-finetuned (df_nf)")
    bars_df_nf = ax.bar(x, merged["mean_cosine_similarity_df"], width, label="Finetuned (df)")
    bars_df_tr = ax.bar(x + width, merged["mean_cosine_similarity"], width, label="TR-finetuned (df_tr)")

    ax.set_xlabel("Object (episode_id_object_id)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Cosine Similarity per Object: Finetuned vs. Non-finetuned")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["object_label"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    if visualize:
        plt.show()
    if save:
        plt.savefig(args.output_plot_path, dpi=300, bbox_inches='tight')


    # Generating data with different standard deviations
    mean_nf = np.expand_dims(result_nf["mean_cosine_similarity"].values, axis=1)
    mean_f = np.expand_dims(result["mean_cosine_similarity"].values, axis=1)
    mean_tr = np.expand_dims(result_tr["mean_cosine_similarity"].values, axis=1)
    data = np.concatenate((mean_nf, mean_f, mean_tr), axis=1)
    # Discard nan due to images with only one caption
    data = data[~np.isnan(data).all(axis=1)]

    policy = os.path.basename(args.csv_finetuned_vanilla).split("_")[0]
    captioner = os.path.basename(args.csv_finetuned_vanilla).split("_")[1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.violinplot(data, showmeans=False, showmedians=True)
    ax.yaxis.grid(True)
    ax.set_xticks([x+1 for x in range(data.shape[1])],
                  labels=['Off-the-shelf', 'Vanilla', 'Triplet'], fontsize=20)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=20)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Finetuning", fontsize=20)
    ax.set_ylabel("Cosine Similarity", fontsize=20)
    ax.set_title("{}_{}".format(policy, captioner))
    plt.tight_layout()
    if visualize:
        plt.show()
    if save:
        plt.savefig(args.output_plot_path.replace(".png", "_violin.png"), dpi=100, bbox_inches='tight')

    print(f"Plots saved as '{args.output_plot_path}'")