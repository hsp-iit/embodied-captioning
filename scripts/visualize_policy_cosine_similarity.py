import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Process function for the three datasets
def process_data(data, label):
    df = pd.DataFrame(list(data.values()))
    df = df[df["images_number"] > 1]  # Filter rows with images_number > 1
    df["dataset"] = label
    return df


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarities from CSV files and generate a grouped bar plot."
    )
    parser.add_argument(
        "--random_path",
        type=str,
        default="/home/tapicella/Downloads/plot_simca/disagreement_images_dict_randomgoal.json",
        # required=True,
        help="Path to the CSV file for the non-finetuned model."
    )
    parser.add_argument(
        "--frontier_path",
        type=str,
        default="/home/tapicella/Downloads/plot_simca/disagreement_images_dict_frontier.json",
        # required=True,
        help="Path to the CSV file for the vanilla finetuned model."
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="/home/tapicella/Downloads/plot_simca/disagreement_images_dict_our_400k.json",
        # required=True,
        help="Path to the CSV file for the contrastive finetuned model."
    )
    parser.add_argument(
        "--output_plot_path",
        type=str,
        default="gibson_policy_cosine_similarity.png",
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
        default=True,
        # required=True,
        help="Whether to save the plot or not."
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    random_path = args.random_path
    frontier_path = args.frontier_path
    policy_path = args.policy_path
    visualize = args.visualize
    save = args.save
    output_plot_path = args.output_plot_path

    with open(random_path, 'r') as f:
        data_random = json.load(f)
    with open(frontier_path, 'r') as f:
        data_frontier = json.load(f)
    with open(policy_path, 'r') as f:
        data_policy = json.load(f)

    # Process each dataset with labels
    df_random = process_data(data_random, "Random")
    df_frontier = process_data(data_frontier, "Frontier")
    df_policy = process_data(data_policy, "CLA")

    # Combine all datasets
    df_combined = pd.concat([df_random, df_frontier, df_policy], ignore_index=True)

    # Create the figure with a compact layout
    positions = [0, 1, 2]
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(6, 3))
    fontsize = 16  # Adjust text size

    # Violin plot with tighter violins and skyblue color
    sns.violinplot(
        x="dataset",
        y="cosine_sim",
        data=df_combined,
        inner="quartile",
        linewidth=1.2,
        width=0.4,  # Narrower violins
        cut=0,
        scale="count",
        bw=0.15,  # Reduced bandwidth for a tighter shape
        color="skyblue",
        order=["Random", "Frontier", "CLA"],
        native_scale=True
    )

    # Set y-axis limits *after* plotting
    ax.set_xticks(positions,
                  labels=["Random", "Frontier", "CLA"],
                  fontsize=fontsize)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=fontsize)

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0.0, 1.00)
    ax.set_xlabel("", fontsize=fontsize)
    ax.set_ylabel("Cosine Similarity", fontsize=fontsize)

    # Enable grid on the y-axis
    ax.yaxis.grid(True)
    plt.tight_layout()

    if visualize:
        plt.show()
    if save:
        plt.savefig(output_plot_path, dpi=100, bbox_inches='tight')
