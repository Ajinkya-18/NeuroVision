import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


rouge_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
rouge_metrics_values = [30.04, 7.23, 27.51]

other_metrics = [ "BLEU Score", "Semantic Similarity"]
other_metrics_values = [2.34, 0.1045]

error_metrics = ["WER", "CER"]
error_metrics_values = [132.68, 101.95]


def show_values_on_bars(ax, **kwargs):
    """
    Add data labels to the top of each bar in a plot.
    """
    for bar in ax.patches:
        # Get the bar's x and y coordinates
        _x = bar.get_x() + bar.get_width() / 2
        _y = bar.get_height()

        # Format the value. Use 4 decimal places for values < 1, 
        # and 2 decimal places for values >= 1.

        if _y < 1:
            value = f'{_y:.4f}'
        else:
            value = f'{_y:.2f}'

        # Annotate the bar
        ax.annotate(
            value, 
            (_x, _y), 
            ha="center", 
            va="bottom", 
            xytext=(0, 5), 
            textcoords = "offset points", 
            **kwargs
        )


# Convert data to Pandas DataFrames
df_rouge = pd.DataFrame({'Metric': rouge_metrics, "Score": rouge_metrics_values})
df_other = pd.DataFrame({"Metric": other_metrics, 'Score': other_metrics_values})
df_error = pd.DataFrame({"Metric": error_metrics, "Score": error_metrics_values})


sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fig.suptitle("NeuroVision: Performance Metrics", fontsize=20, y=1.03)

sns.barplot(
    data=df_rouge, 
    x="Metric",
    y='Score',
    ax=axes[0],
    palette="Blues_d"
)

axes[0].set_title("ROUGE Metrics (Higher is Better)", fontsize=14, pad=15)
axes[0].set_ylabel("Score (%)")
axes[0].set_xlabel("Metric")
show_values_on_bars(axes[0])

sns.barplot(
    data=df_other,
    x="Metric",
    y="Score",
    ax=axes[1],
    palette="Greens_d"
)

axes[1].set_title("reports/Generation Metrics (Higher is Better)", fontsize=14, pad=15)
axes[1].set_ylabel("Score")
axes[1].set_xlabel("Metric")
axes[1].set_ylim(0, max(other_metrics_values) * 1.2)
show_values_on_bars(axes[1])

sns.barplot(
    data=df_error,
    x="Metric", 
    y="Score",
    ax=axes[2],
    palette="Reds_d"
)

axes[2].set_title("Error Metrics (Lower is better)", fontsize=14, pad=15)
axes[2].set_ylabel("Error Rate (%)")
axes[2].set_xlabel("Metric")
axes[2].set_ylim(0, max(error_metrics_values) * 1.2)
show_values_on_bars(axes[2])


plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

fig.savefig("neurovision_performance_metrics.png", dpi=300, bbox_inches='tight')

