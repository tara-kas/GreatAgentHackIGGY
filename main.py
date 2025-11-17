import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """Generate a simple ASR bar chart for the configured agents."""
    # ASR percentages for each agent (already in % units)
    agents = ["Ant", "Bear", "Chameleon", "Eagle", "Elephant", "Fox", "Wolf"]
    asr_percentages = [13.30, 5.60, 0.00, 13.33, 8.45, 4.00, 2.35]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, asr_percentages, color="#4C72B0", alpha=0.85)

    # Add value labels on top of bars
    for bar, value in zip(bars, asr_percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("ASR (%)")
    plt.xlabel("Agent")
    plt.title("Attack Success Rate (ASR) by Agent")
    plt.ylim(0, max(asr_percentages) * 1.2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save the figure in the project root
    output_path = Path(__file__).parent / "asr_agents_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ASR bar chart saved to: {output_path}")


if __name__ == "__main__":
    main()
