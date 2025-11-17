"""
ASR Visualization Dashboard

Creates comprehensive visualizations for ASR analysis:
- ASR matrices (agent × attack type)
- Per-category ASR analysis
- Statistical comparisons
- Trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ASRVisualizer:
    """ASR visualization dashboard."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing ASR results
        """
        if results_dir is None:
            results_dir = Path(__file__).parent / "asr_reports"
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self) -> Dict:
        """Load ASR results from JSON file."""
        json_path = self.results_dir / "asr_results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Results file not found: {json_path}")
        
        with open(json_path, "r") as f:
            return json.load(f)
    
    def plot_asr_matrix(self, results: Dict, output_path: Optional[Path] = None):
        """Plot ASR matrix: agents × attack types."""
        if output_path is None:
            output_path = self.output_dir / "asr_matrix.png"
        
        # Prepare data
        data = []
        agents = []
        attack_types = []
        
        for attack_type, analysis in results.items():
            if "error" in analysis:
                continue
            
            attack_types.append(attack_type)
            for agent, metrics in analysis["agent_asr"].items():
                if agent not in agents:
                    agents.append(agent)
                data.append({
                    "agent": agent,
                    "attack_type": attack_type,
                    "asr": metrics["asr"]
                })
        
        df = pd.DataFrame(data)
        matrix = df.pivot(index="agent", columns="attack_type", values="asr")
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "ASR"},
            linewidths=0.5
        )
        plt.title("ASR Matrix: Agent × Attack Type", fontsize=16, fontweight="bold")
        plt.xlabel("Attack Type", fontsize=12)
        plt.ylabel("Agent", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"ASR matrix saved to: {output_path}")
    
    def plot_asr_by_agent(self, results: Dict, output_path: Optional[Path] = None):
        """Plot ASR comparison across agents."""
        if output_path is None:
            output_path = self.output_dir / "asr_by_agent.png"
        
        # Prepare data
        data = []
        for attack_type, analysis in results.items():
            if "error" in analysis:
                continue
            
            for agent, metrics in analysis["agent_asr"].items():
                data.append({
                    "agent": agent,
                    "attack_type": attack_type,
                    "asr": metrics["asr"],
                    "ci_lower": metrics["ci_lower"],
                    "ci_upper": metrics["ci_upper"]
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        attack_types = df["attack_type"].unique()
        agents = df["agent"].unique()
        x = np.arange(len(agents))
        width = 0.25
        
        for i, attack_type in enumerate(attack_types):
            subset = df[df["attack_type"] == attack_type]
            asr_values = [subset[subset["agent"] == agent]["asr"].values[0] 
                         if len(subset[subset["agent"] == agent]) > 0 else 0
                         for agent in agents]
            
            offset = (i - len(attack_types) / 2) * width + width / 2
            bars = ax.bar(x + offset, asr_values, width, label=attack_type, alpha=0.8)
        
        ax.set_xlabel("Agent", fontsize=12)
        ax.set_ylabel("ASR", fontsize=12)
        ax.set_title("ASR by Agent and Attack Type", fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"ASR by agent plot saved to: {output_path}")
    
    def plot_asr_with_ci(self, results: Dict, output_path: Optional[Path] = None):
        """Plot ASR with confidence intervals."""
        if output_path is None:
            output_path = self.output_dir / "asr_with_ci.png"
        
        # Prepare data
        data = []
        for attack_type, analysis in results.items():
            if "error" in analysis:
                continue
            
            for agent, metrics in analysis["agent_asr"].items():
                data.append({
                    "agent": agent,
                    "attack_type": attack_type,
                    "asr": metrics["asr"],
                    "ci_lower": metrics["ci_lower"],
                    "ci_upper": metrics["ci_upper"]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(1, len(df["attack_type"].unique()), 
                                figsize=(18, 6), sharey=True)
        
        if len(df["attack_type"].unique()) == 1:
            axes = [axes]
        
        for idx, attack_type in enumerate(df["attack_type"].unique()):
            subset = df[df["attack_type"] == attack_type]
            agents = subset["agent"].values
            asr_values = subset["asr"].values
            ci_lower = subset["ci_lower"].values
            ci_upper = subset["ci_upper"].values
            
            x_pos = np.arange(len(agents))
            
            axes[idx].errorbar(
                x_pos, asr_values,
                yerr=[asr_values - ci_lower, ci_upper - asr_values],
                fmt="o", capsize=5, capthick=2, markersize=8
            )
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(agents, rotation=45, ha="right")
            axes[idx].set_ylabel("ASR", fontsize=12)
            axes[idx].set_title(f"{attack_type.upper()}", fontsize=14, fontweight="bold")
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].set_ylim(0, 1.1)
        
        plt.suptitle("ASR with 95% Confidence Intervals", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"ASR with CI plot saved to: {output_path}")
    
    def plot_category_asr(self, results: Dict, attack_type: str = "harmful",
                         output_path: Optional[Path] = None):
        """Plot ASR by harm category."""
        if output_path is None:
            output_path = self.output_dir / f"asr_by_category_{attack_type}.png"
        
        if attack_type not in results or "error" in results[attack_type]:
            print(f"No data for attack type: {attack_type}")
            return
        
        analysis = results[attack_type]
        if not analysis.get("category_asr"):
            print(f"No category data for {attack_type}")
            return
        
        # Prepare data
        data = []
        for category, cat_metrics in analysis["category_asr"].items():
            for agent, metrics in cat_metrics.items():
                data.append({
                    "category": category,
                    "agent": agent,
                    "asr": metrics["asr"]
                })
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        matrix = df.pivot(index="category", columns="agent", values="asr")
        
        plt.figure(figsize=(14, max(8, len(matrix) * 0.5)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "ASR"},
            linewidths=0.5
        )
        plt.title(f"ASR by Harm Category - {attack_type.upper()}", 
                 fontsize=16, fontweight="bold")
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Harm Category", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Category ASR plot saved to: {output_path}")
    
    def plot_statistical_comparison(self, results: Dict, output_path: Optional[Path] = None):
        """Plot statistical comparison between agents."""
        if output_path is None:
            output_path = self.output_dir / "statistical_comparison.png"
        
        # Prepare data for statistical test
        from scipy import stats
        
        # Group by agent across all attack types
        agent_data = {}
        for attack_type, analysis in results.items():
            if "error" in analysis:
                continue
            
            for agent, metrics in analysis["agent_asr"].items():
                if agent not in agent_data:
                    agent_data[agent] = []
                agent_data[agent].append(metrics["asr"])
        
        # Calculate mean and std
        agent_stats = {
            agent: {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }
            for agent, values in agent_data.items()
        }
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot with error bars
        agents = list(agent_stats.keys())
        means = [agent_stats[a]["mean"] for a in agents]
        stds = [agent_stats[a]["std"] for a in agents]
        
        x_pos = np.arange(len(agents))
        ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(agents, rotation=45, ha="right")
        ax1.set_ylabel("Mean ASR", fontsize=12)
        ax1.set_title("Mean ASR Across All Attack Types", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        
        # Box plot
        data_for_box = [agent_stats[a]["values"] for a in agents]
        ax2.boxplot(data_for_box, labels=agents)
        ax2.set_ylabel("ASR", fontsize=12)
        ax2.set_title("ASR Distribution by Agent", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Statistical comparison plot saved to: {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("GENERATING ASR VISUALIZATIONS")
        print("="*60)
        
        results = self.load_results()
        
        # Generate all plots
        self.plot_asr_matrix(results)
        self.plot_asr_by_agent(results)
        self.plot_asr_with_ci(results)
        self.plot_statistical_comparison(results)
        
        # Category plots for each attack type
        for attack_type in ["harmful", "jailbreak"]:
            if attack_type in results:
                self.plot_category_asr(results, attack_type)
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS GENERATED")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR Visualization Dashboard")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory containing ASR results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) if args.results_dir else None
    visualizer = ASRVisualizer(results_dir=results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    #generate a bar chart plot for this data using seaborn, with different colors for each agent with a purple
    data = {"Ant": 13.30, "Bear": 5.60, "Chameleon": 0.00, "Eagle": 13.33, "Elephant": 8.45, "Fox": 4.00, "Wolf": 2.35}
    sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
    plt.xlabel("Agent")
    plt.ylabel("ASR")
    plt.title("ASR by Agent")
    plt.savefig("asr_by_agent.png")
    plt.close()

