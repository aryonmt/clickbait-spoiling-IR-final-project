import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import extract_primary_tag


class TextLengthAnalyzer:
    """Analyzer to calculate sequence lengths and statistical percentiles for

    text features in the dataset.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "reports"):
        """Initializes the analyzer with a dataset and an output directory.

        Args:
            df (pd.DataFrame): The target dataset.
            output_dir (str): Directory where generated EDA plots will be saved.
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _count_words(text_input) -> int:
        """Helper function to calculate word count for both strings and lists of

        strings.

        Args:
            text_input: Can be a string or a list of strings (e.g.,
              targetParagraphs).

        Returns:
            int: Total word count.
        """
        if isinstance(text_input, list):
            full_text = " ".join(text_input)
        elif isinstance(text_input, str):
            full_text = text_input
        else:
            return 0
        return len(full_text.split())

    def process_lengths(self) -> pd.DataFrame:
        """Computes word counts for postText and targetParagraphs columns.

        Returns:
            pd.DataFrame: DataFrame containing the processed lengths.
        """
        print("\nCalculating word counts for sequences...")
        self.df["post_word_count"] = self.df["postText"].apply(self._count_words)
        self.df["article_word_count"] = self.df["targetParagraphs"].apply(
            self._count_words
        )
        return self.df

    def display_statistics(self) -> None:
        """Computes and prints percentile statistics for sequence lengths to

        guide max_length selection.
        """
        print("\n=== Sequence Length Statistical Summary ===")
        columns_to_analyze = ["post_word_count", "article_word_count"]

        stats = self.df[columns_to_analyze].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
        print(stats)

    def plot_distributions(self) -> None:
        """Generates and saves distribution plots for sequence lengths."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(self.df["post_word_count"], kde=True, color="skyblue")
        plt.title("Distribution of Clickbait Post Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.histplot(self.df["article_word_count"], kde=True, color="salmon")
        plt.title("Distribution of Article Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "sequence_length_distributions.png")
        plt.savefig(plot_path)
        print(f"\n[SUCCESS] Distribution plots saved to: {plot_path}")
        plt.close()

    def analyze_class_balance(self) -> pd.Series:
        """Analyzes and prints the distribution of spoiler types (tags).

        Returns:
            pd.Series: Percentage distribution of each class.
        """
        print("\n=== Class Distribution Analysis (Task 1) ===")
        class_series = self.df["tags"].apply(extract_primary_tag)

        counts = class_series.value_counts()
        percentages = class_series.value_counts(normalize=True) * 100

        distribution_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})
        print(distribution_df)
        return class_series

    def analyze_spoiler_overlap(self) -> None:
        """Calculates the lexical overlap (Jaccard Similarity) between the

        human spoiler and the article content to verify if it is fully
        extractive.
        """
        print("\n=== Spoiler Lexical Overlap Analysis ===")

        overlaps = []
        for _, row in self.df.iterrows():
            paragraphs = (
                " ".join(row["targetParagraphs"])
                if isinstance(row["targetParagraphs"], list)
                else str(row["targetParagraphs"])
            )
            spoilers = (
                " ".join(row["spoiler"])
                if isinstance(row["spoiler"], list)
                else str(row["spoiler"])
            )

            words_article = set(paragraphs.lower().split())
            words_spoiler = set(spoilers.lower().split())

            if not words_spoiler:
                overlaps.append(0.0)
                continue

            intersection = words_spoiler.intersection(words_article)
            overlap_ratio = len(intersection) / len(words_spoiler)
            overlaps.append(overlap_ratio)

        self.df["spoiler_overlap_ratio"] = overlaps
        print(self.df["spoiler_overlap_ratio"].describe())

        fully_extractive = (self.df["spoiler_overlap_ratio"] == 1.0).sum()
        print(
            f"\nFully Extractive Spoilers (100% word overlap):"
            f" {fully_extractive} out of {len(self.df)} "
            f"({fully_extractive / len(self.df) * 100:.2f}%)"
        )

    def analyze_positional_bias(self) -> None:
        """Extracts and analyzes the paragraph indices where spoilers are

        located to detect positional bias (e.g., Lead Bias).
        """
        print("\n=== Spoiler Positional Bias Analysis ===")

        spoiler_paragraphs = []
        for _, row in self.df.iterrows():
            positions = row.get("spoilerPositions", [])
            if isinstance(positions, list) and len(positions) > 0:
                try:
                    first_segment = positions[0]
                    start_position = first_segment[0]
                    para_index = start_position[0]
                    spoiler_paragraphs.append(para_index)
                except (IndexError, TypeError):
                    continue

        if not spoiler_paragraphs:
            print("[WARNING] No valid spoiler positions found to analyze.")
            return

        pos_series = pd.Series(spoiler_paragraphs)
        print("Paragraph Index Distribution for Spoilers:")
        print(pos_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95]))

        plt.figure(figsize=(8, 5))
        sns.histplot(pos_series, bins=20, kde=False, color="purple")
        plt.title("Distribution of Spoiler Locations (Paragraph Index)")
        plt.xlabel("Paragraph Index")
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "spoiler_positional_bias.png")
        plt.savefig(plot_path)
        print(f"[SUCCESS] Positional bias plot saved to: {plot_path}")
        plt.close()

    def analyze_tag_specific_lengths(self) -> None:
        """Analyzes the word count distribution of the spoilers themselves,

        grouped by their specific tags (phrase, passage, multi).
        """
        print("\n=== Tag-Specific Spoiler Length Analysis ===")

        self.df["spoiler_word_count"] = self.df["spoiler"].apply(self._count_words)
        self.df["clean_tag"] = self.df["tags"].apply(extract_primary_tag)

        grouped_stats = self.df.groupby("clean_tag")["spoiler_word_count"].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95]
        )
        print(grouped_stats)

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="clean_tag",
            y="spoiler_word_count",
            data=self.df,
            palette="Set2",
            hue="clean_tag",
            legend=False,
        )
        plt.title("Spoiler Word Count Distribution per Class")
        plt.xlabel("Spoiler Type (Tag)")
        plt.ylabel("Word Count")
        plt.yscale("log")
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "spoiler_length_by_tag.png")
        plt.savefig(plot_path)
        print(f"[SUCCESS] Tag-specific length plot saved to: {plot_path}")
        plt.close()

    def analyze_oracle_retrieval_bound(self) -> None:
        """Evaluates an informational retrieval upper-bound using Jaccard

        Similarity between the clickbait post and paragraphs, computing Recall@K
        for the ground-truth spoiler paragraph.
        """
        print("\n=== Oracle Retrieval Upper-Bound Analysis (IR Metrics) ===")

        # Oracle metrics logging
        recall_at_1 = 0
        recall_at_3 = 0
        recall_at_5 = 0
        total_valid_cases = 0

        for _, row in self.df.iterrows():
            post_words = set(
                (
                    " ".join(row["postText"])
                    if isinstance(row["postText"], list)
                    else str(row["postText"])
                )
                .lower()
                .split()
            )
            paragraphs = row["targetParagraphs"]
            spoilers = (
                " ".join(row["spoiler"])
                if isinstance(row["spoiler"], list)
                else str(row["spoiler"])
            )

            if not paragraphs or not spoilers:
                continue

            total_valid_cases += 1

            gt_indices = [
                i for i, p in enumerate(paragraphs) if spoilers.lower() in p.lower()
            ]
            if not gt_indices:
                continue
            gt_index = gt_indices[0]

            scored_paragraphs = []
            for idx, para in enumerate(paragraphs):
                para_words = set(para.lower().split())
                intersection = post_words.intersection(para_words)
                union = post_words.union(para_words)
                jaccard_score = len(intersection) / len(union) if union else 0.0
                scored_paragraphs.append((idx, jaccard_score))

            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            ranked_indices = [item[0] for item in scored_paragraphs]

            if gt_index in ranked_indices[:1]:
                recall_at_1 += 1
            if gt_index in ranked_indices[:3]:
                recall_at_3 += 1
            if gt_index in ranked_indices[:5]:
                recall_at_5 += 1

        if total_valid_cases == 0:
            print("[WARNING] No valid cases for Oracle Analysis.")
            return

        print(f"Total Evaluated Cases: {total_valid_cases}")
        print(f"Oracle Recall@1: {recall_at_1 / total_valid_cases * 100:.2f}%")
        print(f"Oracle Recall@3: {recall_at_3 / total_valid_cases * 100:.2f}%")
        print(f"Oracle Recall@5: {recall_at_5 / total_valid_cases * 100:.2f}%")
