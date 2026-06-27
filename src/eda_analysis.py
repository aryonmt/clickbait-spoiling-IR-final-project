import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TextLengthAnalyzer:
    """Analyzer to calculate sequence lengths and statistical percentiles

    for text features in the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """Initializes the analyzer with a dataset.

        Args:
            df (pd.DataFrame): The target dataset.
        """
        self.df = df.copy()

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

    def plot_distributions(self, output_dir: str = ".") -> None:
        """Generates and saves distribution plots for sequence lengths.

        Args:
            output_dir (str): Directory where the plot image will be saved.
        """
        plt.figure(figsize=(12, 5))

        # Plot for Clickbait Post Lengths
        plt.subplot(1, 2, 1)
        sns.histplot(self.df["post_word_count"], kde=True, color="skyblue")
        plt.title("Distribution of Clickbait Post Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        # Plot for Article Content Lengths
        plt.subplot(1, 2, 2)
        sns.histplot(self.df["article_word_count"], kde=True, color="salmon")
        plt.title("Distribution of Article Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plot_path = f"{output_dir}/sequence_length_distributions.png"
        plt.savefig(plot_path)
        print(f"\n[SUCCESS] Distribution plots saved to: {plot_path}")
        plt.close()
