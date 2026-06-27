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

    def analyze_class_balance(self) -> pd.Series:
        """Analyzes and prints the distribution of spoiler types (tags).

        Returns:
            pd.Series: Percentage distribution of each class.
        """
        print("\n=== Class Distribution Analysis (Task 1) ===")
        # Since 'tags' might contain lists, we extract the first element if necessary
        # Usually it's a list containing one string like ['phrase']
        class_series = self.df["tags"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

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

            # Jaccard Overlap: Intersection / Spoiler Size
            # To see how much of the spoiler words exist in the article
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
