import os
from typing import Optional

import pandas as pd


class JSONLLoader:
    """A professional loader for handling JSON Lines (.jsonl) datasets

    ensuring structural integrity and providing baseline data validation.
    """

    def __init__(self, file_path: str):
        """Initializes the loader with a specific dataset path.

        Args:
            file_path (str): Path to the .jsonl file.
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads the JSONL file into a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found at target path: {self.file_path}"
            )

        print(f"Reading data from {self.file_path}...")
        self.df = pd.read_json(self.file_path, lines=True)
        return self.df

    def run_sanity_checks(self) -> None:
        """Performs structural integrity and missing value checks on the

        dataset.
        """
        if self.df is None:
            raise ValueError(
                "Data is not loaded. Call load_data() before running sanity checks."
            )

        print("\n=== Running Dataset Sanity Checks ===")
        print(f"Total Rows: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")

        # Check for mandatory fields in Clickbait Spoiling challenge
        mandatory_fields = ["uuid", "postText", "targetParagraphs", "tags"]
        missing_fields = [
            field for field in mandatory_fields if field not in self.df.columns
        ]

        if missing_fields:
            print(f"[WARNING] Missing essential columns: {missing_fields}")
        else:
            print("[SUCCESS] All essential columns are present.")

        # Check for null values in critical columns
        null_counts = self.df[mandatory_fields].isnull().sum()
        print("\nMissing Values Count per Critical Column:")
        print(null_counts)
