import pandas as pd
from fastapi import UploadFile
import io

MAX_ROWS_FOR_FULL_PROFILE = 100000


class IngestionService:
    @staticmethod
    async def process_upload(file: UploadFile) -> pd.DataFrame:
        content = await file.read()

        # Determine file type and read
        # Simple implementation supports CSV. Extendable for Parquet/Excel.
        try:
            # Try reading full file
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            # Fallback for encoding issues
            df = pd.read_csv(io.BytesIO(content), encoding='latin1')

        # Check size constraint
        if len(df) > MAX_ROWS_FOR_FULL_PROFILE:
            # Random sample for profiling massive datasets
            df = df.sample(n=MAX_ROWS_FOR_FULL_PROFILE, random_state=42)

        return df

    @staticmethod
    def get_preview(df: pd.DataFrame, rows=5):
        # Handle NaN for JSON serialization
        return df.head(rows).replace({float('nan'): None}).to_dict(orient='records')