import pandas as pd
import numpy as np
from app.models import CleaningReport, DetectedIssue


class CleanerService:
    @staticmethod
    def apply_fixes(df: pd.DataFrame, fixes: list, issues_map: dict) -> tuple[pd.DataFrame, list]:
        audit_log = []
        df_clean = df.copy()

        for fix in fixes:
            issue_id = fix.issue_id
            code = fix.strategy_code

            if code == "ignore":
                continue

            issue: DetectedIssue = issues_map.get(issue_id)
            if not issue:
                continue

            col = issue.column

            # Missing Values
            if code == "drop_rows":
                if col:
                    df_clean = df_clean.dropna(subset=[col])
                    audit_log.append(f"Dropped rows with missing values in '{col}'")
            elif code == "fill_mean":
                val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(val)
                audit_log.append(f"Filled missing '{col}' with mean ({val:.2f})")
            elif code == "fill_median":
                val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(val)
                audit_log.append(f"Filled missing '{col}' with median ({val:.2f})")
            elif code == "fill_mode":
                val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(val)
                audit_log.append(f"Filled missing '{col}' with mode ({val})")
            elif code == "fill_unknown":
                df_clean[col] = df_clean[col].fillna("Unknown")
                audit_log.append(f"Filled missing '{col}' with 'Unknown'")
            elif code == "ffill":
                df_clean[col] = df_clean[col].ffill()
                audit_log.append(f"Forward-filled missing values in '{col}'")

            # Duplicates
            elif code == "remove_duplicates":
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                audit_log.append(f"Removed {before - len(df_clean)} duplicate rows")

            # Outliers
            elif code == "clip_outliers":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean[col] = np.clip(df_clean[col], lower, upper)
                audit_log.append(f"Clipped outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")
            elif code == "drop_outliers":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~((df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR)))
                df_clean = df_clean[mask]
                audit_log.append(f"Dropped outlier rows in '{col}'")

            # Type & Text Issues
            elif code == "convert_numeric":
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                audit_log.append(f"Converted '{col}' to Numeric (invalid values set to NaN)")

            elif code == "title_case":
                df_clean[col] = df_clean[col].astype(str).str.title()
                audit_log.append(f"Standardized '{col}' to Title Case")

            elif code == "lower_case":
                df_clean[col] = df_clean[col].astype(str).str.lower()
                audit_log.append(f"Standardized '{col}' to Lower Case")

            # Viz Risks
            elif code == "group_rare":
                top_10 = df_clean[col].value_counts().nlargest(10).index
                df_clean[col] = df_clean[col].apply(lambda x: x if x in top_10 else "Other")
                audit_log.append(f"Grouped rare categories in '{col}' into 'Other'")

        return df_clean, audit_log

    @staticmethod
    def recommend_charts(df: pd.DataFrame) -> list[str]:
        recommendations = []
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        if len(num_cols) >= 2:
            recommendations.append(f"Scatter Plot: {num_cols[0]} vs {num_cols[1]}")
            recommendations.append(f"Correlation Heatmap: {', '.join(num_cols[:3])}")
        if len(cat_cols) > 0 and len(num_cols) > 0:
            recommendations.append(f"Bar Chart: {cat_cols[0]} vs {num_cols[0]} (Avg)")
        if len(date_cols) > 0 and len(num_cols) > 0:
            recommendations.append(f"Line Chart (Trend): {num_cols[0]} over Time")
        if not recommendations:
            recommendations.append("Data table view (Insufficient columns for advanced charts)")
        return recommendations