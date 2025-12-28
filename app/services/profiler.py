import pandas as pd
import numpy as np
import uuid
from app.models import DetectedIssue, IssueType, Severity, ResolutionStrategy


class ProfilerService:
    @staticmethod
    def analyze(df: pd.DataFrame) -> list[DetectedIssue]:
        issues = []

        # 1. Missing Values
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(df)) * 100
                severity = Severity.HIGH if pct > 20 else Severity.MEDIUM

                strategies = [
                    ResolutionStrategy(name="Drop Rows", description="Remove rows with missing values",
                                       action_code="drop_rows"),
                    ResolutionStrategy(name="Ignore", description="Keep data as is", action_code="ignore")
                ]

                # Context-Aware Strategy Injection
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategies.insert(1, ResolutionStrategy(name="Fill with Median",
                                                            description="Robust fill for skewed data",
                                                            action_code="fill_median"))
                    strategies.insert(2, ResolutionStrategy(name="Fill with Mean",
                                                            description="Standard fill for normal data",
                                                            action_code="fill_mean"))
                    strategies.insert(3, ResolutionStrategy(name="Forward Fill",
                                                            description="Propagate last valid observation",
                                                            action_code="ffill"))
                else:
                    strategies.insert(1, ResolutionStrategy(name="Fill with Mode",
                                                            description="Replace with most frequent value",
                                                            action_code="fill_mode"))
                    strategies.insert(2, ResolutionStrategy(name="Fill 'Unknown'",
                                                            description="Explicitly label as Unknown",
                                                            action_code="fill_unknown"))

                issues.append(DetectedIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.MISSING_VALUES,
                    column=col,
                    description=f"Column '{col}' has {count} missing values ({pct:.1f}%).",
                    severity=severity,
                    impact="Missing data causes errors in aggregation and voids chart rendering.",
                    row_count=int(count),
                    strategies=strategies
                ))

        # 2. Duplicates
        dupes = df.duplicated().sum()
        if dupes > 0:
            issues.append(DetectedIssue(
                id=str(uuid.uuid4()),
                type=IssueType.DUPLICATES,
                column=None,
                description=f"Dataset contains {dupes} exact duplicate rows.",
                severity=Severity.HIGH,
                impact="Duplicates artificially inflate counts and bias statistical models.",
                row_count=int(dupes),
                strategies=[
                    ResolutionStrategy(name="Remove Duplicates", description="Keep only the first occurrence",
                                       action_code="remove_duplicates"),
                    ResolutionStrategy(name="Ignore", description="Keep duplicates", action_code="ignore")
                ]
            ))

        # 3. Numeric Outliers (Skip IDs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Smart Skip: Don't check ID columns for outliers
            if "id" in col.lower() or "key" in col.lower() or "code" in col.lower():
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

            if outliers > 0:
                pct = (outliers / len(df)) * 100
                if 0 < pct < 15:  # Only flag if it's a small percentage (true outliers)
                    issues.append(DetectedIssue(
                        id=str(uuid.uuid4()),
                        type=IssueType.OUTLIERS,
                        column=col,
                        description=f"Column '{col}' has {outliers} outliers.",
                        severity=Severity.MEDIUM,
                        impact="Outliers skew averages and distort visualizations.",
                        row_count=int(outliers),
                        strategies=[
                            ResolutionStrategy(name="Clip Values", description="Cap values at min/max thresholds",
                                               action_code="clip_outliers"),
                            ResolutionStrategy(name="Remove Rows", description="Delete rows with outliers",
                                               action_code="drop_outliers"),
                            ResolutionStrategy(name="Ignore", description="Keep actual values", action_code="ignore")
                        ]
                    ))

        # 4. Inconsistent Types (Numbers as Strings)
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Try to force convert to numeric
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            num_valid = numeric_conversion.notna().sum()

            # If >80% are numbers but the column is Object, it's a dirty numeric column
            if num_valid > 0.8 * len(df) and num_valid < len(df):
                issues.append(DetectedIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.INCONSISTENT_TYPE,
                    column=col,
                    description=f"Column '{col}' looks numeric but contains text/garbage.",
                    severity=Severity.HIGH,
                    impact="Prevents mathematical operations and sorting.",
                    row_count=len(df),
                    strategies=[
                        ResolutionStrategy(name="Convert to Numeric", description="Force conversion (text becomes NaN)",
                                           action_code="convert_numeric"),
                        ResolutionStrategy(name="Ignore", description="Keep as text", action_code="ignore")
                    ]
                ))

        # 5. Text Inconsistency (Case sensitivity)
        for col in object_cols:
            if df[col].nunique() < 50:  # Only check low cardinality columns
                # Count unique values
                unique_vals = df[col].dropna().unique()
                # Count unique values if we lower-case everything
                unique_lower = set(x.lower() for x in unique_vals if isinstance(x, str))

                # If lowering case reduces unique count, we have inconsistencies (e.g. "North" vs "north")
                if len(unique_lower) < len(unique_vals):
                    issues.append(DetectedIssue(
                        id=str(uuid.uuid4()),
                        type=IssueType.TEXT_INCONSISTENCY,
                        column=col,
                        description=f"Column '{col}' has inconsistent text casing (e.g., 'A' vs 'a').",
                        severity=Severity.LOW,
                        impact="Splits identical categories into separate groups in charts.",
                        row_count=len(df),
                        strategies=[
                            ResolutionStrategy(name="Standardize (Title Case)", description="Convert to 'Title Case'",
                                               action_code="title_case"),
                            ResolutionStrategy(name="Standardize (Lower Case)", description="Convert to 'lower case'",
                                               action_code="lower_case"),
                            ResolutionStrategy(name="Ignore", description="Keep as is", action_code="ignore")
                        ]
                    ))

        return issues