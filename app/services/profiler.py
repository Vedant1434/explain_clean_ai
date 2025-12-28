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
                    ResolutionStrategy(name="Fill with Mode", description="Replace with most frequent value",
                                       action_code="fill_mode"),
                    ResolutionStrategy(name="Ignore", description="Keep data as is", action_code="ignore")
                ]

                if pd.api.types.is_numeric_dtype(df[col]):
                    strategies.insert(1, ResolutionStrategy(name="Fill with Mean", description="Replace with average",
                                                            action_code="fill_mean"))
                    strategies.insert(2, ResolutionStrategy(name="Fill with Median", description="Replace with median",
                                                            action_code="fill_median"))
                else:
                    strategies.insert(1, ResolutionStrategy(name="Fill 'Unknown'",
                                                            description="Replace with 'Unknown' string",
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

        # 3. Numeric Outliers (IQR Method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Standard outlier definition: 1.5 * IQR
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

            if outliers > 0:
                pct = (outliers / len(df)) * 100
                # Only flag if significant but not overwhelming
                if 0 < pct < 10:
                    issues.append(DetectedIssue(
                        id=str(uuid.uuid4()),
                        type=IssueType.OUTLIERS,
                        column=col,
                        description=f"Column '{col}' has {outliers} potential outliers.",
                        severity=Severity.MEDIUM,
                        impact="Outliers skew averages and distort axis scaling in charts.",
                        row_count=int(outliers),
                        strategies=[
                            ResolutionStrategy(name="Clip Values", description="Cap values at min/max thresholds",
                                               action_code="clip_outliers"),
                            ResolutionStrategy(name="Remove Rows", description="Delete rows with outliers",
                                               action_code="drop_outliers"),
                            ResolutionStrategy(name="Ignore", description="Keep actual values", action_code="ignore")
                        ]
                    ))

        # 4. High Cardinality (Visualization Risk)
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            unique_count = df[col].nunique()
            if unique_count > 50 and unique_count < len(df):
                issues.append(DetectedIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.VISUALIZATION_RISK,
                    column=col,
                    description=f"Column '{col}' has {unique_count} unique values (High Cardinality).",
                    severity=Severity.LOW,
                    impact="Cannot be used effectively in Bar or Pie charts. Will clutter visualization.",
                    row_count=len(df),
                    strategies=[
                        ResolutionStrategy(name="Group Rare Labels", description="Group infrequent values into 'Other'",
                                           action_code="group_rare"),
                        ResolutionStrategy(name="Ignore", description="Keep all categories", action_code="ignore")
                    ]
                ))

        return issues