from typing import List, Tuple
from app.models import DetectedIssue


class NLPService:
    @staticmethod
    def interpret_command(command: str, issues: List[DetectedIssue]) -> List[Tuple[str, str]]:
        """
        Returns a list of (issue_id, strategy_code) based on natural language.
        """
        command = command.lower()
        actions = []

        # Intent: Fix High Severity
        if "high" in command or "critical" in command or "severe" in command:
            for issue in issues:
                if issue.severity == "High":
                    # Determine best default strategy
                    strategy = NLPService._get_default_strategy(issue)
                    if strategy:
                        actions.append((issue.id, strategy))

        # Intent: Fix Missing Values
        elif "missing" in command or "null" in command or "empty" in command:
            for issue in issues:
                if issue.type == "Missing Values":
                    strategy = NLPService._get_default_strategy(issue)
                    if strategy:
                        actions.append((issue.id, strategy))

        # Intent: Fix Visualization risks
        elif "chart" in command or "viz" in command or "bar" in command:
            for issue in issues:
                if issue.type == "Visualization Risk":
                    strategy = "group_rare"
                    actions.append((issue.id, strategy))

        # Intent: Fix Everything
        elif "all" in command or "everything" in command:
            for issue in issues:
                strategy = NLPService._get_default_strategy(issue)
                if strategy:
                    actions.append((issue.id, strategy))

        return actions

    @staticmethod
    def _get_default_strategy(issue: DetectedIssue) -> str:
        """Heuristic for best default action."""
        if issue.type == "Duplicates":
            return "remove_duplicates"

        if issue.type == "Missing Values":
            # If few rows, drop. If many, fill.
            if issue.row_count / 1000 < 0.05:  # Arbitrary threshold for context
                return "drop_rows"
            if "Numeric" in str(issue.description):  # Simplified check
                return "fill_median"
            return "fill_mode"

        if issue.type == "Outliers":
            return "clip_outliers"

        if issue.type == "Visualization Risk":
            return "group_rare"

        return "ignore"