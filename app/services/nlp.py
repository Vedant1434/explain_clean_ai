from typing import List, Tuple, Dict, Any
from app.models import DetectedIssue, IssueType


class NLPService:
    @staticmethod
    def interpret_command(command: str, issues: List[DetectedIssue]) -> List[Tuple[str, str]]:
        command = command.lower()
        actions = []

        if "high" in command or "critical" in command or "severe" in command:
            for issue in issues:
                if issue.severity == "High":
                    strategy = NLPService._get_default_strategy(issue)
                    if strategy:
                        actions.append((issue.id, strategy))

        elif "missing" in command or "null" in command or "empty" in command:
            for issue in issues:
                if issue.type == IssueType.MISSING_VALUES:
                    strategy = NLPService._get_default_strategy(issue)
                    if strategy:
                        actions.append((issue.id, strategy))

        elif "text" in command or "case" in command:
            for issue in issues:
                if issue.type == IssueType.TEXT_INCONSISTENCY:
                    actions.append((issue.id, "title_case"))

        elif "type" in command or "number" in command:
            for issue in issues:
                if issue.type == IssueType.INCONSISTENT_TYPE:
                    actions.append((issue.id, "convert_numeric"))

        elif "all" in command or "everything" in command:
            for issue in issues:
                strategy = NLPService._get_default_strategy(issue)
                if strategy:
                    actions.append((issue.id, strategy))

        return actions

    @staticmethod
    def generate_insight(issues: List[DetectedIssue]) -> Dict[str, Any]:
        high_sev = [i for i in issues if i.severity == "High"]

        insight_text = ""
        actions = []

        if not issues:
            insight_text = "Great news! The dataset appears to be clean. No major issues were detected."
        else:
            insight_text = f"I have analyzed your data and found {len(issues)} quality issues. "

            if high_sev:
                insight_text += f"Most critically, there are {len(high_sev)} high-severity issues. "
                if any(i.type == IssueType.INCONSISTENT_TYPE for i in high_sev):
                    insight_text += "Some columns look like numbers but are stored as text. "
                if any(i.type == IssueType.DUPLICATES for i in high_sev):
                    insight_text += "There are also duplicate rows found. "

            insight_text += "I've created a custom cleaning plan to standardize these values."

        for issue in issues:
            strategy = NLPService._get_default_strategy(issue)
            if strategy and strategy != "ignore":
                actions.append({"issue_id": issue.id, "strategy_code": strategy})

        return {
            "insight": insight_text,
            "recommended_actions": actions,
            "action_count": len(actions)
        }

    @staticmethod
    def _get_default_strategy(issue: DetectedIssue) -> str:
        """
        Smart Heuristics for Default Actions.
        This is the 'Brain' of the AI recommendation system.
        """
        if issue.type == IssueType.DUPLICATES:
            return "remove_duplicates"

        if issue.type == IssueType.MISSING_VALUES:
            # If it's a very small amount of missing data, dropping is safest.
            if issue.row_count < 20 or issue.row_count / 1000 < 0.05:
                return "drop_rows"
            # If the column name suggests a sequence (like time), use ffill
            if issue.column and ("date" in issue.column.lower() or "time" in issue.column.lower()):
                return "ffill"
            if "Numeric" in str(issue.description):
                return "fill_median"  # Median is safer than mean for un-reviewed data
            return "fill_mode"

        if issue.type == IssueType.OUTLIERS:
            # We already skipped IDs in the profiler, so clipping is generally safe for measurements
            return "clip_outliers"

        if issue.type == IssueType.VISUALIZATION_RISK:
            return "group_rare"

        if issue.type == IssueType.INCONSISTENT_TYPE:
            return "convert_numeric"

        if issue.type == IssueType.TEXT_INCONSISTENCY:
            return "title_case"  # Safest default for names/categories

        return "ignore"