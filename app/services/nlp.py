from typing import List, Tuple, Dict, Any
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
    def generate_insight(issues: List[DetectedIssue]) -> Dict[str, Any]:
        """
        Generates a proactive insight summary and a recommended action plan.
        """
        high_sev = [i for i in issues if i.severity == "High"]

        insight_text = ""
        actions = []

        # 1. Generate Narrative (The "Why")
        if not issues:
            insight_text = "Great news! The dataset appears to be clean. No major issues were detected."
        else:
            insight_text = f"I have analyzed your data and found {len(issues)} quality issues. "

            if high_sev:
                insight_text += f"Most critically, there are {len(high_sev)} high-severity issues (like {high_sev[0].type}) that will break downstream analysis or cause calculations to fail. "

            if any(i.type == "Visualization Risk" for i in issues):
                insight_text += "I also noticed high-cardinality columns that will make your charts unreadable. "

            insight_text += "I have prepared a cleaning plan to fix these problems automatically."

        # 2. Generate Auto-Fix Plan (Heuristic based)
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
        """Heuristic for best default action."""
        if issue.type == "Duplicates":
            return "remove_duplicates"

        if issue.type == "Missing Values":
            if issue.row_count / 1000 < 0.05:
                return "drop_rows"
            if "Numeric" in str(issue.description):
                return "fill_median"
            return "fill_mode"

        if issue.type == "Outliers":
            return "clip_outliers"

        if issue.type == "Visualization Risk":
            return "group_rare"

        return "ignore"