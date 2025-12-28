from typing import Dict, Any
import pandas as pd
import uuid

# In-memory storage for demonstration purposes.
# In production, replace this with Redis or a database.
class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, df: pd.DataFrame, filename: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "original_df": df,
            "current_df": df.copy(),
            "filename": filename,
            "issues": {},  # Map issue_id to issue object
            "audit_log": []
        }
        return session_id

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    def update_dataframe(self, session_id: str, new_df: pd.DataFrame):
        if session_id in self._sessions:
            self._sessions[session_id]["current_df"] = new_df

    def save_issues(self, session_id: str, issues: list):
        if session_id in self._sessions:
            # Index by ID for quick lookup during resolution
            self._sessions[session_id]["issues"] = {i.id: i for i in issues}

    def log_action(self, session_id: str, action: str):
        if session_id in self._sessions:
            self._sessions[session_id]["audit_log"].append(action)

# Global instance
store = SessionStore()