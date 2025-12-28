from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class Severity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class IssueType(str, Enum):
    MISSING_VALUES = "Missing Values"
    DUPLICATES = "Duplicates"
    OUTLIERS = "Outliers"
    INCONSISTENT_TYPE = "Inconsistent Type" # e.g. Numbers as Strings
    TEXT_INCONSISTENCY = "Text Inconsistency" # e.g. "north" vs "North"
    VISUALIZATION_RISK = "Visualization Risk"

class ResolutionStrategy(BaseModel):
    name: str
    description: str
    action_code: str

class DetectedIssue(BaseModel):
    id: str
    type: IssueType
    column: Optional[str] = None
    description: str
    severity: Severity
    impact: str
    row_count: int
    strategies: List[ResolutionStrategy]

class DatasetProfile(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns: List[str]
    issues: List[DetectedIssue]
    sample_data: List[Dict[str, Any]]
    session_id: str

class FixRequest(BaseModel):
    issue_id: str
    strategy_code: str

class BulkFixRequest(BaseModel):
    fixes: List[FixRequest]

class NaturalLanguageQuery(BaseModel):
    query: str

class CleaningReport(BaseModel):
    rows_before: int
    rows_after: int
    actions_taken: List[str]
    chart_recommendations: List[str]
    download_url: str