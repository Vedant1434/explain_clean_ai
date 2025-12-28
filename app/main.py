from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import uvicorn

from app.models import DatasetProfile, BulkFixRequest, NaturalLanguageQuery, CleaningReport, FixRequest
from app.services.ingestion import IngestionService
from app.services.profiler import ProfilerService
from app.services.cleaner import CleanerService
from app.services.nlp import NLPService
from app.store import store

app = FastAPI(title="Explain-Clean AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Mount static files for Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse('static/index.html')


@app.post("/api/upload", response_model=DatasetProfile)
async def upload_dataset(file: UploadFile = File(...)):
    # 1. Ingest
    df = await IngestionService.process_upload(file)

    # 2. Create Session
    session_id = store.create_session(df, file.filename)

    # 3. Profile
    issues = ProfilerService.analyze(df)
    store.save_issues(session_id, issues)

    return DatasetProfile(
        filename=file.filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        issues=issues,
        sample_data=IngestionService.get_preview(df),
        # Abuse the 'filename' field to pass back session_id in this MVP structure
        # In a strict REST API, we'd return a Session object, but let's append it to filename for simplicity
        # or better, use a custom header. Let's use a custom header in response.
    )


@app.post("/api/session/{session_id}/query")
async def nlp_query(session_id: str, query: NaturalLanguageQuery):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    issues_dict = session["issues"]
    issues_list = list(issues_dict.values())

    # Get suggested actions
    suggested_actions = NLPService.interpret_command(query.query, issues_list)

    # Convert to response format
    # We return the list of issue IDs and the suggested strategy code
    return {
        "applied_to": [
            {"issue_id": sid, "strategy_code": code}
            for sid, code in suggested_actions
        ],
        "message": f"Found {len(suggested_actions)} actions for your request."
    }


@app.get("/api/session/{session_id}/analyze")
async def analyze_session(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    issues_list = list(session["issues"].values())

    # Generate proactive insights
    analysis = NLPService.generate_insight(issues_list)

    return analysis


@app.post("/api/session/{session_id}/clean", response_model=CleaningReport)
async def clean_dataset(session_id: str, request: BulkFixRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    original_df = session["current_df"]

    # Apply Fixes
    cleaned_df, actions = CleanerService.apply_fixes(
        original_df, request.fixes, session["issues"]
    )

    # Update Session
    store.update_dataframe(session_id, cleaned_df)
    for action in actions:
        store.log_action(session_id, action)

    # Save to Temp File for Download
    output_filename = f"clean_{session['filename']}"
    output_path = os.path.join("temp", output_filename)
    cleaned_df.to_csv(output_path, index=False)

    # Recommendations
    recs = CleanerService.recommend_charts(cleaned_df)

    return CleaningReport(
        rows_before=len(original_df),
        rows_after=len(cleaned_df),
        actions_taken=actions,
        chart_recommendations=recs,
        download_url=f"/api/download/{output_filename}"
    )


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("temp", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(404, "File not found")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)