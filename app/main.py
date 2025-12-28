from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uvicorn

from app.models import DatasetProfile, BulkFixRequest, NaturalLanguageQuery, CleaningReport
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

os.makedirs("temp", exist_ok=True)
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

    # 4. Return Profile with Session ID (Fixes Validation Error)
    return DatasetProfile(
        filename=file.filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        issues=issues,
        sample_data=IngestionService.get_preview(df),
        session_id=session_id  # <--- CRITICAL FIX
    )


@app.get("/api/session/{session_id}/analyze")
async def analyze_session(session_id: str):
    """
    New endpoint for Proactive AI Insights
    """
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    issues_list = list(session["issues"].values())

    # Generate proactive insights
    analysis = NLPService.generate_insight(issues_list)

    return analysis


@app.post("/api/session/{session_id}/query")
async def nlp_query(session_id: str, query: NaturalLanguageQuery):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    issues_dict = session["issues"]
    issues_list = list(issues_dict.values())

    suggested_actions = NLPService.interpret_command(query.query, issues_list)

    return {
        "applied_to": [
            {"issue_id": sid, "strategy_code": code}
            for sid, code in suggested_actions
        ],
        "message": f"Found {len(suggested_actions)} actions for your request."
    }


@app.post("/api/session/{session_id}/clean", response_model=CleaningReport)
async def clean_dataset(session_id: str, request: BulkFixRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    original_df = session["current_df"]

    cleaned_df, actions = CleanerService.apply_fixes(
        original_df, request.fixes, session["issues"]
    )

    store.update_dataframe(session_id, cleaned_df)
    for action in actions:
        store.log_action(session_id, action)

    output_filename = f"clean_{session['filename']}"
    output_path = os.path.join("temp", output_filename)
    cleaned_df.to_csv(output_path, index=False)

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