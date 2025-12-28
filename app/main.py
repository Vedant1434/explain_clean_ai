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
    df = await IngestionService.process_upload(file)
    session_id = store.create_session(df, file.filename)
    issues = ProfilerService.analyze(df)
    store.save_issues(session_id, issues)

    return DatasetProfile(
        filename=file.filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        issues=issues,
        sample_data=IngestionService.get_preview(df),
        session_id=session_id
    )


@app.get("/api/session/{session_id}/analyze")
async def analyze_session(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    issues_list = list(session["issues"].values())
    return NLPService.generate_insight(issues_list)


@app.post("/api/session/{session_id}/clean", response_model=CleaningReport)
async def clean_dataset(session_id: str, request: BulkFixRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Apply Fixes
    original_df = session["current_df"]
    cleaned_df, actions = CleanerService.apply_fixes(
        original_df, request.fixes, session["issues"]
    )

    # 2. Update Session
    store.update_dataframe(session_id, cleaned_df)
    for action in actions:
        store.log_action(session_id, action)

    # 3. RE-PROFILE (Iterative Logic)
    # Scan the now cleaned data for any remaining or new issues
    remaining_issues = ProfilerService.analyze(cleaned_df)
    store.save_issues(session_id, remaining_issues)

    # 4. Save to Disk (Always save latest version)
    output_filename = f"clean_{session['filename']}"
    output_path = os.path.join("temp", output_filename)
    cleaned_df.to_csv(output_path, index=False)

    return CleaningReport(
        rows_before=len(original_df),
        rows_after=len(cleaned_df),
        actions_taken=actions,
        remaining_issues=remaining_issues,  # Pass back to frontend
        chart_recommendations=CleanerService.recommend_charts(cleaned_df),
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