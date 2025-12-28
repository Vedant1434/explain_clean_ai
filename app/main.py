from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import uvicorn
import pandas as pd
import time
from collections import defaultdict

from app.models import DatasetProfile, BulkFixRequest, NaturalLanguageQuery, CleaningReport
from app.services.ingestion import IngestionService
from app.services.profiler import ProfilerService
from app.services.cleaner import CleanerService
from app.services.nlp import NLPService
from app.store import store

app = FastAPI(title="Explain-Clean Tool")  # Renamed from AI

# 1. CORS Restriction
# In production, replace allow_origins=["*"] with specific domains
# For now, we restrict to localhost or specific development origins if known
# But to be safe "Right Now" as requested, we will just comment out the wildcard and use a safer default or keep it open if you are testing locally from different ports.
# Since you asked to FIX "Open CORS", I will restrict it to common local dev ports.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Add your production frontend domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict methods
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Rate Limiting (Simple In-Memory Implementation)
# In production, use Redis.
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS_PER_MINUTE = 20
request_counts = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()

    # Clean up old requests
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < RATE_LIMIT_DURATION]

    if len(request_counts[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})

    request_counts[client_ip].append(now)
    response = await call_next(request)
    return response


@app.get("/")
async def root():
    return FileResponse('static/index.html')


# 3. File Size & Type Validation
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB limit


@app.post("/api/upload", response_model=DatasetProfile)
async def upload_dataset(file: UploadFile = File(...)):
    # Validate File Type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")

    # Validate File Size (Approximation by reading chunk or seek)
    # Since UploadFile is a SpooledTemporaryFile, we can check size.
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE / 1024 / 1024}MB.")

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
    remaining_issues = ProfilerService.analyze(cleaned_df)
    store.save_issues(session_id, remaining_issues)

    # 4. Return Report
    download_url = f"/api/session/{session_id}/download"

    return CleaningReport(
        rows_before=len(original_df),
        rows_after=len(cleaned_df),
        actions_taken=actions,
        remaining_issues=remaining_issues,
        chart_recommendations=CleanerService.recommend_charts(cleaned_df),
        download_url=download_url
    )


@app.get("/api/session/{session_id}/download")
async def download_session_data(session_id: str):
    """
    Streams the current dataframe state from memory as a CSV download.
    """
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    df = session["current_df"]
    filename = f"clean_{session['filename']}"

    # Create an in-memory buffer
    stream = io.StringIO()
    df.to_csv(stream, index=False)

    # Reset pointer to start of stream
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)