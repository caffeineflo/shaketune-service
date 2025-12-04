"""
Shake&Tune Analysis Service

A lightweight web service that processes Klipper accelerometer data
and generates analysis graphs for input shaper calibration.

Designed for use with Creality K1 and other Klipper-based printers.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import subprocess
import uuid
import os
import shutil
import tempfile
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="Shake&Tune Analysis Service",
    description="Process Klipper accelerometer data and generate calibration graphs",
    version="1.0.0"
)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/app/results")
KLIPPER_DIR = os.environ.get("KLIPPER_DIR", "/app/service/klipper")
SHAKETUNE_DIR = os.environ.get("SHAKETUNE_DIR", "/app/shaketune/graph_creators")

os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


def run_graph_cli(graph_type: str, csv_paths: List[str], output_path: str, extra_args: List[str] = None) -> bool:
    """Run the Shake&Tune CLI to generate a graph."""
    cmd = [
        "python", "-m", "shaketune.cli",
        graph_type,
        "-k", KLIPPER_DIR,
        "-o", output_path,
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(csv_paths)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/app")
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {result.stderr or result.stdout}"
        )
    return True


async def save_uploaded_files(files: List[UploadFile], tmpdir: str) -> List[str]:
    """Save uploaded files to temp directory and return paths."""
    csv_paths = []
    for f in files:
        path = os.path.join(tmpdir, f.filename)
        with open(path, "wb") as out:
            content = await f.read()
            out.write(content)
        csv_paths.append(path)
    return csv_paths


@app.post("/shaper")
async def analyze_shaper(
    files: List[UploadFile] = File(...),
    max_freq: Optional[float] = 200.0,
    scv: Optional[float] = 5.0
):
    """
    Analyze resonance data and generate input shaper calibration graph.

    Upload raw accelerometer CSV files from TEST_RESONANCES commands.
    Returns URL to the generated analysis graph.
    """
    job_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "shaper.png")

        extra_args = ["--max_freq", str(max_freq), "--scv", str(scv)]
        run_graph_cli("input_shaper", csv_paths, output_png, extra_args)

        final_name = f"{timestamp}_{job_id}_shaper.png"
        final_path = os.path.join(RESULTS_DIR, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{final_name}", "id": job_id, "type": "shaper"}


@app.post("/belts")
async def analyze_belts(
    files: List[UploadFile] = File(...),
    max_freq: Optional[float] = 200.0
):
    """
    Analyze belt tension data and generate comparison graph.

    Upload raw accelerometer CSV files from belt resonance tests.
    Returns URL to the generated belt comparison graph.
    """
    job_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "belts.png")

        extra_args = ["--max_freq", str(max_freq)]
        run_graph_cli("belts", csv_paths, output_png, extra_args)

        final_name = f"{timestamp}_{job_id}_belts.png"
        final_path = os.path.join(RESULTS_DIR, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{final_name}", "id": job_id, "type": "belts"}


@app.post("/vibrations")
async def analyze_vibrations(
    files: List[UploadFile] = File(...),
    kinematics: Optional[str] = "corexy",
    max_freq: Optional[float] = 1000.0
):
    """
    Analyze vibration data across speeds and generate analysis graph.

    Upload raw accelerometer CSV files from vibration measurements.
    Returns URL to the generated vibration analysis graph.
    """
    job_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "vibrations.png")

        extra_args = ["--max_freq", str(max_freq), "--kinematics", kinematics]
        run_graph_cli("vibrations", csv_paths, output_png, extra_args)

        final_name = f"{timestamp}_{job_id}_vibrations.png"
        final_path = os.path.join(RESULTS_DIR, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{final_name}", "id": job_id, "type": "vibrations"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "shaketune-service"}


@app.get("/")
async def root():
    """API documentation."""
    return {
        "service": "Shake&Tune Analysis Service",
        "version": "1.0.0",
        "description": "Process Klipper accelerometer data for input shaper calibration",
        "endpoints": {
            "POST /shaper": "Upload resonance CSVs, get input shaper calibration graph",
            "POST /belts": "Upload belt test CSVs, get belt comparison graph",
            "POST /vibrations": "Upload vibration CSVs, get speed analysis graph",
            "GET /results/{filename}": "Retrieve generated graph",
            "GET /health": "Health check",
        },
        "usage": {
            "example": 'curl -X POST http://localhost:8080/shaper -F "files=@resonance_x.csv" -F "files=@resonance_y.csv"',
        }
    }
