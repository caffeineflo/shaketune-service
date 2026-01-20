"""
Shake&Tune Analysis Service

A lightweight web service that processes Klipper accelerometer data
and generates analysis graphs for input shaper calibration.

Designed for use with Creality K1 and other Klipper-based printers.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
import subprocess
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


def get_printer_dir(printer: str) -> str:
    """Get or create printer-specific results directory."""
    printer_dir = os.path.join(RESULTS_DIR, printer)
    os.makedirs(printer_dir, exist_ok=True)
    return printer_dir


@app.post("/shaper")
async def analyze_shaper(
    files: List[UploadFile] = File(...),
    printer: Optional[str] = Form(default="default"),
    timestamp: Optional[str] = Form(default=None),
    max_freq: Optional[float] = Form(default=200.0),
    scv: Optional[float] = Form(default=5.0)
):
    """
    Analyze resonance data and generate input shaper calibration graph.

    Upload raw accelerometer CSV files from TEST_RESONANCES commands.
    Optional 'printer' parameter to organize results by printer name.
    Optional 'timestamp' parameter for predictable URLs (format: YYYYMMDD_HHMMSS).
    Returns URL to the generated analysis graph.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    printer_dir = get_printer_dir(printer)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "shaper.png")

        extra_args = ["--max_freq", str(max_freq), "--scv", str(scv)]
        run_graph_cli("input_shaper", csv_paths, output_png, extra_args)

        final_name = f"{ts}_shaper.png"
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{printer}/{final_name}", "type": "shaper", "printer": printer}


@app.post("/belts")
async def analyze_belts(
    files: List[UploadFile] = File(...),
    printer: Optional[str] = Form(default="default"),
    timestamp: Optional[str] = Form(default=None),
    max_freq: Optional[float] = Form(default=200.0)
):
    """
    Analyze belt tension data and generate comparison graph.

    Upload raw accelerometer CSV files from belt resonance tests.
    Optional 'printer' parameter to organize results by printer name.
    Optional 'timestamp' parameter for predictable URLs (format: YYYYMMDD_HHMMSS).
    Returns URL to the generated belt comparison graph.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    printer_dir = get_printer_dir(printer)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "belts.png")

        extra_args = ["--max_freq", str(max_freq)]
        run_graph_cli("belts", csv_paths, output_png, extra_args)

        final_name = f"{ts}_belts.png"
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{printer}/{final_name}", "type": "belts", "printer": printer}


@app.post("/vibrations")
async def analyze_vibrations(
    files: List[UploadFile] = File(...),
    printer: Optional[str] = Form(default="default"),
    timestamp: Optional[str] = Form(default=None),
    kinematics: Optional[str] = Form(default="corexy"),
    max_freq: Optional[float] = Form(default=1000.0)
):
    """
    Analyze vibration data across speeds and generate analysis graph.

    Upload raw accelerometer CSV files from vibration measurements.
    Optional 'printer' parameter to organize results by printer name.
    Optional 'timestamp' parameter for predictable URLs (format: YYYYMMDD_HHMMSS).
    Returns URL to the generated vibration analysis graph.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    printer_dir = get_printer_dir(printer)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, "vibrations.png")

        extra_args = ["--max_freq", str(max_freq), "--kinematics", kinematics]
        run_graph_cli("vibrations", csv_paths, output_png, extra_args)

        final_name = f"{ts}_vibrations.png"
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{printer}/{final_name}", "type": "vibrations", "printer": printer}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "shaketune-service"}


def get_latest_file(graph_type: str, printer: str = "default") -> Optional[str]:
    """Find the most recent graph file of the given type for a printer."""
    try:
        printer_dir = os.path.join(RESULTS_DIR, printer)
        if not os.path.exists(printer_dir):
            return None
        files = [f for f in os.listdir(printer_dir) if f.endswith(f"_{graph_type}.png")]
        if not files:
            return None
        # Files are named with timestamp prefix, so sorting gives chronological order
        files.sort(reverse=True)
        return files[0]
    except Exception:
        return None


@app.get("/latest/{printer}/{graph_type}")
async def latest_graph_for_printer(printer: str, graph_type: str):
    """
    Redirect to the most recent graph of the specified type for a printer.

    Supported types: shaper, belts, vibrations
    """
    if graph_type not in ["shaper", "belts", "vibrations"]:
        raise HTTPException(status_code=400, detail=f"Invalid graph type: {graph_type}")

    latest = get_latest_file(graph_type, printer)
    if not latest:
        raise HTTPException(status_code=404, detail=f"No {graph_type} graphs found for printer '{printer}'")

    return RedirectResponse(url=f"/results/{printer}/{latest}", status_code=302)


@app.get("/latest/{graph_type}")
async def latest_graph(graph_type: str):
    """
    Redirect to the most recent graph of the specified type (default printer).

    Supported types: shaper, belts, vibrations
    """
    if graph_type not in ["shaper", "belts", "vibrations"]:
        raise HTTPException(status_code=400, detail=f"Invalid graph type: {graph_type}")

    latest = get_latest_file(graph_type, "default")
    if not latest:
        raise HTTPException(status_code=404, detail=f"No {graph_type} graphs found")

    return RedirectResponse(url=f"/results/default/{latest}", status_code=302)


@app.get("/")
async def root():
    """API documentation."""
    return {
        "service": "Shake&Tune Analysis Service",
        "version": "1.1.0",
        "description": "Process Klipper accelerometer data for input shaper calibration",
        "endpoints": {
            "POST /shaper": "Upload resonance CSVs, get input shaper graph (optional: printer=name)",
            "POST /belts": "Upload belt test CSVs, get belt comparison graph (optional: printer=name)",
            "POST /vibrations": "Upload vibration CSVs, get speed analysis graph (optional: printer=name)",
            "GET /results/{printer}/{filename}": "Retrieve generated graph",
            "GET /latest/{printer}/{type}": "Redirect to printer's latest graph",
            "GET /latest/{type}": "Redirect to latest graph (default printer)",
            "GET /health": "Health check",
        },
        "usage": {
            "single_printer": 'curl -X POST http://host:3080/shaper -F "files=@x.csv" -F "files=@y.csv"',
            "multi_printer": 'curl -X POST http://host:3080/shaper -F "files=@x.csv" -F "files=@y.csv" -F "printer=k1v3"',
        }
    }
