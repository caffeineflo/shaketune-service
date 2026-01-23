"""
Shake&Tune Analysis Service

A lightweight web service that processes Klipper accelerometer data
and generates analysis graphs for input shaper calibration.

Designed for use with Creality K1 and other Klipper-based printers.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess
import os
import shutil
import tempfile
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

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

# Templates setup
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_all_results() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Scan results directory and return structured data for all printers."""
    results = {}

    if not os.path.exists(RESULTS_DIR):
        return results

    # Pattern to match result files: YYYYMMDD_HHMMSS_type.png
    pattern = re.compile(r'^(\d{8}_\d{6})_(shaper|belts|vibrations)\.png$')

    for printer_name in os.listdir(RESULTS_DIR):
        printer_path = os.path.join(RESULTS_DIR, printer_name)
        if not os.path.isdir(printer_path):
            continue

        printer_data = {"shaper": [], "belts": [], "vibrations": []}

        for filename in os.listdir(printer_path):
            match = pattern.match(filename)
            if match:
                ts_str, graph_type = match.groups()
                # Parse timestamp for display
                try:
                    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    formatted_ts = ts_str

                printer_data[graph_type].append({
                    "file": filename,
                    "timestamp": formatted_ts,
                    "sort_key": ts_str
                })

        # Sort each type by timestamp (newest first)
        for graph_type in printer_data:
            printer_data[graph_type].sort(key=lambda x: x["sort_key"], reverse=True)

        # Calculate last activity
        all_results = printer_data["shaper"] + printer_data["belts"] + printer_data["vibrations"]
        if all_results:
            latest = max(all_results, key=lambda x: x["sort_key"])
            printer_data["last_activity"] = latest["timestamp"]
        else:
            printer_data["last_activity"] = None

        # Only include printer if it has any results
        if any(printer_data[t] for t in ["shaper", "belts", "vibrations"]):
            results[printer_name] = printer_data

    return results


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
    """Save uploaded files to temp directory and return paths.

    Automatically decompresses .gz files.
    """
    import gzip as gzip_module
    csv_paths = []
    print(f"DEBUG: Received {len(files)} files")
    for f in files:
        print(f"DEBUG: Processing file: {f.filename}, size: {f.size}")
        content = await f.read()

        # Handle gzipped files
        filename = f.filename
        if filename.endswith('.gz'):
            print(f"DEBUG: Decompressing gzipped file: {filename}")
            content = gzip_module.decompress(content)
            filename = filename[:-3]  # Remove .gz extension

        path = os.path.join(tmpdir, filename)
        with open(path, "wb") as out:
            out.write(content)
        csv_paths.append(path)
    print(f"DEBUG: Saved files: {csv_paths}")
    return csv_paths


def get_printer_dir(printer: str) -> str:
    """Get or create printer-specific results directory."""
    printer_dir = os.path.join(RESULTS_DIR, printer)
    os.makedirs(printer_dir, exist_ok=True)
    return printer_dir


@app.post("/shaper")
async def analyze_shaper(
    files: Optional[List[UploadFile]] = File(default=None),
    file_x: Optional[UploadFile] = File(default=None),
    file_y: Optional[UploadFile] = File(default=None),
    printer: Optional[str] = Form(default="default"),
    timestamp: Optional[str] = Form(default=None),
    max_freq: Optional[float] = Form(default=200.0),
    scv: Optional[float] = Form(default=5.0)
):
    """
    Analyze resonance data and generate input shaper calibration graph.

    Upload raw accelerometer CSV files from TEST_RESONANCES commands.
    Accepts either:
    - files: multiple files with same field name (standard curl)
    - file_x and file_y: separate fields (for BusyBox curl compatibility)
    Optional 'printer' parameter to organize results by printer name.
    Optional 'timestamp' parameter for predictable URLs (format: YYYYMMDD_HHMMSS).
    Returns URL to the generated analysis graph.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    printer_dir = get_printer_dir(printer)

    # Build file list from either 'files' array or individual file_x/file_y params
    upload_files = []
    if files:
        upload_files.extend(files)
    if file_x:
        upload_files.append(file_x)
    if file_y:
        upload_files.append(file_y)

    if len(upload_files) < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Shaper analysis requires at least 1 file. Received {len(upload_files)} file(s)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(upload_files, tmpdir)
        output_png = os.path.join(tmpdir, "shaper.png")

        extra_args = ["--max_freq", str(max_freq), "--scv", str(scv)]
        run_graph_cli("input_shaper", csv_paths, output_png, extra_args)

        final_name = f"{ts}_shaper.png"
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {"url": f"/results/{printer}/{final_name}", "type": "shaper", "printer": printer}


@app.post("/belts")
async def analyze_belts(
    files: Optional[List[UploadFile]] = File(default=None),
    file_a: Optional[UploadFile] = File(default=None),
    file_b: Optional[UploadFile] = File(default=None),
    printer: Optional[str] = Form(default="default"),
    timestamp: Optional[str] = Form(default=None),
    max_freq: Optional[float] = Form(default=200.0)
):
    """
    Analyze belt tension data and generate comparison graph.

    Upload raw accelerometer CSV files from belt resonance tests.
    Accepts either:
    - files: multiple files with same field name (standard curl)
    - file_a and file_b: separate fields (for BusyBox curl compatibility)
    Optional 'printer' parameter to organize results by printer name.
    Optional 'timestamp' parameter for predictable URLs (format: YYYYMMDD_HHMMSS).
    Returns URL to the generated belt comparison graph.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    printer_dir = get_printer_dir(printer)

    # Build file list from either 'files' array or individual file_a/file_b params
    upload_files = []
    if files:
        upload_files.extend(files)
    if file_a:
        upload_files.append(file_a)
    if file_b:
        upload_files.append(file_b)

    if len(upload_files) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Belt analysis requires 2 files. Received {len(upload_files)} file(s)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(upload_files, tmpdir)
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Dashboard home page showing all printers."""
    printers = get_all_results()
    return templates.TemplateResponse("home.html", {
        "request": request,
        "printers": printers
    })


@app.get("/printer/{printer_name}", response_class=HTMLResponse)
async def printer_detail(request: Request, printer_name: str):
    """Printer detail page showing all results."""
    all_results = get_all_results()

    if printer_name not in all_results:
        # Check if printer directory exists but is empty
        printer_path = os.path.join(RESULTS_DIR, printer_name)
        if os.path.isdir(printer_path):
            results = {"shaper": [], "belts": [], "vibrations": []}
        else:
            raise HTTPException(status_code=404, detail=f"Printer '{printer_name}' not found")
    else:
        results = all_results[printer_name]

    return templates.TemplateResponse("printer.html", {
        "request": request,
        "printer_name": printer_name,
        "results": results
    })


@app.get("/api")
async def api_docs():
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
            "GET /": "Web dashboard",
            "GET /printer/{name}": "Printer results page",
        },
        "usage": {
            "single_printer": 'curl -X POST http://host:3080/shaper -F "files=@x.csv" -F "files=@y.csv"',
            "multi_printer": 'curl -X POST http://host:3080/shaper -F "files=@x.csv" -F "files=@y.csv" -F "printer=k1v3"',
        }
    }
