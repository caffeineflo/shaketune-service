"""HTTP service for generating Shake&Tune analysis graphs."""

import asyncio
import gzip
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title='Shake&Tune Analysis Service',
    description='Process Klipper accelerometer data and generate calibration graphs',
    version='1.2.0',
)

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/app/results')
KLIPPER_DIR = os.environ.get('KLIPPER_DIR', '/app/service/klipper')
APP_DIR = os.environ.get('APP_DIR', '/app')

_MEBIBYTE = 1024 * 1024
UPLOAD_CHUNK_SIZE = _MEBIBYTE
PRINTER_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
TIMESTAMP_PATTERN = re.compile(r'\d{8}_\d{6}\Z')
RESULT_PATTERN = re.compile(r'^(\d{8}_\d{6})_(shaper|belts|vibrations)(?:_\w+)?\.png$')
TOKEN_HEADER = 'X-ShakeTune-Token'
ANALYSIS_PATHS = frozenset({'/shaper', '/belts', '/vibrations'})


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be a positive integer') from exc
    if value <= 0:
        raise RuntimeError(f'{name} must be a positive integer')
    return value


MAX_UPLOAD_FILES = _positive_int_from_env('SHAKETUNE_MAX_UPLOAD_FILES', 16)
MAX_UPLOAD_BYTES_PER_FILE = _positive_int_from_env('SHAKETUNE_MAX_UPLOAD_BYTES', 32 * _MEBIBYTE)
MAX_DECOMPRESSED_BYTES_PER_FILE = _positive_int_from_env(
    'SHAKETUNE_MAX_DECOMPRESSED_BYTES',
    64 * _MEBIBYTE,
)
MAX_TOTAL_DECOMPRESSED_BYTES = _positive_int_from_env(
    'SHAKETUNE_MAX_TOTAL_DECOMPRESSED_BYTES',
    128 * _MEBIBYTE,
)
MAX_REQUEST_BODY_BYTES = _positive_int_from_env(
    'SHAKETUNE_MAX_REQUEST_BODY_BYTES',
    64 * _MEBIBYTE,
)
ANALYSIS_TIMEOUT_SECONDS = _positive_int_from_env('SHAKETUNE_ANALYSIS_TIMEOUT_SECONDS', 300)
ANALYSIS_CONCURRENCY = _positive_int_from_env('SHAKETUNE_ANALYSIS_CONCURRENCY', 1)
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(ANALYSIS_CONCURRENCY)


def _load_api_token() -> str:
    token = os.environ.get('SHAKETUNE_API_TOKEN')
    token_file = os.environ.get('SHAKETUNE_API_TOKEN_FILE')

    if token is None and token_file:
        try:
            token = Path(token_file).read_text(encoding='utf-8')
        except OSError as exc:
            raise RuntimeError('Unable to read SHAKETUNE_API_TOKEN_FILE') from exc

    if token is None or not token.strip():
        raise RuntimeError('Set SHAKETUNE_API_TOKEN or SHAKETUNE_API_TOKEN_FILE')

    return token.strip()


API_TOKEN = _load_api_token()


def _valid_api_token(supplied_token: Optional[str]) -> bool:
    supplied_bytes = supplied_token.encode('utf-8') if supplied_token is not None else b''
    return secrets.compare_digest(supplied_bytes, API_TOKEN.encode('utf-8'))


class RequestBodyTooLarge(Exception):
    """The incoming HTTP request exceeded the configured body limit."""


class RequestBodyLimitMiddleware:
    """Reject oversized analysis bodies while the ASGI server receives them."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope['type'] != 'http'
            or scope.get('method') != 'POST'
            or scope.get('path') not in ANALYSIS_PATHS
        ):
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get('headers', [])
                if name.lower() == b'content-length'
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return

        received_bytes = 0
        body_too_large = False
        response_started = False

        async def receive_limited():
            nonlocal body_too_large, received_bytes
            message = await receive()
            if message['type'] == 'http.request':
                received_bytes += len(message.get('body', b''))
                if received_bytes > self.max_body_bytes:
                    body_too_large = True
                    raise RequestBodyTooLarge
            return message

        async def send_limited(message):
            nonlocal response_started
            if body_too_large:
                return
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive_limited, send_limited)
        except RequestBodyTooLarge:
            body_too_large = True

        if body_too_large:
            if response_started:
                raise RuntimeError('Request body limit was crossed after the response started')
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={'detail': 'Request body exceeds the configured limit'},
        )
        await response(scope, receive, send)


app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=MAX_REQUEST_BODY_BYTES)


@app.middleware('http')
async def authenticate_analysis_requests(request: Request, call_next):
    """Authenticate analysis requests before FastAPI parses multipart bodies."""
    if request.method == 'POST' and request.url.path in ANALYSIS_PATHS:
        if not _valid_api_token(request.headers.get(TOKEN_HEADER)):
            return JSONResponse(
                status_code=401,
                content={'detail': 'Invalid or missing API token'},
                headers={'WWW-Authenticate': 'ShakeTune-Token'},
            )
    return await call_next(request)


Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
app.mount('/results', StaticFiles(directory=RESULTS_DIR), name='results')

TEMPLATES_DIR = Path(__file__).parent / 'templates'
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class UploadRejected(Exception):
    """An uploaded file cannot be processed safely."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_printer(printer: Optional[str]) -> str:
    candidate = 'default' if printer is None else printer
    if not PRINTER_PATTERN.fullmatch(candidate):
        raise HTTPException(
            status_code=400,
            detail='Printer must contain 1-64 letters, numbers, underscores, or hyphens',
        )
    return candidate


def validate_timestamp(timestamp: Optional[str]) -> str:
    if timestamp is None:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    if not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise HTTPException(status_code=400, detail='Timestamp must use YYYYMMDD_HHMMSS')
    try:
        datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Timestamp is not a valid date and time') from exc
    return timestamp


def _safe_child(root: Path, name: str) -> Path:
    root = root.resolve()
    child = (root / name).resolve()
    if child.parent != root:
        raise ValueError('Path escapes its assigned directory')
    return child


def _validate_upload_name(filename: Optional[str]) -> Tuple[str, str, bool]:
    if not filename:
        raise UploadRejected(400, 'Every upload must have a filename')
    if filename in {'.', '..'} or '/' in filename or '\\' in filename or '\x00' in filename:
        raise UploadRejected(400, f'Upload filename must be a basename: {filename!r}')
    if any(ord(character) < 32 or ord(character) == 127 for character in filename):
        raise UploadRejected(400, 'Upload filename contains control characters')
    if len(filename.encode('utf-8')) > 255:
        raise UploadRejected(400, 'Upload filename is too long')

    is_gzip = filename.endswith('.csv.gz')
    if is_gzip:
        output_name = filename[:-3]
    elif filename.endswith('.csv'):
        output_name = filename
    else:
        raise UploadRejected(400, f'Upload must use a .csv or .csv.gz extension: {filename!r}')

    if output_name.lower() == '.csv':
        raise UploadRejected(400, 'Upload filename must include a name before .csv')
    return filename, output_name, is_gzip


def _upload_spec(upload: UploadFile) -> Tuple[str, str, bool]:
    return _validate_upload_name(upload.filename)


def _validate_upload_metadata(files: List[UploadFile]) -> List[Tuple[UploadFile, str, str, bool]]:
    if not files:
        raise UploadRejected(400, 'At least one upload is required')
    if len(files) > MAX_UPLOAD_FILES:
        raise UploadRejected(413, f'At most {MAX_UPLOAD_FILES} files may be uploaded at once')

    specs = []
    output_names = set()
    for upload in files:
        filename, output_name, is_gzip = _upload_spec(upload)
        if output_name in output_names:
            raise UploadRejected(400, f'Duplicate upload filename: {output_name!r}')
        output_names.add(output_name)
        specs.append((upload, filename, output_name, is_gzip))
    return specs


async def _store_upload(upload: UploadFile, destination: Path) -> int:
    bytes_written = 0
    with destination.open('xb') as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES_PER_FILE:
                raise UploadRejected(
                    413,
                    f'Upload exceeds {MAX_UPLOAD_BYTES_PER_FILE} bytes before decompression',
                )
            output.write(chunk)

    if bytes_written == 0:
        raise UploadRejected(400, f'Upload {upload.filename!r} is empty')
    return bytes_written


def _decompress_gzip(source: Path, destination: Path, total_already_written: int) -> int:
    bytes_written = 0
    try:
        with gzip.open(source, 'rb') as compressed, destination.open('xb') as output:
            while True:
                chunk = compressed.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_DECOMPRESSED_BYTES_PER_FILE:
                    raise UploadRejected(
                        413,
                        f'Decompressed upload exceeds {MAX_DECOMPRESSED_BYTES_PER_FILE} bytes',
                    )
                if total_already_written + bytes_written > MAX_TOTAL_DECOMPRESSED_BYTES:
                    raise UploadRejected(
                        413,
                        f'Uploads exceed {MAX_TOTAL_DECOMPRESSED_BYTES} decompressed bytes in total',
                    )
                output.write(chunk)
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        destination.unlink(missing_ok=True)
        raise UploadRejected(400, f'Upload {source.name!r} is not valid gzip data') from exc
    except UploadRejected:
        destination.unlink(missing_ok=True)
        raise

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise UploadRejected(400, f'Upload {source.name!r} expands to an empty file')
    return bytes_written


async def save_uploaded_files(files: List[UploadFile], tmpdir: str) -> List[str]:
    """Stream uploads into a contained temp directory and expand gzip safely."""
    try:
        specs = _validate_upload_metadata(files)
        root = Path(tmpdir).resolve()
        csv_paths = []
        total_decompressed = 0

        for upload, filename, output_name, is_gzip in specs:
            source_path = _safe_child(root, filename)
            output_path = _safe_child(root, output_name)

            await _store_upload(upload, source_path)
            if is_gzip:
                decompressed = await asyncio.to_thread(
                    _decompress_gzip,
                    source_path,
                    output_path,
                    total_decompressed,
                )
                source_path.unlink()
            else:
                decompressed = source_path.stat().st_size
                if decompressed > MAX_DECOMPRESSED_BYTES_PER_FILE:
                    raise UploadRejected(
                        413,
                        f'Upload exceeds {MAX_DECOMPRESSED_BYTES_PER_FILE} bytes',
                    )
                if total_decompressed + decompressed > MAX_TOTAL_DECOMPRESSED_BYTES:
                    raise UploadRejected(
                        413,
                        f'Uploads exceed {MAX_TOTAL_DECOMPRESSED_BYTES} bytes in total',
                    )

            total_decompressed += decompressed
            csv_paths.append(str(output_path))

        return csv_paths
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Upload path is invalid') from exc
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _bounded_process_error(result: subprocess.CompletedProcess) -> str:
    raw_message = result.stderr or result.stdout or 'no diagnostic output'
    return ' '.join(str(raw_message).split())[:500]


async def run_graph_cli(
    graph_type: str,
    csv_paths: List[str],
    output_path: str,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Run one bounded graph-analysis process without blocking the event loop."""
    cmd = [
        sys.executable,
        '-m',
        'shaketune.cli',
        graph_type,
        '-k',
        KLIPPER_DIR,
        '-o',
        output_path,
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(csv_paths)

    env = os.environ.copy()
    env.pop('SHAKETUNE_API_TOKEN', None)
    env.pop('SHAKETUNE_API_TOKEN_FILE', None)
    env['PYTHONPATH'] = APP_DIR

    async with _ANALYSIS_SEMAPHORE:
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=APP_DIR,
                timeout=ANALYSIS_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=504,
                detail=f'Analysis timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds',
            ) from exc
        except OSError as exc:
            detail = ' '.join(str(exc).split())[:300]
            raise HTTPException(
                status_code=500,
                detail=f'Analysis process could not start: {detail}',
            ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f'Analysis failed: {_bounded_process_error(result)}',
        )

    output = Path(output_path)
    if not output.is_file() or output.stat().st_size == 0:
        raise HTTPException(status_code=500, detail='Analysis produced no graph output')


def get_printer_dir(printer: str) -> str:
    """Get or create a contained printer-specific results directory."""
    root = Path(RESULTS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        printer_dir = _safe_child(root, printer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Printer path is invalid') from exc
    printer_dir.mkdir(exist_ok=True)
    return str(printer_dir)


def _axis_from_filename(filename: str) -> str:
    try:
        _, output_name, _ = _validate_upload_name(filename)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    stem = output_name[:-4]
    axis = re.split(r'[_.-]+', stem.lower())[-1]
    if axis not in {'x', 'y'}:
        raise HTTPException(
            status_code=400,
            detail=f'Shaper filename must end with an x or y axis label: {filename!r}',
        )
    return axis


def _collect_optional_uploads(*groups: Any) -> List[UploadFile]:
    uploads = []
    for group in groups:
        if isinstance(group, list):
            uploads.extend(group)
        elif group is not None:
            uploads.append(group)
    return uploads


@app.post('/shaper')
async def analyze_shaper(
    files: Annotated[Optional[List[UploadFile]], File()] = None,
    file_x: Annotated[Optional[UploadFile], File()] = None,
    file_y: Annotated[Optional[UploadFile], File()] = None,
    printer: Annotated[Optional[str], Form()] = 'default',
    timestamp: Annotated[Optional[str], Form()] = None,
    max_freq: Annotated[float, Form()] = 200.0,
    scv: Annotated[float, Form()] = 5.0,
):
    """Generate one input-shaper graph per uploaded X or Y resonance file."""
    printer_name = validate_printer(printer)
    ts = validate_timestamp(timestamp)
    upload_files = _collect_optional_uploads(files, file_x, file_y)

    if not upload_files:
        raise HTTPException(status_code=400, detail='Shaper analysis requires at least 1 file. Received 0 file(s).')
    if len(upload_files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f'At most {MAX_UPLOAD_FILES} files may be uploaded at once')

    axes = []
    for upload in upload_files:
        axis = _axis_from_filename(upload.filename or '')
        if axis in axes:
            raise HTTPException(status_code=400, detail=f'Duplicate shaper axis: {axis}')
        axes.append(axis)

    printer_dir = get_printer_dir(printer_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(upload_files, tmpdir)
        extra_args = ['--max_freq', str(max_freq), '--scv', str(scv)]
        results = []

        for index, csv_path in enumerate(csv_paths):
            axis = axes[index]
            output_png = os.path.join(tmpdir, f'shaper_{axis}.png')
            await run_graph_cli('input_shaper', [csv_path], output_png, extra_args)

            final_name = f'{ts}_shaper_{axis}.png'
            final_path = os.path.join(printer_dir, final_name)
            shutil.move(output_png, final_path)
            results.append({'url': f'/results/{printer_name}/{final_name}', 'axis': axis})

    return {'urls': results, 'type': 'shaper', 'printer': printer_name}


@app.post('/belts')
async def analyze_belts(
    files: Annotated[Optional[List[UploadFile]], File()] = None,
    file_a: Annotated[Optional[UploadFile], File()] = None,
    file_b: Annotated[Optional[UploadFile], File()] = None,
    printer: Annotated[Optional[str], Form()] = 'default',
    timestamp: Annotated[Optional[str], Form()] = None,
    max_freq: Annotated[float, Form()] = 200.0,
    kinematics: Annotated[str, Form()] = 'corexy',
):
    """Generate a comparison graph from exactly two belt resonance files."""
    printer_name = validate_printer(printer)
    ts = validate_timestamp(timestamp)
    upload_files = _collect_optional_uploads(files, file_a, file_b)

    if len(upload_files) != 2:
        raise HTTPException(
            status_code=400,
            detail=f'Belt analysis requires 2 files. Received {len(upload_files)} file(s).',
        )

    printer_dir = get_printer_dir(printer_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(upload_files, tmpdir)
        output_png = os.path.join(tmpdir, 'belts.png')
        extra_args = ['--max_freq', str(max_freq), '--kinematics', kinematics]
        await run_graph_cli('belts', csv_paths, output_png, extra_args)

        final_name = f'{ts}_belts.png'
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {'url': f'/results/{printer_name}/{final_name}', 'type': 'belts', 'printer': printer_name}


@app.post('/vibrations')
async def analyze_vibrations(
    files: Annotated[List[UploadFile], File()],
    printer: Annotated[Optional[str], Form()] = 'default',
    timestamp: Annotated[Optional[str], Form()] = None,
    kinematics: Annotated[str, Form()] = 'corexy',
    max_freq: Annotated[float, Form()] = 1000.0,
):
    """Generate a vibration graph from one or more measurement files."""
    printer_name = validate_printer(printer)
    ts = validate_timestamp(timestamp)
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f'At most {MAX_UPLOAD_FILES} files may be uploaded at once')

    printer_dir = get_printer_dir(printer_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_paths = await save_uploaded_files(files, tmpdir)
        output_png = os.path.join(tmpdir, 'vibrations.png')
        extra_args = ['--max_freq', str(max_freq), '--kinematics', str(kinematics)]
        await run_graph_cli('vibrations', csv_paths, output_png, extra_args)

        final_name = f'{ts}_vibrations.png'
        final_path = os.path.join(printer_dir, final_name)
        shutil.move(output_png, final_path)

    return {'url': f'/results/{printer_name}/{final_name}', 'type': 'vibrations', 'printer': printer_name}


@app.get('/health')
async def health():
    """Report that the HTTP process is accepting requests."""
    return {'status': 'ok', 'service': 'shaketune-service'}


def get_all_results() -> Dict[str, Dict[str, Any]]:
    """Scan the results directory and return structured printer data."""
    results = {}
    root = Path(RESULTS_DIR)
    if not root.exists():
        return results

    for printer_path in root.iterdir():
        if not printer_path.is_dir() or printer_path.is_symlink():
            continue

        printer_data: Dict[str, Any] = {'shaper': [], 'belts': [], 'vibrations': []}
        for result_path in printer_path.iterdir():
            if not result_path.is_file() or result_path.is_symlink():
                continue
            match = RESULT_PATTERN.match(result_path.name)
            if not match:
                continue
            ts_str, graph_type = match.groups()
            try:
                formatted_ts = datetime.strptime(ts_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                formatted_ts = ts_str
            printer_data[graph_type].append(
                {'file': result_path.name, 'timestamp': formatted_ts, 'sort_key': ts_str},
            )

        for graph_type in ('shaper', 'belts', 'vibrations'):
            printer_data[graph_type].sort(key=lambda item: item['sort_key'], reverse=True)

        all_results = printer_data['shaper'] + printer_data['belts'] + printer_data['vibrations']
        printer_data['last_activity'] = max(all_results, key=lambda item: item['sort_key'])['timestamp'] if all_results else None
        if all_results:
            results[printer_path.name] = printer_data

    return results


def get_latest_file(graph_type: str, printer: str = 'default') -> Optional[str]:
    """Find the most recent graph file of a given type for one printer."""
    printer_name = validate_printer(printer)
    printer_dir = Path(RESULTS_DIR) / printer_name
    if not printer_dir.exists() or not printer_dir.is_dir() or printer_dir.is_symlink():
        return None
    pattern = re.compile(rf'^\d{{8}}_\d{{6}}_{re.escape(graph_type)}(_\w+)?\.png$')
    files = sorted(
        (
            path.name
            for path in printer_dir.iterdir()
            if path.is_file() and not path.is_symlink() and pattern.match(path.name)
        ),
        reverse=True,
    )
    return files[0] if files else None


@app.get('/latest/{printer}/{graph_type}')
async def latest_graph_for_printer(printer: str, graph_type: str):
    """Redirect to one printer's most recent graph of a supported type."""
    if graph_type not in {'shaper', 'belts', 'vibrations'}:
        raise HTTPException(status_code=400, detail=f'Invalid graph type: {graph_type}')
    printer_name = validate_printer(printer)
    latest = get_latest_file(graph_type, printer_name)
    if not latest:
        raise HTTPException(status_code=404, detail=f"No {graph_type} graphs found for printer '{printer_name}'")
    return RedirectResponse(url=f'/results/{printer_name}/{latest}', status_code=302)


@app.get('/latest/{graph_type}')
async def latest_graph(graph_type: str):
    """Redirect to the default printer's most recent graph of a supported type."""
    if graph_type not in {'shaper', 'belts', 'vibrations'}:
        raise HTTPException(status_code=400, detail=f'Invalid graph type: {graph_type}')
    latest = get_latest_file(graph_type)
    if not latest:
        raise HTTPException(status_code=404, detail=f'No {graph_type} graphs found')
    return RedirectResponse(url=f'/results/default/{latest}', status_code=302)


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    """Render the dashboard home page."""
    return templates.TemplateResponse(request, 'home.html', {'printers': get_all_results()})


@app.get('/printer/{printer_name}', response_class=HTMLResponse)
async def printer_detail(request: Request, printer_name: str):
    """Render all available results for one printer."""
    printer_name = validate_printer(printer_name)
    all_results = get_all_results()

    if printer_name not in all_results:
        printer_path = Path(RESULTS_DIR) / printer_name
        if printer_path.is_dir() and not printer_path.is_symlink():
            results = {'shaper': [], 'belts': [], 'vibrations': []}
        else:
            raise HTTPException(status_code=404, detail=f"Printer '{printer_name}' not found")
    else:
        results = all_results[printer_name]

    return templates.TemplateResponse(
        request,
        'printer.html',
        {'printer_name': printer_name, 'results': results},
    )


@app.get('/api')
async def api_docs():
    """Return concise API usage information."""
    return {
        'service': 'Shake&Tune Analysis Service',
        'version': '1.2.0',
        'description': 'Process Klipper accelerometer data for input shaper calibration',
        'authentication': f'POST endpoints require the {TOKEN_HEADER} header',
        'endpoints': {
            'POST /shaper': 'Upload resonance CSVs, get input shaper graphs',
            'POST /belts': 'Upload two belt CSVs, get a comparison graph',
            'POST /vibrations': 'Upload vibration CSVs, get a speed analysis graph',
            'GET /results/{printer}/{filename}': 'Retrieve a generated graph',
            'GET /latest/{printer}/{type}': "Redirect to a printer's latest graph",
            'GET /latest/{type}': 'Redirect to the default printer latest graph',
            'GET /health': 'Health check',
            'GET /': 'Web dashboard',
            'GET /printer/{name}': 'Printer results page',
        },
    }
