from datetime import datetime, timedelta, timezone
from collections import defaultdict
import csv
import json
import os
import sqlite3
from io import BytesIO, StringIO
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

from predictor_disease import DiseasePredictor
from predictor_quality import QualityPredictor

app = Flask(__name__)
CORS(app)

# Paths and database helpers
BASE_DIR = os.path.dirname(__file__)
DATABASE_PATH = os.path.join(BASE_DIR, "app.db")

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS disease_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                sample_id TEXT,
                location TEXT,
                notes TEXT,
                disease TEXT,
                confidence REAL,
                severity TEXT,
                symptoms TEXT,
                treatment TEXT,
                prevention TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                batch_id TEXT,
                notes TEXT,
                annotated_image TEXT,
                detections_json TEXT NOT NULL,
                total INTEGER NOT NULL,
                class_counts_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()
    finally:
        conn.close()



def parse_iso_timestamp(value):
    """Convierte timestamps ISO8601 en datetime con tz UTC."""
    if not value:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None



def compute_quality_score(class_counts: dict[str, int] | None, total: int | float | None) -> float | None:
    """Calcula una puntuacion de 0 a 10 basada en la distribucion detectada."""
    if not class_counts:
        return None

    try:
        total_count = int(total) if total is not None else 0
    except (ValueError, TypeError):
        total_count = 0

    if total_count <= 0:
        total_count = sum(int(value) for value in class_counts.values() if isinstance(value, (int, float)))

    if total_count <= 0:
        return None

    weighted_sum = 0.0
    for key, value in class_counts.items():
        if not isinstance(value, (int, float)):
            continue
        weight = QUALITY_WEIGHTS.get(str(key), DEFAULT_CLASS_WEIGHT)
        weighted_sum += float(value) * weight

    return (weighted_sum / total_count) * 10



def format_severity_label(value: str | None) -> str:
    """Normaliza etiquetas de severidad a un formato legible."""
    if not value:
        return "Sin clasificacion"

    normalized = str(value).strip().lower()
    if not normalized:
        return "Sin clasificacion"

    if normalized.startswith("alto"):
        return "Alto"
    if normalized.startswith("medio"):
        return "Medio"
    if normalized.startswith("bajo"):
        return "Bajo"
    return normalized.capitalize()

# Predictor instances
disease_predictor = None
quality_predictor = None

QUALITY_WEIGHTS = {
    "b_fully_ripened": 1.0,
    "b_half_ripened": 0.7,
    "b_green": 0.4,
}

DEFAULT_CLASS_WEIGHT = 0.5




def initialize_predictors():
    global disease_predictor, quality_predictor

    models_dir = os.path.join(BASE_DIR, "models")
    try:
        disease_path = os.path.join(models_dir, "mymodel_v4.keras")
        if os.path.exists(disease_path):
            # DiseasePredictor espera nombre de archivo relativo a ./models
            disease_predictor = DiseasePredictor("mymodel_v4.keras")
            print(f"Disease model loaded from {disease_path}")
        else:
            print(f"Warning: disease model not found in {disease_path}")

        quality_path = os.path.join(models_dir, "best.pt")
        if os.path.exists(quality_path):
            quality_predictor = QualityPredictor(quality_path)
            print(f"Quality model loaded from {quality_path}")
        else:
            print(f"Warning: quality model not found in {quality_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error initializing models: {exc}")


def serialize_disease_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "timestamp": row["timestamp"],
        "sampleId": row["sample_id"],
        "location": row["location"],
        "notes": row["notes"],
        "disease": row["disease"],
        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
        "severity": row["severity"],
        "symptoms": row["symptoms"],
        "treatment": row["treatment"],
        "prevention": row["prevention"],
    }


def serialize_quality_row(row: sqlite3.Row) -> dict:
    detections = json.loads(row["detections_json"]) if row["detections_json"] else []
    class_counts = json.loads(row["class_counts_json"]) if row["class_counts_json"] else {}
    class_counts = {str(key): int(value) for key, value in class_counts.items()}
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "timestamp": row["timestamp"],
        "batchId": row["batch_id"],
        "notes": row["notes"],
        "annotatedImage": row["annotated_image"],
        "detections": detections,
        "total": int(row["total"] or len(detections)),
        "classCounts": class_counts,
    }


def fetch_user_records(user_id: str) -> tuple[list[dict], list[dict]]:
    conn = get_db()
    try:
        disease_rows = conn.execute(
            "SELECT * FROM disease_analyses WHERE user_id = ? ORDER BY datetime(timestamp) DESC",
            (user_id,),
        ).fetchall()
        quality_rows = conn.execute(
            "SELECT * FROM quality_analyses WHERE user_id = ? ORDER BY datetime(timestamp) DESC",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    disease_records = [serialize_disease_row(row) for row in disease_rows]
    quality_records = [serialize_quality_row(row) for row in quality_rows]
    return disease_records, quality_records


def filter_records_by_range(records: list[dict], start: datetime | None, end: datetime | None) -> list[dict]:
    filtered: list[dict] = []
    for record in records:
        timestamp = parse_iso_timestamp(record.get("timestamp"))
        if start and (timestamp is None or timestamp < start):
            continue
        if end and (timestamp is None or timestamp > end):
            continue
        enriched = dict(record)
        enriched["timestamp_dt"] = timestamp
        filtered.append(enriched)
    return filtered


def calculate_dashboard_metrics(
    disease_records: list[dict],
    quality_records: list[dict],
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    start_dt = parse_iso_timestamp(start) if start else None
    end_dt = parse_iso_timestamp(end) if end else None

    disease_filtered = filter_records_by_range(disease_records, start_dt, end_dt)
    quality_filtered = filter_records_by_range(quality_records, start_dt, end_dt)

    now = datetime.now(timezone.utc)
    recent_window_start = now - timedelta(days=14)

    severity_counts: dict[str, int] = {"alto": 0, "medio": 0, "bajo": 0, "otros": 0}
    confidence_values: list[float] = []
    disease_timeseries: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "highSeverity": 0})

    for record in disease_filtered:
        timestamp = record.get("timestamp_dt")
        severity = (record.get("severity") or "").lower()
        if severity.startswith("alto"):
            severity_counts["alto"] += 1
        elif severity.startswith("medio"):
            severity_counts["medio"] += 1
        elif severity.startswith("bajo"):
            severity_counts["bajo"] += 1
        else:
            severity_counts["otros"] += 1

        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))

        if timestamp:
            day_key = timestamp.date().isoformat()
            bucket = disease_timeseries[day_key]
            bucket["total"] += 1
            if severity.startswith("alto"):
                bucket["highSeverity"] += 1

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

    quality_timeseries: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "premium": 0, "green": 0})
    class_totals: dict[str, int] = defaultdict(int)
    quality_scores: list[float] = []

    for record in quality_filtered:
        class_counts = record.get("classCounts") or {}
        for key, value in class_counts.items():
            class_totals[key] += int(value)
        timestamp = record.get("timestamp_dt")
        if timestamp:
            day_key = timestamp.date().isoformat()
            bucket = quality_timeseries[day_key]
            bucket["total"] += 1
            bucket["premium"] += int(class_counts.get("b_fully_ripened", 0))
            bucket["green"] += int(class_counts.get("b_green", 0))

        score = compute_quality_score(class_counts, record.get("total") or 0)
        if score is not None:
            quality_scores.append(score)

    average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    total_class_counts = sum(class_totals.values())
    class_distribution: dict[str, float] = {}
    if total_class_counts:
        class_distribution = {
            key: round((value / total_class_counts) * 100, 1)
            for key, value in class_totals.items()
        }

    disease_recent_sorted = sorted(
        disease_filtered,
        key=lambda item: item.get("timestamp_dt") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    quality_recent_sorted = sorted(
        quality_filtered,
        key=lambda item: item.get("timestamp_dt") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    disease_recent = [
        {
            "id": record["id"],
            "sampleId": record.get("sampleId"),
            "disease": record.get("disease"),
            "severity": record.get("severity"),
            "timestamp": record.get("timestamp"),
        }
        for record in disease_recent_sorted[:5]
    ]

    quality_recent = []
    for record in quality_recent_sorted[:5]:
        class_counts = record.get("classCounts") or {}
        dominant_key, dominant_value = None, 0
        for key, value in class_counts.items():
            if value > dominant_value:
                dominant_key, dominant_value = key, value
        total = record.get("total") or sum(class_counts.values())
        percentage = round((dominant_value / total * 100), 1) if total else 0.0
        quality_recent.append(
            {
                "id": record["id"],
                "batchId": record.get("batchId"),
                "label": dominant_key,
                "percentage": percentage,
                "timestamp": record.get("timestamp"),
            }
        )

    disease_recent_count = sum(
        1
        for record in disease_filtered
        if record.get("timestamp_dt") and record["timestamp_dt"] >= recent_window_start
    )

    total_analyses = len(disease_filtered) + len(quality_filtered)
    unique_diseases = len(
        {
            (record.get("disease") or "").strip().lower()
            for record in disease_filtered
            if record.get("disease")
        }
    )

    activity_entries = []
    for record in disease_recent_sorted:
        activity_entries.append(
            {
                "id": f"disease-{record['id']}",
                "type": "disease",
                "title": record.get("disease") or "Analisis de enfermedad",
                "detail": record.get("sampleId") or "",
                "timestamp": record.get("timestamp"),
                "status": format_severity_label(record.get("severity")),
            }
        )

    for record in quality_recent_sorted:
        class_counts = record.get("classCounts") or {}
        dominant_key = max(class_counts.items(), key=lambda item: item[1], default=(None, 0))[0]
        activity_entries.append(
            {
                "id": f"quality-{record['id']}",
                "type": "quality",
                "title": record.get("batchId") or "Evaluacion de calidad",
                "detail": dominant_key or "Sin clasificacion",
                "timestamp": record.get("timestamp"),
                "status": "Procesado",
            }
        )

    activity_entries = [
        entry
        for entry in activity_entries
        if parse_iso_timestamp(entry.get("timestamp")) is not None
    ]

    fallback_timestamp = datetime.min.replace(tzinfo=timezone.utc)
    activity_entries.sort(
        key=lambda item: parse_iso_timestamp(item.get("timestamp")) or fallback_timestamp,
        reverse=True,
    )

    activity = activity_entries[:5]

    alerts: list[dict] = []
    if severity_counts["alto"] >= 3:
        alerts.append(
            {
                "type": "disease",
                "title": "Casos criticos detectados",
                "message": f"Se registraron {severity_counts['alto']} analisis con severidad alta en las ultimas semanas.",
            }
        )
    if total_analyses and average_confidence and average_confidence < 60:
        alerts.append(
            {
                "type": "model",
                "title": "Confianza media baja",
                "message": f"La confianza promedio del modelo es {average_confidence:.1f}%. Verifica las imagenes de entrada.",
            }
        )
    if total_class_counts:
        green_ratio = class_totals.get("b_green", 0) / total_class_counts
        if green_ratio > 0.5:
            alerts.append(
                {
                    "type": "quality",
                    "title": "Alta proporcion de tomates verdes",
                    "message": "Mas del 50% de las detecciones recientes corresponden a tomates verdes.",
                }
            )

    disease_timeseries_list = [
        {
            "date": date,
            "total": bucket["total"],
            "highSeverity": bucket["highSeverity"],
        }
        for date, bucket in sorted(disease_timeseries.items())
    ]

    quality_timeseries_list = [
        {
            "date": date,
            "total": bucket["total"],
            "premium": bucket["premium"],
            "green": bucket["green"],
        }
        for date, bucket in sorted(quality_timeseries.items())
    ]

    summary = {
        "analysis": {
            "total": total_analyses,
            "deltaLabel": "Sin analisis registrados" if total_analyses == 0 else f"{total_analyses} acumulados",
        },
        "disease": {
            "total": len(disease_filtered),
            "unique": unique_diseases,
            "recentCount": disease_recent_count,
            "recentLabel": "Sin registros recientes" if disease_recent_count == 0 else f"{disease_recent_count} en las ultimas 2 semanas",
        },
        "quality": {
            "score": round(average_quality_score, 1),
            "total": len(quality_filtered),
            "label": "Sin lotes evaluados" if not quality_filtered else f"Promedio basado en {len(quality_filtered)} lotes",
        },
        "confidence": {
            "percentage": round(average_confidence, 1),
            "label": "Sin datos de confianza" if not confidence_values else "Promedio de confianza del modelo",
        },
    }

    return {
        "summary": summary,
        "disease": {
            "recent": disease_recent,
            "severityBreakdown": severity_counts,
        },
        "quality": {
            "recent": quality_recent,
            "classDistribution": class_distribution,
        },
        "activity": activity,
        "alerts": alerts,
        "timeseries": {
            "disease": disease_timeseries_list,
            "quality": quality_timeseries_list,
        },
    }



@app.route("/metrics/overview", methods=["GET", "OPTIONS"])
def metrics_overview():
    if request.method == "OPTIONS":
        return ("", 204)

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id es obligatorio"}), 400

    start = request.args.get("start")
    end = request.args.get("end")

    try:
        disease_records, quality_records = fetch_user_records(user_id)
        metrics = calculate_dashboard_metrics(
            disease_records,
            quality_records,
            start=start,
            end=end,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error calculating dashboard metrics: {exc}")
        return jsonify({"error": "Error calculando metricas"}), 500

    return jsonify(metrics), 200



def generate_history_csv(
    disease_records: list[dict],
    quality_records: list[dict],
    *,
    start: str | None = None,
    end: str | None = None,
) -> bytes:
    start_dt = parse_iso_timestamp(start) if start else None
    end_dt = parse_iso_timestamp(end) if end else None

    disease_filtered = filter_records_by_range(disease_records, start_dt, end_dt)
    quality_filtered = filter_records_by_range(quality_records, start_dt, end_dt)

    combined_rows = []

    for record in disease_filtered:
        combined_rows.append(
            {
                "timestamp": record.get("timestamp"),
                "tipo": "enfermedad",
                "identificador": record.get("sampleId"),
                "detalle": record.get("disease"),
                "categoria": record.get("severity"),
                "confianza": record.get("confidence"),
                "notas": record.get("notes"),
            }
        )

    for record in quality_filtered:
        class_counts = record.get("classCounts") or {}
        dominant_key = max(class_counts.items(), key=lambda item: item[1], default=(None, 0))[0]
        combined_rows.append(
            {
                "timestamp": record.get("timestamp"),
                "tipo": "calidad",
                "identificador": record.get("batchId"),
                "detalle": dominant_key,
                "categoria": "",
                "confianza": record.get("total"),
                "notas": record.get("notes"),
            }
        )

    combined_rows.sort(
        key=lambda item: parse_iso_timestamp(item["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "tipo", "identificador", "detalle", "categoria", "valor/confianza", "notas"])
    for row in combined_rows:
        confianza = row.get("confianza")
        if isinstance(confianza, (int, float)):
            confianza_str = f"{confianza:.2f}"
        else:
            confianza_str = confianza or ""
        writer.writerow(
            [
                row.get("timestamp") or "",
                row.get("tipo") or "",
                row.get("identificador") or "",
                row.get("detalle") or "",
                row.get("categoria") or "",
                confianza_str,
                row.get("notas") or "",
            ]
        )

    return output.getvalue().encode("utf-8")


def build_simple_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n%\xff\xff\xff\xff\n")

    objects: list[bytes] = []

    def write_obj(payload: bytes) -> None:
        if not payload.endswith(b"\n"):
            payload += b"\n"
        objects.append(payload)

    write_obj(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    write_obj(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    write_obj(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")

    content_lines = ["BT", "/F1 12 Tf", "12 TL", "1 0 0 1 50 760 Tm"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(f"({safe}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")

    content_bytes = "\n".join(content_lines).encode("latin-1", "replace")
    write_obj(
        f"4 0 obj << /Length {len(content_bytes)} >> stream\n".encode("ascii") + content_bytes + b"\nendstream\nendobj"
    )
    write_obj(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj")

    offsets = []
    for obj in objects:
        offsets.append(buffer.tell())
        buffer.write(obj)
        buffer.write(b"\n")

    xref_pos = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))

    buffer.write(b"trailer\n<< /Size ")
    buffer.write(str(len(offsets) + 1).encode("ascii"))
    buffer.write(b" /Root 1 0 R >>\nstartxref\n")
    buffer.write(str(xref_pos).encode("ascii"))
    buffer.write(b"\n%%EOF")

    return buffer.getvalue()


def generate_report_pdf(
    metrics: dict,
    disease_records: list[dict],
    quality_records: list[dict],
    *,
    start: str | None = None,
    end: str | None = None,
) -> bytes:
    start_dt = parse_iso_timestamp(start) if start else None
    end_dt = parse_iso_timestamp(end) if end else None
    disease_filtered = filter_records_by_range(disease_records, start_dt, end_dt)
    quality_filtered = filter_records_by_range(quality_records, start_dt, end_dt)

    now_local = datetime.now(timezone.utc).astimezone()
    lines = [
        'Informe consolidado de metricas',
        f"Generado: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]
    if start or end:
        lines.append('Periodo: ' + f"{start or 'inicio'} a {end or 'ahora'}")
    lines.extend([
        '',
        'Resumen general:',
        f"- Analisis totales: {metrics['summary']['analysis']['total']}",
        f"- Enfermedades detectadas: {metrics['summary']['disease']['total']} (unicas: {metrics['summary']['disease']['unique']})",
        f"- Puntaje de calidad promedio: {metrics['summary']['quality']['score']}/10",
        f"- Confianza media del modelo: {metrics['summary']['confidence']['percentage']}%",
        '',
        'Alertas destacadas:',
    ])

    alerts = metrics.get('alerts') or []
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert.get('title')}: {alert.get('message')}")
    else:
        lines.append('- Sin alertas registradas en el periodo.')

    lines.extend([
        '',
        'Ultimos analisis de enfermedad:',
    ])
    disease_recent = metrics.get('disease', {}).get('recent', [])
    if disease_recent:
        for record in disease_recent:
            lines.append(
                f"- {record.get('timestamp')} - {record.get('disease') or 'Sin etiqueta'}"
                f" ({record.get('severity') or 'Sin clasificacion'})"
            )
    else:
        lines.append('- No hay registros disponibles.')

    lines.extend([
        '',
        'Ultimos analisis de calidad:',
    ])
    quality_recent = metrics.get('quality', {}).get('recent', [])
    if quality_recent:
        for record in quality_recent:
            label = record.get('label') or 'Sin clasificacion'
            lines.append(
                f"- {record.get('timestamp')} - {record.get('batchId') or 'Lote sin nombre'}"
                f" ({label})"
            )
    else:
        lines.append('- No hay registros disponibles.')

    lines.extend([
        '',
        'Resumen de distribuciones:',
    ])
    severity_breakdown = metrics.get('disease', {}).get('severityBreakdown', {})
    for key, value in severity_breakdown.items():
        lines.append(f"- Severidad {key}: {value} casos")

    class_distribution = metrics.get('quality', {}).get('classDistribution', {})
    if class_distribution:
        for key, value in class_distribution.items():
            lines.append(f"- {key}: {value}% del total de detecciones")

    lines.append('')
    lines.append('Observaciones registradas:')
    notes = [
        record.get('notes')
        for record in disease_filtered + quality_filtered
        if record.get('notes')
    ]
    if notes:
        for idx, note in enumerate(notes[:5], 1):
            lines.append(f"- Nota {idx}: {note}")
        if len(notes) > 5:
            lines.append(f"- ... {len(notes) - 5} notas adicionales")
    else:
        lines.append('- No se registraron notas en este periodo.')

    return build_simple_pdf(lines)
@app.route("/records/history", methods=["GET"])
def list_combined_history():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id es obligatorio"}), 400

    record_type = (request.args.get("type") or "all").lower()
    severity_filter = (request.args.get("severity") or "all").lower()
    search_term = (request.args.get("search") or "").strip().lower()
    start = request.args.get("start")
    end = request.args.get("end")
    limit = request.args.get("limit", type=int)

    disease_records, quality_records = fetch_user_records(user_id)
    start_dt = parse_iso_timestamp(start) if start else None
    end_dt = parse_iso_timestamp(end) if end else None

    combined = []

    for record in filter_records_by_range(disease_records, start_dt, end_dt):
        severity = (record.get("severity") or "").lower()
        if record_type not in ("all", "disease"):
            continue
        if severity_filter not in ("all", "") and not severity.startswith(severity_filter):
            continue
        if search_term:
            haystack = " ".join(
                filter(
                    None,
                    [
                        record.get("sampleId"),
                        record.get("disease"),
                        record.get("notes"),
                        record.get("location"),
                    ],
                )
            ).lower()
            if search_term not in haystack:
                continue
        combined.append(
            {
                "recordType": "disease",
                "id": record.get("id"),
                "timestamp": record.get("timestamp"),
                "severity": record.get("severity"),
                "title": record.get("disease"),
                "identifier": record.get("sampleId"),
                "notes": record.get("notes"),
                "meta": {
                    "confidence": record.get("confidence"),
                    "location": record.get("location"),
                },
            }
        )

    for record in filter_records_by_range(quality_records, start_dt, end_dt):
        if record_type not in ("all", "quality"):
            continue
        if severity_filter not in ("all", ""):
            continue
        class_counts = record.get("classCounts") or {}
        dominant_key = max(class_counts.items(), key=lambda item: item[1], default=(None, 0))[0]
        if search_term:
            haystack = " ".join(
                filter(
                    None,
                    [
                        record.get("batchId"),
                        record.get("notes"),
                        dominant_key,
                    ],
                )
            ).lower()
            if search_term not in haystack:
                continue
        combined.append(
            {
                "recordType": "quality",
                "id": record.get("id"),
                "timestamp": record.get("timestamp"),
                "severity": None,
                "title": dominant_key,
                "identifier": record.get("batchId"),
                "notes": record.get("notes"),
                "meta": {
                    "totalDetections": record.get("total"),
                    "classCounts": class_counts,
                },
            }
        )

    combined.sort(
        key=lambda item: parse_iso_timestamp(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    if limit is not None and limit > 0:
        combined = combined[:limit]

    return jsonify({"records": combined}), 200





@app.route("/records/disease", methods=["POST"])
def create_disease_record():
    payload = request.get_json(force=True) or {}

    user_id = payload.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id es obligatorio"}), 400

    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"error": "user_id debe ser numerico"}), 400

    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    timestamp = (payload.get("timestamp") or datetime.utcnow().isoformat()).strip()
    sample_id = (payload.get("sampleId") or "").strip() or None
    location = (payload.get("location") or "").strip() or None
    notes = (payload.get("notes") or "").strip() or None
    disease = (payload.get("disease") or "").strip() or "Desconocido"
    severity = (payload.get("severity") or "").strip()
    symptoms = (payload.get("symptoms") or "").strip() or None
    treatment = (payload.get("treatment") or "").strip() or None
    prevention = (payload.get("prevention") or "").strip() or None

    conn = get_db()
    try:
        cursor = conn.execute(
            """
            INSERT INTO disease_analyses (
                user_id, timestamp, sample_id, location, notes, disease, confidence, severity, symptoms, treatment, prevention
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id_int,
                timestamp,
                sample_id,
                location,
                notes,
                disease,
                confidence_value,
                severity,
                symptoms,
                treatment,
                prevention,
            ),
        )
        conn.commit()
        new_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM disease_analyses WHERE id = ?", (new_id,)).fetchone()
    finally:
        conn.close()

    return jsonify({"record": serialize_disease_row(row)}), 201


@app.route("/records/quality", methods=["POST"])
def create_quality_record():
    payload = request.get_json(force=True) or {}

    user_id = payload.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id es obligatorio"}), 400

    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"error": "user_id debe ser numerico"}), 400

    timestamp = (payload.get("timestamp") or datetime.utcnow().isoformat()).strip()
    batch_id = (payload.get("batchId") or "").strip() or None
    notes = (payload.get("notes") or "").strip() or None
    annotated_image = payload.get("annotatedImage") or None

    detections = payload.get("detections") or []
    class_counts = payload.get("classCounts") or {}

    try:
        total = int(payload.get("total") or 0)
    except (TypeError, ValueError):
        total = 0

    detections_json = json.dumps(detections, ensure_ascii=False)
    class_counts_json = json.dumps(class_counts, ensure_ascii=False)

    conn = get_db()
    try:
        cursor = conn.execute(
            """
            INSERT INTO quality_analyses (
                user_id, timestamp, batch_id, notes, annotated_image, detections_json, total, class_counts_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id_int,
                timestamp,
                batch_id,
                notes,
                annotated_image,
                detections_json,
                total,
                class_counts_json,
            ),
        )
        conn.commit()
        new_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM quality_analyses WHERE id = ?", (new_id,)).fetchone()
    finally:
        conn.close()

    return jsonify({"record": serialize_quality_row(row)}), 201


@app.route("/records/disease/<int:record_id>", methods=["PATCH"])
def update_disease_record(record_id: int):
    payload = request.get_json(force=True)
    user_id = payload.get("user_id")
    notes = payload.get("notes")

    if not user_id:
        return jsonify({"error": "user_id es obligatorio"}), 400
    if notes is None:
        return jsonify({"error": "notes es obligatorio"}), 400

    conn = get_db()
    try:
        cursor = conn.execute(
            "UPDATE disease_analyses SET notes = ? WHERE id = ? AND user_id = ?",
            (notes, record_id, user_id),
        )
        if cursor.rowcount == 0:
            return jsonify({"error": "Registro no encontrado"}), 404
        conn.commit()
        row = conn.execute(
            "SELECT * FROM disease_analyses WHERE id = ?",
            (record_id,),
        ).fetchone()
    finally:
        conn.close()

    return jsonify({"record": serialize_disease_row(row)}), 200


@app.route("/records/quality/<int:record_id>", methods=["PATCH"])
def update_quality_record(record_id: int):
    payload = request.get_json(force=True)
    user_id = payload.get("user_id")
    notes = payload.get("notes")

    if not user_id:
        return jsonify({"error": "user_id es obligatorio"}), 400
    if notes is None:
        return jsonify({"error": "notes es obligatorio"}), 400

    conn = get_db()
    try:
        cursor = conn.execute(
            "UPDATE quality_analyses SET notes = ? WHERE id = ? AND user_id = ?",
            (notes, record_id, user_id),
        )
        if cursor.rowcount == 0:
            return jsonify({"error": "Registro no encontrado"}), 404
        conn.commit()
        row = conn.execute(
            "SELECT * FROM quality_analyses WHERE id = ?",
            (record_id,),
        ).fetchone()
    finally:
        conn.close()

    return jsonify({"record": serialize_quality_row(row)}), 200


@app.route("/reports/export", methods=["GET"])
def export_report():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id es obligatorio"}), 400

    fmt = (request.args.get("format") or "csv").lower()
    start = request.args.get("start")
    end = request.args.get("end")

    disease_records, quality_records = fetch_user_records(user_id)
    metrics = calculate_dashboard_metrics(disease_records, quality_records, start=start, end=end)

    if fmt == "csv":
        payload = generate_history_csv(disease_records, quality_records, start=start, end=end)
        response = make_response(payload)
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = "attachment; filename=metricas.csv"
        return response

    if fmt == "pdf":
        payload = generate_report_pdf(metrics, disease_records, quality_records, start=start, end=end)
        response = make_response(payload)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=metricas.pdf"
        return response

    return jsonify({"error": "Formato no soportado"}), 400


# ---------- Predictor endpoints ----------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'disease_model': disease_predictor is not None,
        'quality_model': quality_predictor is not None
    })

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        if disease_predictor is None:
            return jsonify({'error': 'X No se pudo establecer conexion con el modelo de enfermedades'}), 500

        image_file = request.files['image']
        result = disease_predictor.predict(image_file)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': f'X Error en prediccion de enfermedades: {str(exc)}'}), 500

@app.route('/predict/quality', methods=['POST'])
def predict_quality():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        if quality_predictor is None:
            return jsonify({'error': 'X No se pudo establecer conexion con el modelo YOLO de calidad'}), 500

        image_file = request.files['image']
        result = quality_predictor.predict(image_file)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': f'X Error en prediccion de calidad: {str(exc)}'}), 500

@app.route("/auth/register", methods=["POST", "OPTIONS"])
def register_user():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    password_hash = generate_password_hash(password)
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "username already exists"}), 409
    finally:
        conn.close()

    return jsonify({"message": "user registered"}), 201


@app.route("/auth/login", methods=["POST", "OPTIONS"])
def login_user():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    conn = get_db()
    try:
        row = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid credentials"}), 401

    return jsonify({"user": {"id": row["id"], "username": row["username"]}}), 200


if __name__ == '__main__':
    init_db()
    initialize_predictors()
    print("🚀 Starting Flask server on Render...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
