"""Flask application for the AI cybersecurity voice agent with fraud detection and RAG."""

from __future__ import annotations

import ipaddress
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path FIRST
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import threading
import queue
import time
import random
import json
from flask import Response

# ============================================
# RAG + LLM IMPORTS
# ============================================
try:
    from app.llm.rag_engine import RAGEngine
    from app.llm.intent_parser import IntentParser
    from app.orchestration.agent import SecurityAgent
    from app.orchestration.actions import ActionExecutor
    AI_AGENT_AVAILABLE = True
except ImportError as e:
    AI_AGENT_AVAILABLE = False
    print(f"⚠️ AI Agent components not available: {e}")

from app.asr.transcriber import ASREngine
from app.tts.speaker import TTSEngine
from app.fraud_detector import FraudDetector

load_dotenv(PROJECT_ROOT / ".env")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "secure.log"
MODEL_DIR = PROJECT_ROOT / "models"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

asr = ASREngine(model_name=os.getenv("ASR_MODEL", "base"), language=os.getenv("ASR_LANGUAGE", "en-US"))
tts = TTSEngine(
    rate=int(os.getenv("TTS_RATE", "175")),
    volume=float(os.getenv("TTS_VOLUME", "0.9")),
    language=os.getenv("TTS_LANGUAGE", "en"),
)

# ============================================
# FRAUD DETECTION INITIALIZATION
# ============================================
fraud_detector = FraudDetector()
fraud_available = fraud_detector.available

command_history: List[Dict[str, Any]] = []

# --- Live alerts queue (used by optional transaction simulator) ---
alert_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ============================================
# RAG + LLM INITIALIZATION
# ============================================
rag = None
intent_parser = None
action_executor = None
agent = None
ai_agent_ready = False

if AI_AGENT_AVAILABLE:
    try:
        print("🧠 Initializing RAG + LLM components...")
        
        rag = RAGEngine(persist_directory="./data/vector_store")
        rag.load_logs("./data/logs")
        
        intent_parser = IntentParser(use_local=True, model_name='llama3.2')
        action_executor = ActionExecutor(log_file="./data/logs/secure.log")
        agent = SecurityAgent(rag, intent_parser, action_executor)
        ai_agent_ready = True
        print("✅ Security Agent initialized successfully")
    except Exception as e:
        print(f"⚠️ AI Agent initialization error: {e}")
        ai_agent_ready = False

# ============================================
# FRAUD DETECTION FUNCTIONS
# ============================================

def check_transaction_fraud(amount: float, transaction_type: int, old_balance: float, new_balance: float) -> Dict[str, Any]:
    """
    Check if a transaction is fraudulent using the calibrated fraud pipeline.
    """
    if not fraud_available:
        return {
            "is_fraud": False,
            "probability": 0.0,
            "risk_level": "UNAVAILABLE",
            "message": "⚠️ Fraud detection unavailable - model not loaded"
        }
    
    return fraud_detector.check_transaction(amount, transaction_type, old_balance, new_balance)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _read_secure_events() -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    events: List[Dict[str, Any]] = []
    line_pattern = re.compile(
        r"^(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+(?P<service>[^:]+):\s+(?P<message>.+)$"
    )

    for raw_line in LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = line_pattern.match(line)
        if match is None:
            continue

        message = match.group("message")
        source_match = re.search(r"from\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})", message)
        source_ip = source_match.group("ip") if source_match else None

        event_type = "other"
        if re.search(r"\bFailed\b|\bfailure\b", message, flags=re.IGNORECASE):
            event_type = "failed"
        elif re.search(r"\bAccepted\b|\bsuccess\b", message, flags=re.IGNORECASE):
            event_type = "successful"

        parsed_at = datetime.strptime(
            f"{datetime.utcnow().year} {match.group('month')} {match.group('day')} {match.group('time')}",
            "%Y %b %d %H:%M:%S",
        )

        events.append(
            {
                "timestamp": parsed_at,
                "host": match.group("host"),
                "service": match.group("service"),
                "message": message,
                "event_type": event_type,
                "source_ip": source_ip,
                "raw": line,
            }
        )

    events.sort(key=lambda event: event["timestamp"])
    return events


def _is_internal_ip(ip_value: str) -> bool:
    try:
        address = ipaddress.ip_address(ip_value)
        return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
    except ValueError:
        return False


def _extract_ip(text: str) -> Optional[str]:
    match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    return match.group(0) if match else None


def _extract_amount(text: str) -> Optional[float]:
    match = re.search(r"\b(\d+[\.,]?\d*)\b", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def _build_failed_login_stats() -> str:
    events = _read_secure_events()
    failed_events = [event for event in events if event["event_type"] == "failed"]
    if not failed_events:
        return "No failed login attempts were found in data/logs/secure.log."

    source_counts = Counter(event["source_ip"] for event in failed_events if event["source_ip"])
    top_sources = source_counts.most_common(3)

    lines = [
        "Failed login statistics:",
        f"- Failed SSH attempts: {len(failed_events)}",
    ]

    if top_sources:
        lines.append("- Top source IPs:")
        for ip_value, count in top_sources:
            lines.append(f"  - {ip_value}: {count} attempts")

    last_failure = failed_events[-1]
    lines.append(f"- Latest failure observed on {last_failure['timestamp'].strftime('%b %d %H:%M:%S')}")
    return "\n".join(lines)


def _build_activity_summary() -> str:
    events = _read_secure_events()
    if not events:
        return "No activity data is available in data/logs/secure.log."

    failed_events = [event for event in events if event["event_type"] == "failed"]
    successful_events = [event for event in events if event["event_type"] == "successful"]
    unique_ips = sorted({event["source_ip"] for event in events if event["source_ip"]})
    internal_successes = [event for event in successful_events if event["source_ip"] and _is_internal_ip(event["source_ip"])]

    summary = [
        "Activity summary:",
        f"- Total log entries: {len(events)}",
        f"- Failed SSH attempts: {len(failed_events)}",
        f"- Successful logins: {len(successful_events)}",
        f"- Internal successful logins: {len(internal_successes)}",
        f"- Unique source IPs: {len(unique_ips)}",
    ]

    if failed_events:
        attacker_counts = Counter(event["source_ip"] for event in failed_events if event["source_ip"])
        if attacker_counts:
            top_attacker, top_count = attacker_counts.most_common(1)[0]
            summary.append(f"- Most active source: {top_attacker} with {top_count} failed attempts")

    summary.append(f"- Latest event: {events[-1]['timestamp'].strftime('%b %d %H:%M:%S')} - {events[-1]['message']}")
    return "\n".join(summary)


def _enrich_ip(ip_value: str) -> str:
    if not ip_value:
        return "Please provide a valid IP address."

    try:
        address = ipaddress.ip_address(ip_value)
    except ValueError:
        return f"{ip_value} is not a valid IP address."

    events = _read_secure_events()
    matched_events = [event for event in events if event["source_ip"] == ip_value]
    lines = [f"IP enrichment for {ip_value}:"]

    if address.is_private or address.is_loopback or address.is_link_local:
        lines.append("- Classification: internal/private address")
    else:
        lines.append("- Classification: public address")
        try:
            response = requests.get(f"https://ipapi.co/{ip_value}/json/", timeout=3)
            if response.ok:
                data = response.json()
                country = data.get("country_name") or data.get("country")
                city = data.get("city")
                org = data.get("org") or data.get("asn_org")
                if country:
                    lines.append(f"- Country: {country}")
                if city:
                    lines.append(f"- City: {city}")
                if org:
                    lines.append(f"- Organization: {org}")
        except Exception:
            lines.append("- External enrichment unavailable; using local observations only")

    if matched_events:
        failed_count = sum(1 for event in matched_events if event["event_type"] == "failed")
        success_count = sum(1 for event in matched_events if event["event_type"] == "successful")
        lines.append(f"- Observed in secure.log: {len(matched_events)} events")
        lines.append(f"- Failed attempts from this IP: {failed_count}")
        lines.append(f"- Successful logins from this IP: {success_count}")
    else:
        lines.append("- No local log entries reference this IP")

    if matched_events and sum(1 for event in matched_events if event["event_type"] == "failed") >= 2:
        lines.append("- Risk: elevated due to repeated failed logins")
    elif address.is_private:
        lines.append("- Risk: low for internal monitoring")
    else:
        lines.append("- Risk: review context before taking action")

    return "\n".join(lines)


def _append_history(entry_type: str, command: str, response: str, transcript: Optional[str] = None) -> None:
    command_history.append(
        {
            "id": len(command_history) + 1,
            "timestamp": _utc_timestamp(),
            "type": entry_type,
            "command": command,
            "transcript": transcript,
            "response": response,
        }
    )


# --- LIVE TRANSACTION GENERATOR + MONITOR (optional) ---
def _generate_live_transaction() -> Dict[str, Any]:
    """Create a synthetic transaction dict."""
    old_balance = round(random.uniform(1000, 50000), 2)
    suspicious = random.random() < 0.35
    transaction_type = random.choice([1, 1, 2]) if suspicious else 1
    amount = old_balance if suspicious else round(random.uniform(5, old_balance * 0.35), 2)
    tx = {
        "id": random.randint(100000, 999999),
        "amount": amount,
        "old_balance": old_balance,
        "transaction_type": transaction_type,
        "location": random.choice(["New York", "London", "Berlin", "Nairobi", "Tokyo"]),
        "device": random.choice(["iPhone 14", "Samsung Galaxy", "Chrome Browser", "Old Laptop"]),
        "is_new_device": random.random() < 0.25,
        "hour": random.randint(0, 23),
        "status": "PENDING",
    }
    return tx


def _fraud_monitor_loop(poll_interval: float = 2.5) -> None:
    """Background thread that publishes simulated transactions for browser alerts."""
    print("▶️ Fraud Monitor Thread starting (simulator)...")
    while True:
        try:
            time.sleep(poll_interval)
            tx = _generate_live_transaction()
            # Use existing fraud checker
            try:
                result = check_transaction_fraud(
                    tx["amount"],
                    tx["transaction_type"],
                    tx["old_balance"],
                    tx["old_balance"] - tx["amount"],
                )
            except Exception:
                result = {"is_fraud": False, "probability": 0.0, "risk_level": "UNKNOWN"}

            tx.update({
                "is_fraud": result.get("is_fraud", False),
                "risk": result.get("probability", 0.0),
                "risk_level": result.get("risk_level", "UNKNOWN"),
                "message": result.get("message"),
            })
            alert_queue.put(tx)
            print(f"📊 Enqueued transaction: {tx['id']} ({tx['amount']}) probability={tx['risk']:.6f}")
        except Exception as e:
            print(f"Fraud monitor error: {e}")


def start_fraud_monitor_thread() -> threading.Thread:
    t = threading.Thread(target=_fraud_monitor_loop, daemon=True)
    t.start()
    return t


# ============================================
# COMMAND PROCESSING (AI + FALLBACK)
# ============================================

def process_command(command: str) -> str:
    """
    Process command using the AI agent (LLM + RAG) if available,
    otherwise fall back to rule-based processing.
    """
    normalized = re.sub(r"\s+", " ", command.strip().lower())
    if (
        "check transaction" in normalized
        or "fraud check" in normalized
        or "is this fraud" in normalized
        or "check payment" in normalized
        or "check transfer" in normalized
        or "check cash out" in normalized
    ):
        return process_command_fallback(command)

    if ai_agent_ready and agent:
        try:
            return agent.process(command)
        except Exception as e:
            print(f"⚠️ AI agent error: {e}")
            # Fall through to rule-based
    
    # Fallback to rule-based processing
    return process_command_fallback(command)


def process_command_with_approval(command: str, approval=None) -> str:
    """Process a command with an explicit browser approval decision."""
    normalized = re.sub(r"\s+", " ", command.strip().lower())
    if (
        "check transaction" in normalized
        or "fraud check" in normalized
        or "is this fraud" in normalized
        or "check payment" in normalized
        or "check transfer" in normalized
        or "check cash out" in normalized
    ):
        return process_command_fallback(command)

    if ai_agent_ready and agent:
        try:
            return agent.process(command, approval=approval)
        except Exception as e:
            print(f"⚠️ AI agent error: {e}")
    return process_command_fallback(command)


def is_approval_request(response_text: str) -> bool:
    return response_text.startswith("Approval required before ")


def process_command_fallback(command: str) -> str:
    """Fallback rule-based command processing."""
    normalized = re.sub(r"\s+", " ", command.strip().lower())
    if not normalized:
        return "Please enter a command."

    # Fraud Detection Commands
    if "check transaction" in normalized or "fraud check" in normalized or "is this fraud" in normalized:
        amount = _extract_amount(command)
        if amount is None:
            return "Please specify an amount. Example: 'Check transaction 5000 for fraud'"
        result = check_transaction_fraud(amount, 1, 1000, 1000 - amount)
        return result['message']

    if "check payment" in normalized:
        amount = _extract_amount(command)
        if amount is None:
            return "Please specify an amount. Example: 'Check payment 100'"
        result = check_transaction_fraud(amount, 0, 1000, 1000 - amount)
        return result['message']

    if "check transfer" in normalized:
        amount = _extract_amount(command)
        if amount is None:
            return "Please specify an amount. Example: 'Check transfer 5000'"
        result = check_transaction_fraud(amount, 1, 1000, 1000 - amount)
        return result['message']

    if "check cash out" in normalized:
        amount = _extract_amount(command)
        if amount is None:
            return "Please specify an amount. Example: 'Check cash out 200'"
        result = check_transaction_fraud(amount, 2, 5000, 5000 - amount)
        return result['message']

    # Security Commands
    if re.search(r"\bshow( me)? failed logins?\b", normalized) or "failed login" in normalized:
        return _build_failed_login_stats()

    if "summarize activity" in normalized or "summarise activity" in normalized or "summary" in normalized:
        return _build_activity_summary()

    if normalized.startswith("enrich ip") or "enrich ip" in normalized:
        ip_value = _extract_ip(command)
        return _enrich_ip(ip_value) if ip_value else "Please include a valid IP address to enrich."

    if normalized.startswith("isolate host") or "isolate host" in normalized:
        ip_value = _extract_ip(command)
        target = ip_value or "the requested host"
        return (
            f"Isolation request received for {target}. "
            "This action requires approval and orchestration with endpoint controls before execution."
        )

    if normalized.startswith("block ip") or "block ip" in normalized:
        ip_value = _extract_ip(command)
        target = ip_value or "the requested IP"
        return (
            f"Blocking request received for {target}. "
            "This action requires approval and should be routed through the network control plane."
        )

    if "fraud" in normalized or "suspicious" in normalized:
        return (
            "I can help with fraud detection. Try commands like:\n"
            "- 'Check transaction 5000 for fraud'\n"
            "- 'Check payment 100'\n"
            "- 'Check transfer 5000'\n"
            "- 'Check cash out 200'"
        )

    return (
        "I understand commands like:\n"
        "- Show me failed logins\n"
        "- Summarize activity\n"
        "- Enrich IP x.x.x.x\n"
        "- Isolate host x.x.x.x\n"
        "- Block IP x.x.x.x\n"
        "- Check transaction 5000 for fraud\n"
        "- Check payment 100\n"
        "- Check transfer 5000\n"
        "- Check cash out 200"
    )


# ============================================
# ROUTES
# ============================================

@app.route("/")
def index() -> Any:
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health")
def health() -> Any:
    return jsonify(
        {
            "status": "healthy",
            "asr_available": asr.available,
            "tts_available": tts.available,
            "fraud_available": fraud_available,
            "ai_agent_available": ai_agent_ready,
            "rag_available": rag is not None and rag.collection is not None,
            "asr_backend": asr.backend,
            "tts_backend": tts.backend,
            "history_count": len(command_history),
            "timestamp": _utc_timestamp(),
            "version": "0.4.0",
        }
    )


@app.route("/api/fraud_check", methods=["POST"])
def fraud_check_endpoint() -> Any:
    try:
        data = request.get_json(silent=True) or {}
        amount = float(data.get("amount", 0))
        transaction_type = int(data.get("transaction_type", 1))
        old_balance = float(data.get("old_balance", 1000))
        new_balance = float(data.get("new_balance", 1000 - amount))
        
        result = check_transaction_fraud(amount, transaction_type, old_balance, new_balance)
        result["transaction"] = {
            "amount": amount,
            "type": transaction_type,
            "old_balance": old_balance,
            "new_balance": new_balance
        }
        return jsonify({"status": "success", "result": result})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/voice", methods=["POST"])
def process_voice() -> Any:
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"status": "error", "error": "No audio file provided"}), 400

    suffix = Path(audio_file.filename or "voice-input.webm").suffix or ".webm"
    temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    try:
        audio_file.save(str(temp_path))
        transcript = asr.transcribe_file(str(temp_path))
        if not transcript:
            return jsonify({"status": "error", "error": asr.last_error or "Could not transcribe audio"}), 400

        response_text = process_command(transcript)
        tts.speak(response_text)
        _append_history("voice", transcript, response_text, transcript=transcript)

        return jsonify({
            "status": "success",
            "transcript": transcript,
            "response": response_text,
            "approval_required": is_approval_request(response_text),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


@app.route("/api/text", methods=["POST"])
def process_text() -> Any:
    payload = request.get_json(silent=True) or {}
    command = str(payload.get("command", "")).strip()
    approval = payload.get("approval")
    if not command:
        return jsonify({"status": "error", "error": "Missing command"}), 400

    response_text = process_command_with_approval(command, approval=approval)
    _append_history("text", command, response_text)
    return jsonify({
        "status": "success",
        "command": command,
        "response": response_text,
        "approval_required": is_approval_request(response_text),
    })


@app.route("/api/history", methods=["GET"])
def get_history() -> Any:
    limit = request.args.get("limit", default=20, type=int) or 20
    history_slice = command_history[-max(limit, 1):]
    return jsonify({"status": "success", "count": len(command_history), "history": history_slice})


@app.route("/api/rag_query", methods=["POST"])
def rag_query_endpoint() -> Any:
    """REST endpoint for RAG queries."""
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        if not query:
            return jsonify({"status": "error", "error": "Missing query"}), 400
        if not rag:
            return jsonify({"status": "error", "error": "RAG not available"}), 503
        
        results = rag.retrieve(query, k=5)
        return jsonify({"status": "success", "query": query, "results": results, "count": len(results)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route('/stream')
def stream() -> Any:
    def generate():
        while True:
            try:
                if not alert_queue.empty():
                    tx = alert_queue.get()
                    yield f"data: {json.dumps(tx, ensure_ascii=False)}\n\n"
            except GeneratorExit:
                break
            except Exception:
                pass
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')


def _print_banner() -> None:
    print("=" * 68)
    print("AI Cybersecurity Voice Agent with Fraud Detection & RAG")
    print("=" * 68)
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/api/health")
    print(f"ASR: {'✅ ready' if asr.available else '❌ unavailable'} ({asr.backend})")
    print(f"TTS: {'✅ ready' if tts.available else '❌ unavailable'} ({tts.backend})")
    print(f"Fraud Detection: {'✅ loaded' if fraud_available else '❌ not loaded'}")
    print(f"AI Agent (LLM + RAG): {'✅ ready' if ai_agent_ready else '❌ not available'}")
    print(f"RAG Vector Store: {'✅ loaded' if rag and rag.collection else '❌ not loaded'}")
    print("=" * 68)
    print("\n📊 Available Commands:")
    print("  🔐 Security: Show failed logins, Summarize activity, Enrich IP, Isolate host, Block IP")
    print("  💳 Fraud: Check transaction, Check payment, Check transfer, Check cash out")
    print("  🧠 RAG: Natural language queries against logs (via AI agent)")
    print("=" * 68)


if __name__ == "__main__":
    if os.getenv("FRAUD_MONITOR_ENABLED", "0").lower() in {"1", "true", "yes"}:
        try:
            start_fraud_monitor_thread()
        except Exception as e:
            print(f"Could not start fraud monitor thread: {e}")
    else:
        print("Fraud monitor simulator disabled; set FRAUD_MONITOR_ENABLED=1 to enable it.")

    _print_banner()
    app.run(host="0.0.0.0", port=5000, debug=True)
