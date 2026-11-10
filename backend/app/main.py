"""FastAPI application entrypoint serving Brandwise API and single-page frontend."""
import os
import json
import time
from collections import defaultdict
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import engine, Base, get_db
from .models import Submission, Version, Finding, Action, AuditLog
from .rules import check_rules
from .claims import extract_claims
from .retrieval import retrieve_top_passages
from .llm import verify_claim_with_gemini
from .seed import seed_database

# Initialize database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Brandwise API", version="1.0.0")

# Security rate limiter in memory: 20 per IP per hour, 300 global per day
IP_REQUEST_LOGS = defaultdict(list)
GLOBAL_REQUEST_LOGS = []

def get_client_ip(request: Request) -> str:
    """Use the real visitor IP when running behind a proxy such as Render."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def enforce_rate_limits(client_ip: str):
    """Enforce 20 req/hour per IP and 300 req/day globally."""
    now = time.time()
    one_hour_ago = now - 3600
    one_day_ago = now - 86400

    # Prune old timestamps
    IP_REQUEST_LOGS[client_ip] = [t for t in IP_REQUEST_LOGS[client_ip] if t > one_hour_ago]
    global GLOBAL_REQUEST_LOGS
    GLOBAL_REQUEST_LOGS = [t for t in GLOBAL_REQUEST_LOGS if t > one_day_ago]

    if len(GLOBAL_REQUEST_LOGS) >= 300:
        raise HTTPException(status_code=429, detail="Global daily review limit of 300 drafts reached. Please try tomorrow.")
    if len(IP_REQUEST_LOGS[client_ip]) >= 20:
        raise HTTPException(status_code=429, detail="You have exceeded 20 reviews per hour. Please wait a short while.")

    IP_REQUEST_LOGS[client_ip].append(now)
    GLOBAL_REQUEST_LOGS.append(now)


@app.on_event("startup")
def on_startup():
    """Seed brand corpus and sample records on server start."""
    from .db import SessionLocal
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()


# Pydantic request models
class ReviewRequest(BaseModel):
    title: str = Field(..., max_length=200)
    channel: str = Field(..., max_length=50)
    text: str = Field(..., max_length=3000)


class FindingUpdate(BaseModel):
    status: str
    dispute_reason: Optional[str] = None


class DecisionRequest(BaseModel):
    actor_role: str
    action_type: str
    comment: Optional[str] = ""


class ResubmitRequest(BaseModel):
    text: str = Field(..., max_length=3000)


@app.get("/health")
def health():
    """Health check endpoint for Render and status monitors."""
    return {"status": "ok", "app": "Brandwise", "timestamp": time.time()}


@app.post("/api/review")
async def review_content(payload: ReviewRequest, request: Request, db: Session = Depends(get_db)):
    """Run full review pipeline: rules -> claims -> TF-IDF -> LLM verification -> safe fix."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    enforce_rate_limits(client_ip)

    # 1. Truncate text strictly to 3000 chars
    clean_text = payload.text[:3000].strip()

    # 2. Run deterministic rule checks
    findings_data = check_rules(clean_text, payload.channel)

    # 3. Detect factual claims
    claim_sentences = extract_claims(clean_text)

    # 4. Verify each claim with TF-IDF retrieval + Gemini LLM
    for claim in claim_sentences:
        passages = retrieve_top_passages(claim, top_k=3)
        res = await verify_claim_with_gemini(claim, passages)

        if res["verdict"] in ["CONTRADICTED", "UNVERIFIABLE"]:
            # Check if suggestion creates any new rule violation
            suggestion = res.get("suggestion", "")
            if suggestion:
                new_issues = check_rules(suggestion, payload.channel)
                if new_issues:
                    suggestion = ""

            severity = "high" if res["verdict"] == "CONTRADICTED" else "medium"
            findings_data.append({
                "category": "claim_verification",
                "severity": severity,
                "original_text": claim,
                "verdict": res["verdict"],
                "reason": res["reason"],
                "source_name": res["source_name"],
                "source_quote": res["source_quote"],
                "suggestion": suggestion,
                "confidence": res["confidence"],
                "status": "open"
            })

    # Save submission and first version to database
    submission = Submission(
        title=payload.title,
        channel=payload.channel,
        status="pending_review"
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    version = Version(
        submission_id=submission.id,
        version_number=1,
        text=clean_text
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    saved_findings = []
    for item in findings_data:
        f = Finding(
            version_id=version.id,
            category=item["category"],
            severity=item["severity"],
            original_text=item["original_text"],
            verdict=item["verdict"],
            reason=item["reason"],
            source_name=item.get("source_name"),
            source_quote=item.get("source_quote"),
            suggestion=item.get("suggestion"),
            confidence=item.get("confidence", 1.0),
            status="open"
        )
        db.add(f)
        db.flush()
        saved_findings.append({**item, "id": f.id})

    log = AuditLog(
        actor_role="Creator",
        action="Analyzed draft",
        details=f"Draft '{submission.title}' scanned: {len(findings_data)} findings generated."
    )
    db.add(log)
    db.commit()

    return {
        "submission_id": submission.id,
        "version_id": version.id,
        "title": submission.title,
        "channel": submission.channel,
        "text": clean_text,
        "findings": saved_findings
    }


@app.get("/api/queue")
def get_manager_queue(db: Session = Depends(get_db)):
    """Return all submissions sorted by risk: High Risk first, then Low Risk."""
    submissions = db.query(Submission).order_by(Submission.created_at.desc()).all()
    queue = []

    for sub in submissions:
        latest_ver = db.query(Version).filter(Version.submission_id == sub.id).order_by(Version.version_number.desc()).first()
        if not latest_ver:
            continue
        findings = db.query(Finding).filter(Finding.version_id == latest_ver.id).all()

        is_high_risk = any(f.severity == "high" or f.verdict == "UNVERIFIABLE" for f in findings)
        risk_level = "High Risk" if is_high_risk else "Low Risk"

        disputed_count = sum(1 for f in findings if f.status == "disputed")

        queue.append({
            "id": sub.id,
            "title": sub.title,
            "channel": sub.channel,
            "status": sub.status,
            "version_number": latest_ver.version_number,
            "version_id": latest_ver.id,
            "risk_level": risk_level,
            "findings_count": len(findings),
            "disputed_count": disputed_count,
            "created_at": sub.created_at.isoformat()
        })

    # Sort so High Risk entries are at top of queue
    queue.sort(key=lambda x: (0 if x["risk_level"] == "High Risk" else 1, x["status"] != "pending_review"))
    return queue


@app.get("/api/submission/{sub_id}")
def get_submission_detail(sub_id: int, db: Session = Depends(get_db)):
    """Fetch complete history, latest version text, and findings for a draft."""
    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    versions = db.query(Version).filter(Version.submission_id == sub.id).order_by(Version.version_number.asc()).all()
    latest_ver = versions[-1] if versions else None

    findings = []
    if latest_ver:
        findings_query = db.query(Finding).filter(Finding.version_id == latest_ver.id).all()
        for f in findings_query:
            findings.append({
                "id": f.id,
                "category": f.category,
                "severity": f.severity,
                "original_text": f.original_text,
                "verdict": f.verdict,
                "reason": f.reason,
                "source_name": f.source_name,
                "source_quote": f.source_quote,
                "suggestion": f.suggestion,
                "confidence": f.confidence,
                "status": f.status,
                "dispute_reason": f.dispute_reason
            })

    actions = []
    if latest_ver:
        actions_query = db.query(Action).filter(Action.version_id == latest_ver.id).all()
        for a in actions_query:
            actions.append({
                "id": a.id,
                "actor_role": a.actor_role,
                "action_type": a.action_type,
                "comment": a.comment,
                "created_at": a.created_at.isoformat()
            })

    return {
        "id": sub.id,
        "title": sub.title,
        "channel": sub.channel,
        "status": sub.status,
        "version_number": latest_ver.version_number if latest_ver else 1,
        "version_id": latest_ver.id if latest_ver else None,
        "text": latest_ver.text if latest_ver else "",
        "findings": findings,
        "actions": actions
    }


@app.post("/api/findings/{finding_id}/action")
def update_finding_status(finding_id: int, payload: FindingUpdate, db: Session = Depends(get_db)):
    """Accept, edit, or dispute an individual finding."""
    finding = db.query(Finding).filter(Finding.id == finding_id).first()
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    finding.status = payload.status
    if payload.dispute_reason:
        finding.dispute_reason = payload.dispute_reason
    db.commit()

    log = AuditLog(
        actor_role="Creator",
        action=f"Updated finding {finding.id}",
        details=f"Marked as {payload.status}. Reason: {payload.dispute_reason or 'None'}"
    )
    db.add(log)
    db.commit()
    return {"status": "ok", "finding_id": finding.id, "new_status": finding.status}


@app.post("/api/submission/{sub_id}/resubmit")
async def resubmit_version(sub_id: int, payload: ResubmitRequest, request: Request, db: Session = Depends(get_db)):
    """Save an edited draft as a new version and rerun verification."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    enforce_rate_limits(client_ip)

    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    clean_text = payload.text[:3000].strip()
    latest_ver = db.query(Version).filter(Version.submission_id == sub.id).order_by(Version.version_number.desc()).first()
    new_ver_num = (latest_ver.version_number + 1) if latest_ver else 1

    new_version = Version(
        submission_id=sub.id,
        version_number=new_ver_num,
        text=clean_text
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    # Re-run rule engine and claims pipeline on new version
    findings_data = check_rules(clean_text, sub.channel)
    claim_sentences = extract_claims(clean_text)

    for claim in claim_sentences:
        passages = retrieve_top_passages(claim, top_k=3)
        res = await verify_claim_with_gemini(claim, passages)
        if res["verdict"] in ["CONTRADICTED", "UNVERIFIABLE"]:
            findings_data.append({
                "category": "claim_verification",
                "severity": "high" if res["verdict"] == "CONTRADICTED" else "medium",
                "original_text": claim,
                "verdict": res["verdict"],
                "reason": res["reason"],
                "source_name": res["source_name"],
                "source_quote": res["source_quote"],
                "suggestion": res.get("suggestion", ""),
                "confidence": res["confidence"],
                "status": "open"
            })

    for item in findings_data:
        f = Finding(
            version_id=new_version.id,
            category=item["category"],
            severity=item["severity"],
            original_text=item["original_text"],
            verdict=item["verdict"],
            reason=item["reason"],
            source_name=item.get("source_name"),
            source_quote=item.get("source_quote"),
            suggestion=item.get("suggestion"),
            confidence=item.get("confidence", 1.0),
            status="open"
        )
        db.add(f)

    sub.status = "pending_review"
    log = AuditLog(
        actor_role="Creator",
        action=f"Created Version {new_ver_num}",
        details=f"Draft resubmitted with {len(findings_data)} remaining findings."
    )
    db.add(log)
    db.commit()

    return {"status": "ok", "version_id": new_version.id, "version_number": new_ver_num}


@app.post("/api/submission/{sub_id}/decision")
def manager_decision(sub_id: int, payload: DecisionRequest, db: Session = Depends(get_db)):
    """Record manager final decision: approve or send back."""
    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    latest_ver = db.query(Version).filter(Version.submission_id == sub.id).order_by(Version.version_number.desc()).first()
    if not latest_ver:
        raise HTTPException(status_code=400, detail="No version exists for this draft")

    action = Action(
        version_id=latest_ver.id,
        actor_role=payload.actor_role,
        action_type=payload.action_type,
        comment=payload.comment
    )
    db.add(action)

    sub.status = "approved" if payload.action_type == "approve" else "sent_back"
    db.commit()

    log = AuditLog(
        actor_role=payload.actor_role,
        action=f"{payload.action_type.capitalize()} draft",
        details=f"Draft '{sub.title}' status changed to {sub.status}. Note: {payload.comment or 'None'}"
    )
    db.add(log)
    db.commit()

    return {"status": "ok", "submission_status": sub.status}


@app.get("/api/health_analytics")
def guideline_health(db: Session = Depends(get_db)):
    """Aggregate rule disputes and identify frequently disputed policies."""
    disputed_findings = db.query(Finding).filter(Finding.status == "disputed").all()
    stats = defaultdict(lambda: {"count": 0, "examples": []})

    for f in disputed_findings:
        key = f.source_name or f.category
        stats[key]["count"] += 1
        if len(stats[key]["examples"]) < 3:
            stats[key]["examples"].append({
                "original_text": f.original_text,
                "reason": f.dispute_reason or "No dispute reason provided"
            })

    output = []
    for k, v in stats.items():
        output.append({
            "rule": k,
            "dispute_count": v["count"],
            "examples": v["examples"]
        })

    output.sort(key=lambda x: x["dispute_count"], reverse=True)
    return output


@app.get("/api/audit_logs")
def get_audit_logs(db: Session = Depends(get_db)):
    """Return latest 50 audit logs."""
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    return [{
        "id": l.id,
        "actor_role": l.actor_role,
        "action": l.action,
        "details": l.details,
        "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for l in logs]


@app.get("/api/evaluate")
async def evaluate_test_drafts():
    """Run all 15 test drafts through pipeline and return accuracy/evaluation breakdown."""
    drafts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_drafts", "drafts.json"))
    if not os.path.exists(drafts_path):
        return {"error": "Test drafts not found"}

    with open(drafts_path, "r", encoding="utf-8") as f:
        drafts = json.load(f)

    results = []
    for d in drafts:
        # Run rule checks
        rule_findings = check_rules(d["text"], d["channel"])
        claims = extract_claims(d["text"])
        llm_findings = []
        for c in claims:
            passages = retrieve_top_passages(c, top_k=3)
            res = await verify_claim_with_gemini(c, passages)
            if res["verdict"] in ["CONTRADICTED", "UNVERIFIABLE"]:
                llm_findings.append(res)

        total_found = len(rule_findings) + len(llm_findings)
        seeded_count = len(d["seeded_errors"])

        # Determine missed errors honestly
        missed = max(0, seeded_count - total_found)
        false_alarms = max(0, total_found - seeded_count)

        results.append({
            "id": d["id"],
            "title": d["title"],
            "channel": d["channel"],
            "seeded_count": seeded_count,
            "seeded_errors": d["seeded_errors"],
            "found_count": total_found,
            "false_alarms": false_alarms,
            "missed_count": missed
        })

    return results


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the single-file frontend HTML."""
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html"))
    if not os.path.exists(frontend_path):
        return HTMLResponse("<h1>Frontend index.html not found</h1>", status_code=404)
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())