"""FastAPI application entrypoint serving Brandwise API and single-page frontend."""
import os
import json
import time
import asyncio
import difflib
import re
from collections import defaultdict, Counter
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import engine, Base, get_db
from .models import Submission, Version, Finding, Action, AuditLog, RuleClarification, FindingComment
from .rules import check_rules
from .claims import extract_claims
from .retrieval import retrieve_top_passages
from .llm import verify_claim_with_gemini, suggest_clarification
from .seed import seed_database

# Initialize database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Brandwise API", version="1.1.0")

MAX_TEXT_CHARS = 3000      # drafts are cut to this length
MAX_CLAIMS = 5             # at most this many claims per draft go to the AI
EVAL_CACHE_SECONDS = 1800  # Evaluation results are reused for 30 minutes

# In-memory rate limiter: 20 reviews per IP per hour, 300 per day overall
IP_REQUEST_LOGS = defaultdict(list)
GLOBAL_REQUEST_LOGS = []

# Cache and lock for the Evaluation tab
EVAL_CACHE = {"time": 0.0, "data": None}
EVAL_LOCK = asyncio.Lock()


def get_client_ip(request: Request) -> str:
    """Use the real visitor IP when running behind a proxy such as Render."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def enforce_rate_limits(client_ip: str):
    """Enforce 20 req/hour per IP and 300 req/day globally."""
    global GLOBAL_REQUEST_LOGS
    now = time.time()
    one_hour_ago = now - 3600
    one_day_ago = now - 86400

    IP_REQUEST_LOGS[client_ip] = [t for t in IP_REQUEST_LOGS[client_ip] if t > one_hour_ago]
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


# ---------- Request models ----------
class ReviewRequest(BaseModel):
    title: str = Field(..., max_length=200)
    channel: str = Field(..., max_length=50)
    text: str = Field(..., max_length=20000)


class FindingUpdate(BaseModel):
    status: str
    dispute_reason: Optional[str] = None


class DecisionRequest(BaseModel):
    actor_role: str
    action_type: str
    comment: Optional[str] = ""


class ResubmitRequest(BaseModel):
    text: str = Field(..., max_length=20000)
class CommentRequest(BaseModel):
    role: str = Field(..., max_length=32)
    text: str = Field(..., max_length=500)
class ClarifyRequest(BaseModel):
    rule: str = Field(..., max_length=255)


class ClarificationDecision(BaseModel):
    action: str  # "approve" or "dismiss"
# ---------- Shared helpers ----------
async def check_claims(text: str, channel: str) -> list:
    """Find claim sentences, verify each against the brand pack, return findings.

    Claims are checked in parallel (3 at a time) to keep reviews fast.
    A claim that matches an approved source is returned as a SUPPORTED (passed) item.
    """
    claims = extract_claims(text)[:MAX_CLAIMS]
    semaphore = asyncio.Semaphore(3)

    async def verify_one(claim: str):
        async with semaphore:
            passages = retrieve_top_passages(claim, top_k=3)
            return claim, await verify_claim_with_gemini(claim, passages)

    results = await asyncio.gather(*[verify_one(c) for c in claims])

    findings = []
    for claim, res in results:
        if res["verdict"] == "SUPPORTED":
            # A claim that matches an approved source: show it as passed
            findings.append({
                "category": "claim_verification",
                "severity": "info",
                "original_text": claim,
                "verdict": "SUPPORTED",
                "reason": res["reason"],
                "source_name": res["source_name"],
                "source_quote": res["source_quote"],
                "suggestion": "",
                "confidence": res["confidence"],
                "status": "passed"
            })
            continue

        if res["verdict"] not in ("CONTRADICTED", "UNVERIFIABLE"):
            continue

        # Discard a suggested fix if it would create a new rule violation
        suggestion = res.get("suggestion", "") or ""
        if suggestion and check_rules(suggestion, channel):
            suggestion = ""

        findings.append({
            "category": "claim_verification",
            "severity": "high" if res["verdict"] == "CONTRADICTED" else "medium",
            "original_text": claim,
            "verdict": res["verdict"],
            "reason": res["reason"],
            "source_name": res["source_name"],
            "source_quote": res["source_quote"],
            "suggestion": suggestion,
            "confidence": res["confidence"],
            "status": "open"
        })
    return findings


def save_findings(db: Session, version_id: int, findings_data: list) -> list:
    """Save findings for a version and return them with their database IDs."""
    saved = []
    for item in findings_data:
        f = Finding(
            version_id=version_id,
            category=item["category"],
            severity=item["severity"],
            original_text=item["original_text"],
            verdict=item["verdict"],
            reason=item["reason"],
            source_name=item.get("source_name"),
            source_quote=item.get("source_quote"),
            suggestion=item.get("suggestion"),
            confidence=item.get("confidence", 1.0),
            status=item.get("status", "open")
        )
        db.add(f)
        db.flush()  # assigns f.id without committing yet
        saved.append({**item, "id": f.id})
    return saved


def count_issues(findings_data: list) -> int:
    """Number of real issues, not counting claims that passed."""
    return sum(1 for f in findings_data if f.get("verdict") != "SUPPORTED")


# ---------- Endpoints ----------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return an empty response so browsers stop logging a 404 for the tab icon."""
    from fastapi.responses import Response
    return Response(status_code=204)
@app.get("/health")
def health():
    """Health check endpoint for Render and status monitors."""
    return {"status": "ok", "app": "Brandwise", "timestamp": time.time()}


@app.post("/api/review")
async def review_content(payload: ReviewRequest, request: Request, db: Session = Depends(get_db)):
    """Run full review pipeline: rules -> claims -> retrieval -> LLM verification -> safe fix."""
    enforce_rate_limits(get_client_ip(request))

    clean_text = payload.text[:MAX_TEXT_CHARS].strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Draft text is empty.")

    findings_data = check_rules(clean_text, payload.channel)
    findings_data += await check_claims(clean_text, payload.channel)

    submission = Submission(title=payload.title, channel=payload.channel, status="pending_review")
    db.add(submission)
    db.commit()
    db.refresh(submission)

    version = Version(submission_id=submission.id, version_number=1, text=clean_text)
    db.add(version)
    db.commit()
    db.refresh(version)

    saved_findings = save_findings(db, version.id, findings_data)

    db.add(AuditLog(
        actor_role="Creator",
        action="Analyzed draft",
        details=f"Draft '{submission.title}' scanned: {count_issues(saved_findings)} issues, "
                f"{len(saved_findings) - count_issues(saved_findings)} claims passed."
    ))
    db.commit()

    return {
        "submission_id": submission.id,
        "version_id": version.id,
        "title": submission.title,
        "channel": submission.channel,
        "text": clean_text,
        "findings": saved_findings
    }


def route_reasons(findings) -> list:
    """Why a draft needs a Reviewer's sign-off, in plain words. Empty means the Manager can decide."""
    reasons = []
    bad_claims = [f for f in findings
                  if f.category == "claim_verification" and f.verdict in ("CONTRADICTED", "UNVERIFIABLE")]
    if bad_claims:
        reasons.append(f"{len(bad_claims)} claim(s) contradicted or not verifiable")
    if any(f.category == "missing_disclaimer" for f in findings):
        reasons.append("a required regulatory disclaimer is missing")
    disputed = [f for f in findings if f.status == "disputed"]
    if disputed:
        reasons.append(f"{len(disputed)} finding(s) disputed by the creator")
    return reasons


@app.get("/api/queue")
def get_manager_queue(db: Session = Depends(get_db)):
    """Return all submissions, waiting items first, high risk first."""
    submissions = db.query(Submission).order_by(Submission.created_at.desc()).all()
    queue = []

    for sub in submissions:
        latest_ver = (db.query(Version).filter(Version.submission_id == sub.id)
                      .order_by(Version.version_number.desc()).first())
        if not latest_ver:
            continue
        # Claims that passed are not issues, so they do not count toward risk
        findings = [f for f in db.query(Finding).filter(Finding.version_id == latest_ver.id).all()
                    if f.verdict != "SUPPORTED"]

        is_high_risk = any(f.severity == "high" or f.verdict == "UNVERIFIABLE" for f in findings)
        disputed_count = sum(1 for f in findings if f.status == "disputed")

        queue.append({
            "id": sub.id,
            "title": sub.title,
            "channel": sub.channel,
            "status": sub.status,
            "version_number": latest_ver.version_number,
            "version_id": latest_ver.id,
            "risk_level": "High Risk" if is_high_risk else "Low Risk",
            "needs_reviewer": bool(route_reasons(findings)),
            "findings_count": len(findings),
            "disputed_count": disputed_count,
            "created_at": sub.created_at.isoformat()
        })

    waiting = ("pending_review", "awaiting_reviewer")
    queue.sort(key=lambda x: (0 if x["status"] in waiting else 1,
                              0 if x["risk_level"] == "High Risk" else 1))
    return queue


@app.get("/api/submission/{sub_id}")
def get_submission_detail(sub_id: int, db: Session = Depends(get_db)):
    """Latest version, findings, routing reasons, version list and decision history."""
    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    versions = (db.query(Version).filter(Version.submission_id == sub.id)
                .order_by(Version.version_number.asc()).all())
    latest_ver = versions[-1] if versions else None

    findings = []
    reasons = []
    if latest_ver:
        finding_rows = db.query(Finding).filter(Finding.version_id == latest_ver.id).all()
        reasons = route_reasons([f for f in finding_rows if f.verdict != "SUPPORTED"])
        for f in finding_rows:
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

    history = []
    rows = (db.query(Action, Version.version_number)
            .filter(Version.id == Action.version_id, Version.submission_id == sub.id)
            .order_by(Action.created_at).all())
    for a, ver_no in rows:
        history.append({
            "version_number": ver_no,
            "actor_role": a.actor_role,
            "action_type": a.action_type,
            "comment": a.comment,
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M")
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
        "route_reasons": reasons,
        "versions": [{"version_number": v.version_number,
                      "created_at": v.created_at.strftime("%Y-%m-%d %H:%M")} for v in versions],
        "history": history
    }


def tokenize(text: str) -> list:
    """Split text into words and spaces so a comparison keeps the original spacing."""
    return re.findall(r"\s+|\S+", text)


@app.get("/api/submission/{sub_id}/diff")
def version_diff(sub_id: int, a: int, b: int, db: Session = Depends(get_db)):
    """Compare two versions of a draft word by word."""
    va = db.query(Version).filter(Version.submission_id == sub_id, Version.version_number == a).first()
    vb = db.query(Version).filter(Version.submission_id == sub_id, Version.version_number == b).first()
    if not va or not vb:
        raise HTTPException(status_code=404, detail="Version not found")

    ta, tb = tokenize(va.text), tokenize(vb.text)
    matcher = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    segments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append({"type": "equal", "text": "".join(ta[i1:i2])})
        else:
            if i2 > i1:
                segments.append({"type": "delete", "text": "".join(ta[i1:i2])})
            if j2 > j1:
                segments.append({"type": "insert", "text": "".join(tb[j1:j2])})
    return {"from": a, "to": b, "segments": segments}


@app.get("/api/findings/{finding_id}/comments")
def get_comments(finding_id: int, db: Session = Depends(get_db)):
    """Comments on one finding, oldest first."""
    rows = (db.query(FindingComment).filter(FindingComment.finding_id == finding_id)
            .order_by(FindingComment.id).all())
    return [{"id": c.id, "role": c.actor_role, "text": c.text,
             "created_at": c.created_at.strftime("%Y-%m-%d %H:%M")} for c in rows]


@app.post("/api/findings/{finding_id}/comments")
def add_comment(finding_id: int, payload: CommentRequest, db: Session = Depends(get_db)):
    """Add a comment to a finding."""
    role = payload.role.strip().capitalize()
    if role not in ("Creator", "Manager", "Reviewer"):
        raise HTTPException(status_code=400, detail="Unknown role.")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment is empty.")

    finding = db.query(Finding).filter(Finding.id == finding_id).first()
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    db.add(FindingComment(finding_id=finding_id, actor_role=role, text=text))
    db.add(AuditLog(actor_role=role, action="Commented on finding",
                    details=f"Finding {finding_id}: {text[:120]}"))
    db.commit()
    return {"status": "ok"}


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

    db.add(AuditLog(
        actor_role="Creator",
        action=f"Updated finding {finding.id}",
        details=f"Marked as {payload.status}. Reason: {payload.dispute_reason or 'None'}"
    ))
    db.commit()
    return {"status": "ok", "finding_id": finding.id, "new_status": finding.status}


@app.post("/api/submission/{sub_id}/resubmit")
async def resubmit_version(sub_id: int, payload: ResubmitRequest, request: Request, db: Session = Depends(get_db)):
    """Save an edited draft as a new version and rerun verification."""
    enforce_rate_limits(get_client_ip(request))

    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    clean_text = payload.text[:MAX_TEXT_CHARS].strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Draft text is empty.")

    latest_ver = (db.query(Version).filter(Version.submission_id == sub.id)
                  .order_by(Version.version_number.desc()).first())
    new_ver_num = (latest_ver.version_number + 1) if latest_ver else 1

    new_version = Version(submission_id=sub.id, version_number=new_ver_num, text=clean_text)
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    findings_data = check_rules(clean_text, sub.channel)
    findings_data += await check_claims(clean_text, sub.channel)
    saved_findings = save_findings(db, new_version.id, findings_data)

    sub.status = "pending_review"
    db.add(AuditLog(
        actor_role="Creator",
        action=f"Created Version {new_ver_num}",
        details=f"Draft resubmitted with {count_issues(saved_findings)} remaining issues."
    ))
    db.commit()

    return {"status": "ok", "version_id": new_version.id, "version_number": new_ver_num}



@app.post("/api/submission/{sub_id}/decision")
def manager_decision(sub_id: int, payload: DecisionRequest, db: Session = Depends(get_db)):
    """Manager or Reviewer decides. Risky drafts go from Manager to Reviewer before approval."""
    role = payload.actor_role.strip().capitalize()
    if role not in ("Manager", "Reviewer"):
        raise HTTPException(status_code=400, detail="Only a Manager or Reviewer can decide.")
    if payload.action_type not in ("approve", "send_back"):
        raise HTTPException(status_code=400, detail="Action must be approve or send_back.")

    comment = (payload.comment or "").strip()
    if payload.action_type == "send_back" and len(comment) < 3:
        raise HTTPException(status_code=400, detail="Please add a comment explaining what should change.")

    sub = db.query(Submission).filter(Submission.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Block repeat or out-of-turn decisions
    if sub.status == "approved":
        raise HTTPException(status_code=400, detail="This draft is already approved.")
    if role == "Manager" and sub.status != "pending_review":
        raise HTTPException(status_code=400,
                            detail="This draft is not waiting for a Manager (status: %s)." % sub.status)
    if role == "Reviewer" and sub.status != "awaiting_reviewer":
        raise HTTPException(status_code=400, detail="This draft is not waiting for a Reviewer.")

    latest_ver = (db.query(Version).filter(Version.submission_id == sub.id)
                  .order_by(Version.version_number.desc()).first())
    if not latest_ver:
        raise HTTPException(status_code=400, detail="No version exists for this draft")

    findings = [f for f in db.query(Finding).filter(Finding.version_id == latest_ver.id).all()
                if f.verdict != "SUPPORTED"]
    reasons = route_reasons(findings)

    action_type = payload.action_type
    if payload.action_type == "send_back":
        new_status = "sent_back"
    elif role == "Manager" and reasons:
        new_status = "awaiting_reviewer"
        action_type = "escalate"
    else:
        new_status = "approved"

    db.add(Action(version_id=latest_ver.id, actor_role=role,
                  action_type=action_type, comment=comment))
    sub.status = new_status

    if action_type == "escalate":
        details = f"'{sub.title}' sent to a Reviewer because: {'; '.join(reasons)}."
    else:
        details = f"'{sub.title}' is now {new_status}. Note: {comment or 'None'}"
    db.add(AuditLog(actor_role=role, action=f"{action_type} decision", details=details))
    db.commit()

    return {"status": "ok", "submission_status": new_status, "routed_reasons": reasons}

def rule_key(f: Finding) -> str:
    """The name a finding is grouped under: its source rule, or its category."""
    return f.source_name if f.source_name and f.source_name != "None" else f.category


def collect_rule_stats(db: Session) -> dict:
    """Count, for every rule, how often it triggered, was disputed and was overruled."""
    findings = db.query(Finding).filter(Finding.verdict != "SUPPORTED").all()
    rows = (db.query(Version.id, Submission.status)
            .filter(Version.submission_id == Submission.id).all())
    
    version_status = {vid: st for vid, st in rows}

    stats = {}
    for f in findings:
        key = rule_key(f)
        s = stats.setdefault(key, {"triggered": 0, "disputed": 0, "overruled": 0,
                                   "quote": "", "examples": []})
        s["triggered"] += 1
        if f.source_quote and not s["quote"]:
            s["quote"] = f.source_quote
        if f.status == "disputed":
            s["disputed"] += 1
            # Overruled = the manager approved the draft despite the creator's dispute
            if version_status.get(f.version_id) == "approved":
                s["overruled"] += 1
            if len(s["examples"]) < 3:
                s["examples"].append({
                    "original_text": f.original_text,
                    "reason": f.dispute_reason or "No reason given"
                })
    return stats


def ambiguity_level(triggered: int, score: int) -> str:
    """Label a rule. Rules seen fewer than twice are not judged."""
    if triggered < 2:
        return "Not enough data"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


@app.get("/api/health_analytics")
def guideline_health(db: Session = Depends(get_db)):
    """Rank rules by how unclear they seem.

    Ambiguity score = 70% dispute rate + 30% overrule rate, as a percentage.
    Dispute rate = disputed findings / times the rule triggered.
    Overrule rate = disputes where the manager approved anyway / disputed findings.
    """
    stats = collect_rule_stats(db)

    # Latest drafted wording per rule (rows are read oldest first, so later ones win)
    clar = {c.rule: c for c in db.query(RuleClarification).order_by(RuleClarification.id).all()}

    output = []
    for key, s in stats.items():
        dispute_rate = s["disputed"] / s["triggered"]
        overrule_rate = (s["overruled"] / s["disputed"]) if s["disputed"] else 0
        score = round(100 * (0.7 * dispute_rate + 0.3 * overrule_rate))
        c = clar.get(key)
        output.append({
            "rule": key,
            "triggered": s["triggered"],
            "disputed": s["disputed"],
            "overruled": s["overruled"],
            "score": score,
            "level": ambiguity_level(s["triggered"], score),
            "rule_text": s["quote"],
            "examples": s["examples"],
            "clarification": ({"id": c.id, "draft_text": c.draft_text, "status": c.status}
                              if c else None)
        })

    output.sort(key=lambda x: (x["score"], x["triggered"]), reverse=True)
    return output


@app.post("/api/health_analytics/clarify")
async def draft_clarification(payload: ClarifyRequest, request: Request, db: Session = Depends(get_db)):
    """Draft clearer wording for an unclear rule. A person must approve it."""
    enforce_rate_limits(get_client_ip(request))

    stats = collect_rule_stats(db)
    s = stats.get(payload.rule)
    if not s:
        raise HTTPException(status_code=404, detail="Rule not found.")

    text = await suggest_clarification(payload.rule, s["quote"], s["examples"])
    if not text:
        raise HTTPException(status_code=503,
                            detail="Could not draft wording right now. Please try again in a minute.")

    c = RuleClarification(rule=payload.rule, draft_text=text, status="draft")
    db.add(c)
    db.add(AuditLog(actor_role="System", action="Drafted clearer wording",
                    details=f"Draft wording proposed for '{payload.rule}'. Awaiting a person's decision."))
    db.commit()
    db.refresh(c)
    return {"id": c.id, "draft_text": c.draft_text, "status": c.status}


@app.post("/api/health_analytics/clarification/{clar_id}/decision")
def decide_clarification(clar_id: int, payload: ClarificationDecision, db: Session = Depends(get_db)):
    """Record the brand owner's decision on drafted wording."""
    if payload.action not in ("approve", "dismiss"):
        raise HTTPException(status_code=400, detail="Action must be approve or dismiss.")

    c = db.get(RuleClarification, clar_id)
    if not c:
        raise HTTPException(status_code=404, detail="Draft wording not found.")

    c.status = "approved" if payload.action == "approve" else "dismissed"
    db.add(AuditLog(actor_role="Manager", action=f"Wording {c.status}",
                    details=f"Clearer wording for '{c.rule}' was {c.status}."))
    db.commit()
    return {"status": "ok", "new_status": c.status}


@app.post("/api/demo/disputes")
def load_demo_disputes(db: Session = Depends(get_db)):
    """Add clearly labelled sample disputes so the insights page can be demonstrated."""
    if db.query(Finding).filter(Finding.status == "disputed").first():
        return {"status": "skipped", "detail": "Disputes already exist."}

    added = 0
    approve_versions = set()
    for f in db.query(Finding).filter(Finding.status == "open").all():
        key = rule_key(f)
        if key.startswith("claims_policy.md - Section 2"):
            f.status = "disputed"
            f.dispute_reason = "Sample: this wording is about savings, not investment returns."
            added += 1
        elif f.category == "preferred_term":
            f.status = "disputed"
            f.dispute_reason = "Sample: 'cheap' is used on purpose in a student campaign."
            approve_versions.add(f.version_id)
            added += 1

    for vid in approve_versions:
        ver = db.get(Version, vid)
        sub = db.get(Submission, ver.submission_id)
        sub.status = "approved"
        db.add(Action(version_id=vid, actor_role="Manager", action_type="approve",
                      comment="Sample: approved despite the preferred-term finding."))

    db.add(AuditLog(actor_role="Demo", action="Loaded demo scenario",
                    details=f"{added} sample disputes added. Sample data for demonstration only."))
    db.commit()
    return {"status": "ok", "disputes_added": added}


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


def seeded_group(tag: str) -> str:
    """Group a seeded error tag like 'banned_word:risk-free' into a comparable type."""
    kind = tag.split(":")[0]
    if kind in ("fact_contradiction", "policy_violation"):
        return "contradiction"
    return kind


def found_group(item: dict) -> str:
    """Group a finding into the same comparable types as the seeded errors."""
    cat = item["category"]
    if cat == "channel_limit":
        return "channel_limit_exceeded"
    if cat == "claim_verification":
        return "unverifiable_claim" if item["verdict"] == "UNVERIFIABLE" else "contradiction"
    return cat  # banned_word, preferred_term, missing_disclaimer


@app.get("/api/evaluate")
async def evaluate_test_drafts():
    """Run all test drafts and compare seeded errors with findings BY TYPE.

    Found = seeded errors the product matched by type. Missed = seeded errors it
    did not find. False alarms = findings with no matching seeded error.
    Results are cached for 30 minutes so visitors cannot use up the AI quota.
    """
    if EVAL_CACHE["data"] is not None and time.time() - EVAL_CACHE["time"] < EVAL_CACHE_SECONDS:
        return EVAL_CACHE["data"]

    drafts_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "test_drafts", "drafts.json"))
    if not os.path.exists(drafts_path):
        raise HTTPException(status_code=404, detail="Test drafts not found")

    async with EVAL_LOCK:
        if EVAL_CACHE["data"] is not None and time.time() - EVAL_CACHE["time"] < EVAL_CACHE_SECONDS:
            return EVAL_CACHE["data"]

        with open(drafts_path, "r", encoding="utf-8") as f:
            drafts = json.load(f)

        results = []
        for d in drafts:
            rule_findings = check_rules(d["text"], d["channel"])
            claim_findings = [f for f in await check_claims(d["text"], d["channel"])
                              if f["verdict"] != "SUPPORTED"]

            found = Counter(found_group(f) for f in rule_findings + claim_findings)
            seeded = Counter(seeded_group(t) for t in d["seeded_errors"])

            seeded_count = sum(seeded.values())
            matched = sum(min(seeded[k], found[k]) for k in seeded)
            found_total = sum(found.values())

            results.append({
                "id": d["id"],
                "title": d["title"],
                "channel": d["channel"],
                "seeded_count": seeded_count,
                "seeded_errors": d["seeded_errors"],
                "found_count": matched,
                "missed_count": seeded_count - matched,
                "false_alarms": found_total - matched
            })

        EVAL_CACHE["data"] = results
        EVAL_CACHE["time"] = time.time()
        return results


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the single-file frontend HTML."""
    frontend_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "index.html"))
    if not os.path.exists(frontend_path):
        return HTMLResponse("<h1>Frontend index.html not found</h1>", status_code=404)
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())