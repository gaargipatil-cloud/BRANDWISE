"""Seed demo drafts with their rule findings on application startup."""
import os
import json
from sqlalchemy.orm import Session
from .models import Submission, Version, Finding, AuditLog
from .retrieval import init_brand_corpus
from .rules import check_rules

# Drafts that appear in the Manager queue for the demo
DEMO_DRAFT_IDS = ["draft_01", "draft_03", "draft_06", "draft_07", "draft_13"]


def seed_database(db: Session):
    """Populate the database with sample submissions if it is empty."""
    init_brand_corpus()

    if db.query(Submission).first():
        return

    drafts_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "test_drafts", "drafts.json"))
    if not os.path.exists(drafts_path):
        return

    with open(drafts_path, "r", encoding="utf-8") as f:
        drafts = {d["id"]: d for d in json.load(f)}

    for draft_id in DEMO_DRAFT_IDS:
        d = drafts.get(draft_id)
        if not d:
            continue

        sub = Submission(title=d["title"], channel=d["channel"], status="pending_review")
        db.add(sub)
        db.commit()
        db.refresh(sub)

        ver = Version(submission_id=sub.id, version_number=1, text=d["text"])
        db.add(ver)
        db.commit()
        db.refresh(ver)

        # Rule checks only, so starting the app never calls the AI service
        for item in check_rules(d["text"], d["channel"]):
            db.add(Finding(
                version_id=ver.id,
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
            ))

        db.add(AuditLog(
            actor_role="Creator",
            action="Created submission",
            details=f"Created '{sub.title}' for channel {sub.channel}."
        ))
        db.commit()