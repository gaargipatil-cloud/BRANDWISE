"""Seed demo drafts, guidelines, and audit logs on application startup."""
import os
import json
from sqlalchemy.orm import Session
from .models import Submission, Version, Finding, AuditLog
from .retrieval import init_brand_corpus


def seed_database(db: Session):
    """Populate database with sample submissions if empty."""
    # Ensure retrieval corpus is loaded
    init_brand_corpus()

    # Check if data already exists
    if db.query(Submission).first():
        return

    drafts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_drafts", "drafts.json"))
    if not os.path.exists(drafts_path):
        return

    with open(drafts_path, "r", encoding="utf-8") as f:
        drafts = json.load(f)

    # Seed top 3 demo submissions into DB for manager/creator views
    for d in drafts[:4]:
        sub = Submission(
            title=d["title"],
            channel=d["channel"],
            status="pending_review"
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)

        ver = Version(
            submission_id=sub.id,
            version_number=1,
            text=d["text"]
        )
        db.add(ver)
        db.commit()
        db.refresh(ver)

        # Seed initial audit log entry
        log = AuditLog(
            actor_role="Creator",
            action="Created submission",
            details=f"Created '{sub.title}' for channel {sub.channel}."
        )
        db.add(log)
        db.commit()