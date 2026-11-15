"""Image and accessibility review endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .models import Submission, Version, AuditLog, ImageAsset
from .rules import check_rules
from .imagecheck import (decode_data_url, open_image, image_findings,
                         make_thumbnail, make_finding, rgb_to_hex)
from .llm import read_image_text

router = APIRouter()


class ImageReviewRequest(BaseModel):
    title: str = Field(..., max_length=200)
    alt_text: str = Field("", max_length=500)
    image_data: str = Field(..., max_length=3_600_000)


@router.post("/api/image-review")
async def review_image(payload: ImageReviewRequest, request: Request, db: Session = Depends(get_db)):
    """Check an image for colour, contrast, alt text and size, then review the text inside it."""
    # Imported here, not at the top, because main.py imports this file (avoids a circular import)
    from .main import enforce_rate_limits, get_client_ip, check_claims, save_findings, count_issues

    enforce_rate_limits(get_client_ip(request))

    try:
        raw = decode_data_url(payload.image_data)
        img, mime = open_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    findings, colours = image_findings(img, payload.alt_text)

    # Read the text in the image, then review it like any other draft
    extracted, text_status = await read_image_text(raw, mime)
    if text_status == "ok":
        text = extracted[:3000]
        findings += check_rules(text, "image")
        findings += await check_claims(text, "image")
    elif text_status == "failed":
        findings.append(make_finding(
            "image_text", "medium", "(text inside the image)",
            "The text in this image could not be read automatically, so it was not checked "
            "against the brand sources.",
            "Automated check", "", "A person should read the text in the image and check it.",
            confidence=0.0, verdict="UNVERIFIABLE"))

    submission = Submission(title=payload.title.strip() or "Image review", channel="image",
                            status="pending_review")
    db.add(submission)
    db.commit()
    db.refresh(submission)

    version_text = extracted if extracted else "(no readable text found in the image)"
    version = Version(submission_id=submission.id, version_number=1, text=version_text[:3000])
    db.add(version)
    db.commit()
    db.refresh(version)

    saved = save_findings(db, version.id, findings)

    thumbnail = make_thumbnail(img)
    db.add(ImageAsset(submission_id=submission.id, data_url=thumbnail,
                      alt_text=payload.alt_text.strip(), width=img.size[0], height=img.size[1]))
    issues = count_issues(saved)
    db.add(AuditLog(actor_role="Creator", action="Reviewed image",
                    details=f"Image '{submission.title}' scanned: {issues} issues, "
                            f"{len(saved) - issues} claims passed."))
    db.commit()

    return {
        "submission_id": submission.id,
        "findings": saved,
        "extracted_text": extracted,
        "text_status": text_status,
        "width": img.size[0],
        "height": img.size[1],
        "colours": [{"hex": c["hex"], "share": round(c["share"], 3)} for c in colours]
    }


@router.get("/api/submission/{sub_id}/image")
def get_submission_image(sub_id: int, db: Session = Depends(get_db)):
    """The stored preview and alt text for an image submission."""
    asset = db.query(ImageAsset).filter(ImageAsset.submission_id == sub_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="No image for this submission")
    return {"data_url": asset.data_url, "alt_text": asset.alt_text,
            "width": asset.width, "height": asset.height}