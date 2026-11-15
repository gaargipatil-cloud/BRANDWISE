# Brandwise

Pre-publication brand and fact review. Brandwise checks drafts and images against a company's own approved sources before they are published, explains each problem with a quoted source, suggests a fix that keeps the author's meaning, and routes the content to a Manager and, when the risk requires it, a Reviewer. People make the final decision.

- Live product: https://brandwise-h0pc.onrender.com
- Repository: https://github.com/gaargipatil-cloud/BRANDWISE
- Team: CodeX (CodeTitans India 2026). Built by the team leader, Gargi Patil.

All company data (Finwise, voice guide, fact sheet, claims policy, palette) is fictional and was written during the event.

## What it does
- **Draft review:** banned and preferred terms, channel length limits and required disclaimers (plain Python), plus source-checked claim verification.
- **Three-state verdict:** each claim is Supported, Contradicted or Unverifiable. A quote from the model is accepted only if it appears word for word in the brand documents.
- **Suggested fixes:** shown as before and after, and re-checked against the rules before they are shown.
- **Workflow:** dispute findings, save new versions, Manager decision, Reviewer sign-off for risky drafts, comments on findings, side-by-side version comparison, activity log.
- **Image and accessibility review:** brand colour match, estimated colour contrast (WCAG), alt text, image size, and the text inside the image checked like any other draft.
- **Guideline health:** scores each rule by how often it is disputed or overruled, and drafts clearer wording for a person to approve or dismiss.
- **Evaluation tab:** runs 15 labelled test drafts and reports found, missed and false alarms by error type.

## Run it locally (Windows)
1. Install Python 3.12.
2. From the project folder:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create `backend\.env` with your own Google Gemini key (from Google AI Studio):
   ```
   GEMINI_API_KEY=your-key
   GEMINI_MODEL=a-gemini-flash-model-id
   ```
   Without a key, the app still runs, but claim checks and image text reading return "Unverifiable" or "unavailable".
4. Start it:
   ```
   cd backend
   uvicorn app.main:app --reload
   ```
5. Open http://127.0.0.1:8000/

## Tests
```
python tests\run_checks.py
```
This checks the rules, colour maths, alt text checks and image checks. It does not call any AI service. Last results are in `tests\results.txt`.

## Project layout
```
backend/app/    main.py (API and workflow), rules.py, claims.py, retrieval.py (TF-IDF),
                llm.py (Gemini), imagecheck.py, image_routes.py, models.py, seed.py
data/           brand_pack/ (fictional Finwise documents), test_drafts/ (15 labelled drafts)
frontend/       index.html (single page, plain JavaScript)
tests/          run_checks.py
```

## Deployment
Deployed on Render (free plan) from this repository.
- Build command: `pip install -r requirements.txt`
- Start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Environment variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `PYTHON_VERSION`

## Privacy and safety
- No login: a Creator, Manager or Reviewer view is chosen from a menu (demo mode).
- Only the text under review is sent to the Gemini API. Do not enter confidential or personal information.
- User text is escaped before it is shown. Input is length-limited, uploads are limited to PNG, JPEG or WEBP up to 2 MB, and reviews are rate limited.
- Suggested wording for guidelines is a draft only. A person decides whether to use it.

## Known limitations
- The free host sleeps when idle (the first load can take about a minute), and stored data resets when it restarts.
- Claim checks depend on Google's Gemini service. If it is busy or slow, claims are shown as Unverifiable with the reason.
- Colour contrast in images is an estimate from the two main colours, not a reading of the actual text.
- Text in images is read by Gemini and can be wrong, so a person should check important images.
- The Evaluation tab matches errors by type, not by position.
- No DOCX or PDF upload. English only.

## AI assistance and disclosure
- The first version of the code was generated with Google AI Studio (Gemini) from a prompt written by the team leader.
- Claude (Anthropic) helped with debugging, new features, deployment steps and documents.
- At run time the product calls the Google Gemini API.
- The team leader reviewed, tested, changed and deployed the code.
- Libraries: FastAPI, SQLAlchemy, scikit-learn, httpx, Pillow, python-dotenv and pydantic.