# Brandwise

Pre-publication brand integrity & claims review engine for Finwise. Verifies social posts, marketing emails, and web copy against company-approved brand documents using a **three-state verdict**: `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIABLE`.

---

## 1. Quick Setup & Run (Local)

### Prerequisites
- Python 3.11+
- Virtual environment tool

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone <your-repo-url>
cd brandwise

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate