lonexfury/
├── youtube_scraper.py     ← your original
├── pdf_processor.py       ← latest version (tested)
├── ocr_processor.py       ← latest version (tested)
├── processor.py           ← latest version (tested)
└── requirements.txt       ← this file

SETUP (one-time, takes ~15 min):

1. Install Python 3.11 from python.org (check "Add to PATH")

2. Install Tesseract OCR:
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - During install: CHECK "Hindi" under Additional language data
   - Install to default path: C:\Program Files\Tesseract-OCR
   - Add C:\Program Files\Tesseract-OCR to System PATH

3. Open PowerShell, go to the lonexfury folder:
   cd path\to\lonexfury

4. Install all Python packages:
   pip install -r requirements.txt
   (takes 5-10 min, downloads ~3GB)

5. Verify Tesseract works:
   tesseract --version


RUNNING (every time):

1. Open PowerShell, go to lonexfury folder:
   cd path\to\lonexfury

2. Start the service:
   py processor.py

3. Should show:
   "Uvicorn running on http://0.0.0.0:8001"

4. Test in browser:
   http://localhost:8001/health
   Should return JSON with "status": "ok"


THAT'S IT. Keep the PowerShell window open during hackathon.