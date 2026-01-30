from fastapi import FastAPI, HTTPException
from backend.schemas import TextRequest, FileRequest, AnalysisResponse
from backend.services import analyze_single_text, analyze_batch

app = FastAPI(
    title="AI Chat Analysis API",
    description="تحليل المحادثات العربية وتوليد بروفايل معرفي",
    version="1.0"
)

@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextRequest):
    try:
        result = analyze_single_text(req.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", response_model=AnalysisResponse)
def analyze_file(req: FileRequest):
    try:
        result = analyze_batch(req.conversations)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
