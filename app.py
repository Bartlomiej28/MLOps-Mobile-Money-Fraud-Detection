from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from fraud_detection.constants import APP_HOST, APP_PORT
from fraud_detection.pipeline.prediction_pipeline import (
    TRANSACTION_TYPES,
    FraudDetectionClassifier,
    FraudTransactionData,
)
from fraud_detection.pipeline.training_pipeline import TrainPipeline

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.step: Optional[str] = None
        self.type: Optional[str] = None
        self.amount: Optional[str] = None
        self.oldbalanceOrg: Optional[str] = None
        self.newbalanceOrig: Optional[str] = None
        self.oldbalanceDest: Optional[str] = None
        self.newbalanceDest: Optional[str] = None

    async def get_transaction_data(self):
        form = await self.request.form()
        self.step = form.get("step")
        self.type = form.get("type")
        self.amount = form.get("amount")
        self.oldbalanceOrg = form.get("oldbalanceOrg")
        self.newbalanceOrig = form.get("newbalanceOrig")
        self.oldbalanceDest = form.get("oldbalanceDest")
        self.newbalanceDest = form.get("newbalanceDest")


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=0, le=743)
    type: str
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "fraud_detection.html",
        {"request": request, "context": None, "transaction_types": TRANSACTION_TYPES},
    )


@app.get("/train")
async def train_route_client():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predict_route_client(request: Request):
    try:
        form = DataForm(request)
        await form.get_transaction_data()

        transaction = FraudTransactionData(
            step=form.step,
            type=form.type,
            amount=form.amount,
            oldbalanceOrg=form.oldbalanceOrg,
            newbalanceOrig=form.newbalanceOrig,
            oldbalanceDest=form.oldbalanceDest,
            newbalanceDest=form.newbalanceDest,
        )

        transaction_df = transaction.get_input_data_frame()

        model_predictor = FraudDetectionClassifier()
        prediction, label, probability = model_predictor.predict_with_probability(transaction_df)

        context = {
            "label": label,
            "is_fraud": prediction == 1,
            "status": "Fraudulent transaction" if prediction == 1 else "Legitimate transaction",
            "probability": None if probability is None else round(probability * 100, 2),
        }

        return templates.TemplateResponse(
            "fraud_detection.html",
            {"request": request, "context": context, "transaction_types": TRANSACTION_TYPES},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "fraud_detection.html",
            {
                "request": request,
                "context": {"error": str(e)},
                "transaction_types": TRANSACTION_TYPES,
            },
        )


@app.post("/predict")
async def predict_api(payload: TransactionRequest):
    try:
        transaction = FraudTransactionData(**payload.model_dump())
        model_predictor = FraudDetectionClassifier()
        prediction, label, probability = model_predictor.predict_with_probability(
            transaction.get_input_data_frame()
        )
        return {
            "prediction": prediction,
            "label": label,
            "fraud_probability": probability,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": False, "error": str(e)})


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
