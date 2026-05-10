"""FastAPI додаток: форма для live predict."""

import os
import sys

# Додаємо корінь проєкту в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.predict import predict, load_production_model
from src.config import FEATURE_LABELS_UA

app = FastAPI(title="Credit Scoring Demo")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.on_event("startup")
def startup():
    load_production_model()


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "labels": FEATURE_LABELS_UA,
        "result": None,
        "values": {},
    })


@app.post("/predict", response_class=HTMLResponse)
async def do_predict(
    request: Request,
    age: float = Form(...),
    monthly_income: float = Form(...),
    num_delinquencies: float = Form(...),
    credit_term_months: float = Form(...),
    credit_amount: float = Form(...),
    debt_to_income: float = Form(...),
):
    data = {
        "age": age,
        "monthly_income": monthly_income,
        "num_delinquencies": num_delinquencies,
        "credit_term_months": credit_term_months,
        "credit_amount": credit_amount,
        "debt_to_income": debt_to_income,
    }

    result = predict(data)

    return templates.TemplateResponse(request=request, name="index.html", context={
        "labels": FEATURE_LABELS_UA,
        "result": result,
        "values": data,
    })
