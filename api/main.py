
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import ast

app = FastAPI(
    title="Market Basket Recommendation API",
    description="Recommends products using association rules.",
    version="1.0.0",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RULES_FILE = PROJECT_DIR / "notebooks" / "rules_baseline.csv"


class RecommendationRequest(BaseModel):
    items: list[str] = Field(
        min_length=1,
        description="Products the customer already has"
    )


class RecommendationResponse(BaseModel):
    recommendations: list[str]



def load_rules():
    if not RULES_FILE.exists():
        return []

    df = pd.read_csv(RULES_FILE)
    records = []

    for _, row in df.iterrows():
        # Convert values like frozenset({'bread'}) into ordinary sets
        antecedents = set(
            str(item).strip()
            for item in row["antecedents"]
            .replace("frozenset(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .split(",")
            if item.strip()
        )

        consequents = set(
            str(item).strip()
            for item in row["consequents"]
            .replace("frozenset(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .split(",")
            if item.strip()
        )

        records.append({
            "antecedents": antecedents,
            "consequents": consequents,
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
        })

    return records

    return records


@app.get("/")
def home():
    return {"message": "Market Basket Recommendation API is running"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse
)
def recommend(request: RecommendationRequest):
    rules = load_rules()
    customer_items = {item.strip().lower() for item in request.items}
    recommendations = []

    for rule in rules:
        if rule["antecedents"].issubset(customer_items):
            for product in rule["consequents"]:
                if product not in customer_items:
                    recommendations.append(product)

    return {
        "recommendations": sorted(set(recommendations))
    }
