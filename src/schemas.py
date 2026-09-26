"""Pydantic schemas: the exact shape of what the API accepts and returns."""
from pydantic import BaseModel, Field


class Customer(BaseModel):
    """One customer's profile, exactly the 19 features the model expects."""
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(examples=[0], ge=0, le=1)
    Partner: str = Field(examples=["No"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[2], ge=0)
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["No"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["No"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[90.0], ge=0)
    TotalCharges: float = Field(examples=[180.0], ge=0)


class PredictionResponse(BaseModel):
    """What the API sends back."""
    churn_probability: float
    at_risk: bool
    threshold: float