from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input for a single time-to-failure prediction.

    Carries only the fields the model was trained on. Fields used solely
    to derive the training target (actualBreakDate, mesesAteQuebra) are
    intentionally absent here: providing them would leak the answer into
    the input instead of asking the model to predict it."""

    category: str = Field(description="Device category, e.g. 'notebook', 'desktop', 'monitor'.")
    manufacturer: str = Field(description="Device manufacturer/brand, e.g. 'Dell', 'HP', 'Lenovo'.")
    model: str = Field(description="Device model name or code, e.g. 'Latitude 5420'.")
    climateZone: str = Field(description="Climate zone where the device operates, e.g. 'TROPICAL', 'TEMPERATE'.")
    usageIntensity: int = Field(description="How intensively the device is used, on the dataset's original scale.")
    manufacturingDate: int = Field(description="Year the device was manufactured, e.g. 2021.")
    acquiredAt: date = Field(description="Date the device was acquired/put into service, ISO format (YYYY-MM-DD).")
