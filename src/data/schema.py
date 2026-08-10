from datetime import date

from pydantic import BaseModel, ConfigDict


class DeviceRecord(BaseModel):
    """Validates a device record from either the training dataset or a
    production inference payload. Fields present only in real payloads
    (status, barcode, serialNumber, unitId, nextPredictionDate) are
    intentionally not declared here and get dropped by extra="ignore" —
    they carry no predictive signal or risk leaking the target."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    category: str
    manufacturer: str
    model: str
    climateZone: str
    usageIntensity: int
    manufacturingDate: int
    acquiredAt: date
    actualBreakDate: date | None = None
    mesesAteQuebra: float | None = None