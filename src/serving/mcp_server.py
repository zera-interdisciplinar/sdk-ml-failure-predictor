import os

from mcp.server import MCPServer

from src.data.features import UNK_TOKEN
from src.serving.inference import TTFPredictor
from src.serving.schema import PredictionRequest

mcp: MCPServer = MCPServer("ttf-failure-predictor")
predictor: TTFPredictor = TTFPredictor()


@mcp.tool()
def predict_time_to_failure(request: PredictionRequest) -> float:
    """Predicts how many months until an IT device is expected to break.

    Use this when you need a time-to-failure estimate for a single device
    to support maintenance planning, replacement scheduling, or risk
    triage — e.g. "how long until this device is likely to fail?" or
    "which of these devices should we prioritize for replacement?".

    Args:
        request: The device's identifying and usage attributes (category,
            manufacturer, model, climate zone, usage intensity,
            manufacturing year and acquisition date). See PredictionRequest
            for field-level descriptions.

    Returns:
        The predicted number of months remaining until failure, counted
        from the device's acquisition date. Always a non-negative float;
        may exceed the device's actual remaining useful life since it is
        a statistical estimate, not a guarantee.
    """
    return predictor.predict(request.model_dump(mode="json"))


@mcp.tool()
def list_valid_categories() -> list[str]:
    """Lists the device categories the model was trained on.

    Call this before predict_time_to_failure if you need to validate or
    suggest a `category` value up front, e.g. to build a picklist or
    reject typos before making a prediction request.

    Returns:
        The known category values, sorted alphabetically. A `category`
        outside this list is not rejected by predict_time_to_failure —
        it is treated as an unknown category, which typically makes the
        prediction less accurate. This list can grow over time as the
        model is retrained on new data, so callers should fetch it at
        runtime rather than hardcoding a copy.
    """
    return sorted(v for v in predictor.encoder.vocabs["category"] if v != UNK_TOKEN)


@mcp.tool()
def list_valid_climate_zones() -> list[str]:
    """Lists the climate zones the model was trained on.

    Call this before predict_time_to_failure if you need to validate or
    suggest a `climateZone` value up front, e.g. to build a picklist or
    reject typos before making a prediction request.

    Returns:
        The known climate zone values, sorted alphabetically. A
        `climateZone` outside this list is not rejected by
        predict_time_to_failure — it is treated as an unknown zone, which
        typically makes the prediction less accurate. This list can grow
        over time as the model is retrained on new data, so callers
        should fetch it at runtime rather than hardcoding a copy.
    """
    return sorted(v for v in predictor.encoder.vocabs["climateZone"] if v != UNK_TOKEN)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("MCP_HTTP_PORT", "8000")),
        json_response=True,
        stateless_http=True,
    )
