import os

from mcp.server import MCPServer

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


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("MCP_HTTP_PORT", "8000")),
        json_response=True,
        stateless_http=True,
    )
