import os

from mcp.server import MCPServer

from src.serving.inference import TTFPredictor

mcp: MCPServer = MCPServer("ttf-failure-predictor")
predictor: TTFPredictor = TTFPredictor()


@mcp.tool()
def predict_time_to_failure(
    category: str,
    manufacturer: str,
    model: str,
    climateZone: str,
    usageIntensity: int,
    manufacturingDate: int,
    acquiredAt: str,
) -> float:
    """Predicts how many months until an IT device is expected to break.

    Given the device's category, manufacturer, model, climate zone,
    usage intensity, manufacturing year and acquisition date (ISO format,
    e.g. "2023-05-01"), returns the predicted number of months remaining
    until failure.
    """
    payload: dict[str, str | int] = {
        "category": category,
        "manufacturer": manufacturer,
        "model": model,
        "climateZone": climateZone,
        "usageIntensity": usageIntensity,
        "manufacturingDate": manufacturingDate,
        "acquiredAt": acquiredAt,
    }
    return predictor.predict(payload)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("MCP_HTTP_PORT", "8000")),
        json_response=True,
        stateless_http=True,
    )
