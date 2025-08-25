import fastapi
from fastapi.responses import JSONResponse
import pandas as pd

from challenge.model import DelayModel

app = fastapi.FastAPI()

_model = DelayModel()
_KNOWN_OPERA: set = set()
_LOADED = False
_ALLOWED_TIPOVUELO = {"I", "N"}


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    df = pd.read_csv("data/data.csv", low_memory=False)
    X, y = _model.preprocess(data=df, target_column="delay")
    _model.fit(features=X, target=y)
    _KNOWN_OPERA.clear()
    _KNOWN_OPERA.update(df["OPERA"].dropna().unique().tolist())
    _LOADED = True


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


def _bad_request(msg: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": msg})


def _validate_and_build_df(payload: dict) -> pd.DataFrame | JSONResponse:
    if not isinstance(payload, dict) or "flights" not in payload:
        return _bad_request("Payload must contain 'flights' list.")
    flights = payload["flights"]
    if not isinstance(flights, list) or not flights:
        return _bad_request("'flights' must be a non-empty list.")

    rows = []
    for idx, row in enumerate(flights):
        if not isinstance(row, dict):
            return _bad_request(f"Row {idx} must be an object.")
        for key in ("OPERA", "TIPOVUELO", "MES"):
            if key not in row:
                return _bad_request(f"Missing '{key}' in row {idx}.")

        opera = row["OPERA"]
        tipovuelo = row["TIPOVUELO"]
        mes = row["MES"]

        if tipovuelo not in _ALLOWED_TIPOVUELO:
            return _bad_request("Unknown TIPOVUELO.")
        if not isinstance(mes, int) or not (1 <= mes <= 12):
            return _bad_request("Unknown MES.")
        if opera not in _KNOWN_OPERA:
            return _bad_request("Unknown OPERA.")

        rows.append({"OPERA": opera, "TIPOVUELO": tipovuelo, "MES": mes})

    df = pd.DataFrame(rows)
    if "Fecha-I" not in df.columns:
        df["Fecha-I"] = pd.NaT
    if "Fecha-O" not in df.columns:
        df["Fecha-O"] = pd.NaT
    return df


@app.post("/predict", status_code=200)
async def post_predict(payload: dict) -> dict | JSONResponse:
    _ensure_loaded()
    df_or_resp = _validate_and_build_df(payload)
    if isinstance(df_or_resp, JSONResponse):
        return df_or_resp
    X = _model.preprocess(data=df_or_resp)
    preds = _model.predict(features=X)
    return {"predict": preds}
