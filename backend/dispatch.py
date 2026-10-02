from typing import Dict, List, Any
import math


# ---------------------------------------------------------
# Dispatch Decision Layer
# ---------------------------------------------------------
#
# This module does NOT train an ML model.
#
# It consumes:
#   - current inventory
#   - 7-day predicted demand
#
# and produces:
#   - total forecast demand
#   - projected stock
#   - stockout day
#   - safety stock
#   - recommended replenishment
#   - dispatch priority
#
# Operational values in this prototype are synthetic/
# representative and must not be interpreted as real
# military logistics data.
# ---------------------------------------------------------


def calculate_safety_stock(
    forecast: List[float],
    safety_factor: float = 0.20
) -> float:
    """
    Estimate safety stock from forecast variability.

    This is a prototype decision-layer calculation.
    It is NOT a claim about actual military reserve policy.
    """

    if not forecast:
        return 0.0

    mean_demand = sum(forecast) / len(forecast)

    variance = sum(
        (x - mean_demand) ** 2
        for x in forecast
    ) / len(forecast)

    std_demand = math.sqrt(variance)

    # Safety buffer based on forecast variability.
    safety_stock = std_demand * safety_factor

    # Keep a minimum buffer relative to average demand.
    minimum_buffer = mean_demand * 0.10

    return round(
        max(safety_stock, minimum_buffer),
        2
    )


def calculate_stock_projection(
    current_stock: float,
    forecast: List[float]
) -> List[float]:
    """
    Subtract predicted daily demand from current stock.
    """

    stock = float(current_stock)
    projection = []

    for demand in forecast:
        stock -= float(demand)
        projection.append(round(stock, 2))

    return projection


def find_stockout_day(
    stock_projection: List[float]
) -> int | None:
    """
    Return the first forecast day where projected stock
    reaches zero or below.

    Returns:
        1-based day number
        None if no stockout occurs in forecast horizon
    """

    for index, stock in enumerate(stock_projection):
        if stock <= 0:
            return index + 1

    return None


def calculate_dispatch_quantity(
    current_stock: float,
    forecast: List[float]
) -> Dict[str, Any]:

    if not forecast:
        return {
            "forecast_demand": 0,
            "safety_stock": 0,
            "recommended_replenishment": 0,
            "stockout_day": None,
            "stock_projection": [],
        }

    forecast = [
        max(0.0, float(x))
        for x in forecast
    ]

    total_demand = sum(forecast)

    safety_stock = calculate_safety_stock(
        forecast
    )

    stock_projection = calculate_stock_projection(
        current_stock,
        forecast
    )

    stockout_day = find_stockout_day(
        stock_projection
    )

    # Required quantity to cover the forecast horizon
    # plus the safety buffer.
    replenishment = (
        total_demand
        + safety_stock
        - float(current_stock)
    )

    replenishment = max(
        0.0,
        replenishment
    )

    return {
        "forecast_demand": round(total_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "recommended_replenishment": round(
            replenishment,
            2
        ),
        "stockout_day": stockout_day,
        "stock_projection": stock_projection,
    }


def determine_priority(
    stockout_day: int | None,
    current_stock: float,
    forecast_demand: float
) -> str:

    if forecast_demand <= 0:
        return "LOW"

    if stockout_day == 1:
        return "CRITICAL"

    if stockout_day is not None and stockout_day <= 3:
        return "HIGH"

    if stockout_day is not None and stockout_day <= 5:
        return "MEDIUM"

    # Coverage-based fallback.
    average_daily_demand = forecast_demand / 7

    if average_daily_demand > 0:
        cover_days = current_stock / average_daily_demand

        if cover_days <= 3:
            return "HIGH"

        if cover_days <= 5:
            return "MEDIUM"

    return "LOW"


def build_dispatch_decision(
    record: Dict[str, Any]
) -> Dict[str, Any]:

    region_id = record.get(
        "region_id",
        record.get("region", "UNKNOWN")
    )

    commodity = record.get(
        "commodity",
        "UNKNOWN"
    )

    current_stock = float(
        record.get(
            "current_stock",
            record.get("stock", 0)
        )
    )

    forecast = record.get(
        "forecast",
        record.get("predictions", [])
    )

    result = calculate_dispatch_quantity(
        current_stock=current_stock,
        forecast=forecast
    )

    priority = determine_priority(
        stockout_day=result["stockout_day"],
        current_stock=current_stock,
        forecast_demand=result["forecast_demand"]
    )

    return {
        "region_id": region_id,
        "commodity": commodity,
        "current_stock": round(
            current_stock,
            2
        ),

        "forecast_demand_7d": result[
            "forecast_demand"
        ],

        "safety_stock": result[
            "safety_stock"
        ],

        "recommended_replenishment": result[
            "recommended_replenishment"
        ],

        "stockout_day": result[
            "stockout_day"
        ],

        "stock_projection": result[
            "stock_projection"
        ],

        "priority": priority,

        "dispatch_required":
            result["recommended_replenishment"] > 0,

        "decision": (
            "DISPATCH_REQUIRED"
            if result["recommended_replenishment"] > 0
            else "NO_DISPATCH_REQUIRED"
        )
    }


def build_dispatch_plan(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    decisions = [
        build_dispatch_decision(record)
        for record in records
    ]

    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    decisions.sort(
        key=lambda x: (
            priority_order.get(
                x["priority"],
                99
            ),
            x["stockout_day"]
            if x["stockout_day"] is not None
            else 999
        )
    )

    return decisions