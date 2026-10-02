const API = "http://127.0.0.1:8000/api";

async function get(path) {
  const response = await fetch(API + path);

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export const getRegions = () =>
  get("/regions");

export const getWeatherAll = () =>
  get("/weather/all");

export const getOperations = () =>
  get("/operations");

export const getMetrics = () =>
  get("/metrics");

export const getHealth = () =>
  get("/health");

export const getRoads = (id) =>
  get(
    "/roads?region_id=" +
    encodeURIComponent(id)
  );

export const getForecast = (start, surge = 0) =>
  get(
    `/forecast?start_date=${encodeURIComponent(start)}&demand_surge=${surge}`
  );

export const getRoutes = (
  origin,
  destination,
  rain = null,
  wind = null
) => {
  let path =
    `/routes?origin_id=${encodeURIComponent(origin)}` +
    `&destination_id=${encodeURIComponent(destination)}`;

  if (rain !== null) {
    path += `&rain_override=${rain}`;
  }

  if (wind !== null) {
    path += `&wind_override=${wind}`;
  }

  return get(path);
};

export const generateDispatchPlan = async (records) => {
  const response = await fetch(
    `${API}/dispatch/plan`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(records)
    }
  );

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
};