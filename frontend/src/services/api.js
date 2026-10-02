const API="http://127.0.0.1:8000/api";
async function get(p){const r=await fetch(API+p);if(!r.ok)throw Error(await r.text());return r.json()}
export const getRegions=()=>get("/regions");
export const getWeatherAll=()=>get("/weather/all");
export const getOperations=()=>get("/operations");
export const getMetrics=()=>get("/metrics");
export const getHealth=()=>get("/health");
export const getRoads=id=>get("/roads?region_id="+encodeURIComponent(id));
export const getForecast=(start,surge=0)=>get(`/forecast?start_date=${encodeURIComponent(start)}&demand_surge=${surge}`);
export const getRoutes=(o,d,rain=null,wind=null)=>{let p=`/routes?origin_id=${o}&destination_id=${d}`;if(rain!==null)p+=`&rain_override=${rain}`;if(wind!==null)p+=`&wind_override=${wind}`;return get(p)};
