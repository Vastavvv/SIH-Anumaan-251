from pathlib import Path
from datetime import date, datetime, timedelta
import json, math
import httpx, numpy as np, joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .train import train_models, demand_features, route_features, ALT
from backend.dispatch import build_dispatch_plan

ROOT=Path(__file__).resolve().parents[1]
REGIONS=json.loads((ROOT/'data/regions.json').read_text()); OPS=json.loads((ROOT/'data/operational_demo.json').read_text())
ART=ROOT/'backend/artifacts'; METRICS=ROOT/'frontend/src/data/metrics.json'
if not (ART/'demand_model.joblib').exists() or not (ART/'route_risk_model.joblib').exists(): train_models()
DEMAND=joblib.load(ART/'demand_model.joblib'); ROUTE=joblib.load(ART/'route_risk_model.joblib')
app=FastAPI(title='Anumaan Decision Support API',version='2.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])

def region(rid):
    x=next((r for r in REGIONS if r['id']==rid),None)
    if not x: raise HTTPException(404,'Unknown region')
    return x

async def fetch_weather(r):
    params={'latitude':r['lat'],'longitude':r['lon'],'current':'temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,weather_code','timezone':'auto'}
    async with httpx.AsyncClient(timeout=10) as c:
        q=await c.get('https://api.open-meteo.com/v1/forecast',params=params); q.raise_for_status(); d=q.json()
    x=d['current']; return {'region':r['id'],'city':r['city'],'temperature':x['temperature_2m'],'humidity':x['relative_humidity_2m'],'rain':x['rain'],'precipitation':x['precipitation'],'windSpeed':x['wind_speed_10m'],'weatherCode':x['weather_code'],'updatedAt':x['time'],'source':'Open-Meteo','mode':'LIVE'}

def make_forecast(start_date:date,surge:float=0.0):
    out=[]
    for row in OPS:
        projected=float(row['stock']); lag=float(row['daily_demand']); hist=[lag]*7; days=[]; stockout=None
        for i in range(7):
            dt=start_date+timedelta(days=i); roll=float(np.mean(hist[-7:]))
            feat=np.array([demand_features(row,dt,lag,roll,.45,.30)])
            demand=max(1,float(DEMAND.predict(feat)[0])*(1+surge/100))
            projected-=demand
            if projected<=0 and stockout is None: stockout=i+1
            days.append({'day':i+1,'date':dt.isoformat(),'dayName':dt.strftime('%a'),'predictedDemand':round(demand),'projectedStock':round(projected)})
            hist.append(demand); lag=demand
        minstock=min(d['projectedStock'] for d in days); total=sum(d['predictedDemand'] for d in days)
        safety=round(row['daily_demand']*1.5); qty=max(0,round(total+safety-row['stock']))
        if stockout: status='CRITICAL'
        elif minstock<=safety: status='WATCH'
        else: status='HEALTHY'
        out.append({'region':row['region'],'commodity':row['commodity'],'currentStock':row['stock'],'dailyBaseline':row['daily_demand'],'days':days,
                    'minimumProjectedStock':minstock,'stockoutDay':stockout,'status':status,'sevenDayPredictedDemand':total,
                    'safetyStock':safety,'recommendedDispatchQty':qty})
    return out

@app.get('/api/health')
def health(): return {'status':'ok','version':'2.0','policy':'Live public environment + public geography + synthetic operational ML training'}
@app.get('/api/regions')
def regions(): return REGIONS
@app.get('/api/operations')
def operations(): return {'mode':'SYNTHETIC / ANONYMIZED','records':OPS}
@app.get('/api/metrics')
def metrics(): return json.loads(METRICS.read_text())
@app.get('/api/forecast')
def forecast(start_date:str|None=None,demand_surge:float=Query(0,ge=-30,le=80)):
    try: dt=date.fromisoformat(start_date) if start_date else date.today()
    except: raise HTTPException(400,'start_date must be YYYY-MM-DD')
    return {'startDate':dt.isoformat(),'demandSurge':demand_surge,'forecast':make_forecast(dt,demand_surge)}
@app.post("/api/dispatch/plan")
def dispatch_plan(records: list[dict]):
    return {
        "status": "success",
        "count": len(records),
        "dispatch_plan": build_dispatch_plan(records)
    }
@app.get('/api/weather/all')
async def weather_all():
    out=[]
    for r in REGIONS:
        try: out.append(await fetch_weather(r))
        except Exception as e: out.append({'region':r['id'],'city':r['city'],'mode':'UNAVAILABLE','error':str(e)})
    return out
@app.get('/api/weather')
async def weather(region_id:str):
    try: return await fetch_weather(region(region_id))
    except HTTPException: raise
    except Exception as e: raise HTTPException(502,f'Weather unavailable: {e}')
@app.get('/api/roads')
async def roads(region_id:str):
    r=region(region_id); q='[out:json][timeout:15];way(around:5000,%s,%s)["highway"];out tags;'%(r['lat'],r['lon'])
    try:
        async with httpx.AsyncClient(timeout=18) as c:
            resp=await c.post('https://overpass-api.de/api/interpreter',data=q); resp.raise_for_status(); d=resp.json()
        classes={}
        for w in d.get('elements',[]):
            k=w.get('tags',{}).get('highway','unknown'); classes[k]=classes.get(k,0)+1
        return {'region':region_id,'roadFeatures':len(d.get('elements',[])),'classes':classes,'source':'OpenStreetMap / Overpass','mode':'PUBLIC GEO'}
    except Exception as e: return {'region':region_id,'roadFeatures':0,'classes':{},'source':'OpenStreetMap / Overpass','mode':'UNAVAILABLE','error':str(e)}

def roughness_from_classes(classes):
    if not classes:return .35
    rough=sum(v for k,v in classes.items() if k in ['track','path','unclassified','service'])
    total=max(1,sum(classes.values())); return min(1,.18+rough/total)

@app.get('/api/routes')
async def routes(origin_id:str='R-04',destination_id:str='R-02',rain_override:float|None=None,wind_override:float|None=None):
    o=region(origin_id); d=region(destination_id)
    if origin_id==destination_id: raise HTTPException(400,'Choose different origin and destination')

    # OSRM is used for actual road geometry. The public OSRM endpoint often returns
    # only one useful alternative, so Anumaan also requests waypoint-diversified
    # corridors to create genuinely different candidate routes.
    lat1,lon1=o['lat'],o['lon']; lat2,lon2=d['lat'],d['lon']
    midlat=(lat1+lat2)/2; midlon=(lon1+lon2)/2
    dlat=lat2-lat1; dlon=lon2-lon1; norm=max((dlat*dlat+dlon*dlon)**0.5,0.01)
    # Perpendicular offsets create three corridor candidates; bounded to keep them geographic.
    perp_lat=-dlon/norm; perp_lon=dlat/norm
    offset=max(0.8,min(2.8,norm*0.10))
    waypoints=[
        (midlat+perp_lat*offset, midlon+perp_lon*offset),
        (midlat, midlon),
        (midlat-perp_lat*offset, midlon-perp_lon*offset),
    ]
    async def osrm(points):
        coords=';'.join(f'{lon},{lat}' for lat,lon in points)
        url=f'https://router.project-osrm.org/route/v1/driving/{coords}'
        params={'overview':'full','geometries':'geojson','steps':'false','alternatives':'false'}
        async with httpx.AsyncClient(timeout=18) as c:
            rr=await c.get(url,params=params); rr.raise_for_status(); return rr.json().get('routes',[None])[0]
    candidates=[]
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            direct_url=f'https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}'
            direct=(await c.get(direct_url,params={'alternatives':'true','overview':'full','geometries':'geojson','steps':'false'})).json().get('routes',[])
        candidates.extend(direct[:2])
        for wp in waypoints:
            try:
                r=await osrm([(lat1,lon1),wp,(lat2,lon2)])
                if r: candidates.append(r)
            except Exception: pass
    except Exception: pass

    # Deduplicate nearly identical routes and keep up to three meaningfully different corridors.
    unique=[]
    for r in candidates:
        if not r: continue
        km=r.get('distance',0)/1000
        if km<=0: continue
        if all(abs(km-u['distance']/1000)>3 for u in unique): unique.append(r)
    candidates=sorted(unique,key=lambda x:x['distance'])[:3]

    if not candidates:
        straight=math.hypot((lat1-lat2)*111,(lon1-lon2)*95)
        candidates=[{'distance':straight*1000*1.25,'duration':straight/48*3600,'geometry':{'coordinates':[[lon1,lat1],[lon2,lat2]]}}]
        routing_mode='FALLBACK STRAIGHT-LINE DEMO'
    else:
        routing_mode='LIVE OSRM + WAYPOINT-DIVERSIFIED PUBLIC ROUTING'

    # Sample live weather at points along each route. Weather is route context,
    # not merely a destination card. The ML risk model receives rain/wind/temp.
    async def weather_at(lat,lon):
        params={'latitude':lat,'longitude':lon,'current':'temperature_2m,precipitation,rain,wind_speed_10m,weather_code','timezone':'auto'}
        async with httpx.AsyncClient(timeout=10) as c:
            q=await c.get('https://api.open-meteo.com/v1/forecast',params=params); q.raise_for_status(); x=q.json()['current']
        return x
    async def route_weather(coords):
        pts=[]
        if coords:
            for idx in [0,len(coords)//2,len(coords)-1]:
                if 0<=idx<len(coords):
                    pt=coords[idx]
                    if pt not in pts: pts.append(pt)
        vals=[]
        for lon,lat in pts:
            try: vals.append(await weather_at(lat,lon))
            except Exception: pass
        if not vals: return {'temperature':18,'rain':0,'windSpeed':12,'maxRain':0,'maxWind':12,'weatherCodes':[],'mode':'UNAVAILABLE'}
        return {'temperature':round(float(np.mean([v['temperature_2m'] for v in vals])),1),
                'rain':round(float(np.mean([v.get('rain',0) or 0 for v in vals])),1),
                'windSpeed':round(float(np.mean([v.get('wind_speed_10m',12) or 12 for v in vals])),1),
                'maxRain':round(float(max(v.get('rain',0) or 0 for v in vals)),1),
                'maxWind':round(float(max(v.get('wind_speed_10m',12) or 12 for v in vals)),1),
                'weatherCodes':[v.get('weather_code') for v in vals], 'mode':'LIVE ROUTE SAMPLES'}

    try: road=await roads(destination_id)
    except: road={'classes':{}}
    rough=roughness_from_classes(road.get('classes',{})); alt=ALT.get(destination_id,.3)
    result=[]
    for i,r in enumerate(candidates):
        km=r['distance']/1000; h=r['duration']/3600
        coords=[[lat,lon] for lon,lat in r['geometry']['coordinates']]
        rw=await route_weather(r['geometry']['coordinates'])
        rain=float(rain_override if rain_override is not None else rw['maxRain'])
        wind=float(wind_override if wind_override is not None else rw['maxWind'])
        temp=float(rw['temperature'])
        p=float(ROUTE.predict_proba(np.array([route_features(km,h,rain,wind,temp,alt,rough)]))[:,1][0])
        result.append({'id':chr(65+i),'distanceKm':round(km,1),'durationHours':round(h,2),
                       'riskProbability':round(p,3),'riskLabel':'HIGH' if p>=.6 else ('MEDIUM' if p>=.35 else 'LOW'),
                       'coordinates':coords,'weather':rw,'riskFactors':{'rainMm':rain,'windKmh':wind,'temperatureC':temp,'terrainIndex':round(alt,2),'roadRoughness':round(rough,2)}})
    # Risk is primary, but ETA matters. A route is only selected after comparing all candidates.
    maxh=max(x['durationHours'] for x in result) or 1
    minh=min(x['durationHours'] for x in result) if result else 0
    for x in result:
        eta_penalty=(x['durationHours']-minh)/max(0.01,maxh-minh) if maxh>minh else 0
        x['decisionScore']=round(.80*x['riskProbability']+.20*eta_penalty,3)
    best=min(result,key=lambda x:x['decisionScore']); best['selected']=True
    for x in result: x.setdefault('selected',False)
    selection_reason=(f"Route {best['id']} selected because it has the lowest combined score "
                      f"(80% predicted disruption risk + 20% relative ETA). Live route weather is included "
                      f"using sampled rain/wind/temperature; the severe-weather switch overrides those values for a what-if.")
    return {'origin':o,'destination':d,'routes':result,'selectedRoute':best['id'],'weather':best['weather'],'roadContext':road,
            'routingMode':routing_mode,'riskMode':'ML classifier trained on synthetic disruption scenarios',
            'decisionExplanation':selection_reason,'weatherUsedInRisk':True}

