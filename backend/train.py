from pathlib import Path
from datetime import date, timedelta
import json, math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
import joblib

ROOT=Path(__file__).resolve().parents[1]
OPS=json.loads((ROOT/'data/operational_demo.json').read_text())
ART=ROOT/'backend/artifacts'; ART.mkdir(exist_ok=True)
METRICS=ROOT/'frontend/src/data/metrics.json'
rng=np.random.default_rng(42)
ALT={'R-01':.82,'R-02':1.0,'R-03':.18,'R-04':.28,'R-05':.72,'R-06':.55,'R-07':.22,'R-08':.20}
COMM={'Rations':0,'POL':1,'Medical':2}
# Synthetic recurring pattern intentionally included so the model can learn temporal effects.
DOW_MULT={0:.94,1:.96,2:.99,3:1.03,4:1.10,5:1.25,6:1.30}

def demand_features(row, dt, lag, roll, ops_intensity=.45, weather=.30):
    dow=dt.weekday(); doy=dt.timetuple().tm_yday
    return [row['daily_demand'], lag, roll, math.sin(2*math.pi*dow/7), math.cos(2*math.pi*dow/7),
            math.sin(2*math.pi*doy/365), math.cos(2*math.pi*doy/365), COMM.get(row['commodity'],0),
            ALT.get(row['region'],.3), ops_intensity, weather]

def route_features(distance_km,duration_h,rain_mm,wind_kmh,temp_c,altitude,roughness):
    return [distance_km,duration_h,rain_mm,wind_kmh,abs(temp_c-18)/25,altitude,roughness]

def train_models():
    X=[]; y=[]
    start=date(2025,1,1)
    for row in OPS:
        base=row['daily_demand']; lag=base; hist=[]
        for i in range(520):
            dt=start+timedelta(days=i); dow=dt.weekday(); doy=dt.timetuple().tm_yday
            seasonal=1+.08*math.sin(2*math.pi*doy/365)
            ops_intensity=float(np.clip(rng.normal(.45,.18),0,1))
            weather=float(np.clip(rng.beta(2,5),0,1))
            demand=max(1,base*DOW_MULT[dow]*seasonal*(1+.14*ops_intensity+.07*weather)*rng.normal(1,.045))
            roll=float(np.mean(hist[-7:])) if hist else base
            X.append(demand_features(row,dt,lag,roll,ops_intensity,weather)); y.append(demand)
            hist.append(demand); lag=demand
    X=np.asarray(X); y=np.asarray(y)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42)
    reg=HistGradientBoostingRegressor(max_iter=240,max_leaf_nodes=31,l2_regularization=.2,random_state=42).fit(Xtr,ytr)
    pred=reg.predict(Xte)

    RX=[]; ry=[]
    for _ in range(10000):
        dist=float(rng.uniform(20,650)); dur=dist/float(rng.uniform(28,70))
        rain=float(rng.gamma(1.4,3)); wind=float(rng.uniform(2,65)); temp=float(rng.uniform(-12,43))
        alt=float(rng.uniform(.05,1)); rough=float(rng.beta(2.2,3))
        z=-3.6+.00115*dist+.045*rain+.014*wind+.65*(abs(temp-18)/25)+1.0*alt+1.05*rough
        p=1/(1+math.exp(-z)); label=int(rng.random()<p)
        RX.append(route_features(dist,dur,rain,wind,temp,alt,rough)); ry.append(label)
    RX=np.asarray(RX); ry=np.asarray(ry)
    Rtr,Rte,rtr,rte=train_test_split(RX,ry,test_size=.2,random_state=42,stratify=ry)
    clf=HistGradientBoostingClassifier(max_iter=220,l2_regularization=.25,random_state=42).fit(Rtr,rtr)
    prob=clf.predict_proba(Rte)[:,1]
    joblib.dump(reg,ART/'demand_model.joblib'); joblib.dump(clf,ART/'route_risk_model.joblib')
    m={'generatedAt':'training run','dataMode':'synthetic operational training data','demandModel':'HistGradientBoostingRegressor',
       'routeRiskModel':'HistGradientBoostingClassifier','mae':float(mean_absolute_error(yte,pred)),
       'routeRiskAUC':float(roc_auc_score(rte,prob)),'brier':float(brier_score_loss(rte,prob)),
       'trainingRows':len(X),'routeTrainingRows':len(RX),
       'temporalPattern':'Synthetic history includes recurring day-of-week effects; the model learns them from features.'}
    METRICS.write_text(json.dumps(m,indent=2)); return m

if __name__=='__main__':
    m=train_models(); print('Anumaan training complete'); print(json.dumps(m,indent=2))
