"""FastAPI router backed by the fitted restaurant TF-IDF content model."""
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(tags=['Restaurant recommendations'])
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'restaurant_recommendation_model.joblib'

def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail='Restaurant recommendation model is not trained. Run train_recommendation_models.py.')
    return joblib.load(MODEL_PATH)

@router.get('/restaurants/recommend')
def recommend_restaurants(cuisine: Optional[str]=None, location: Optional[str]=None, max_price: Optional[float]=None, min_rating: float=0, limit: int=10, query: Optional[str]=None):
    if limit < 1 or limit > 100: raise HTTPException(400, detail='limit must be between 1 and 100')
    model=load_model(); rows=model['restaurants']; df=pd.DataFrame(rows)
    mask=pd.Series(True,index=df.index)
    if cuisine: mask &= df['cuisine'].astype(str).str.contains(cuisine.strip().strip('"\''),case=False,na=False,regex=False)
    if location: mask &= (df['location'].astype(str).str.contains(location.strip(),case=False,na=False,regex=False) | df['area'].astype(str).str.contains(location.strip(),case=False,na=False,regex=False))
    if max_price is not None: mask &= pd.to_numeric(df['price_numeric'],errors='coerce').fillna(float('inf')) <= max_price
    mask &= pd.to_numeric(df['rating'],errors='coerce').fillna(0) >= min_rating
    ids=df.index[mask].tolist()
    if not ids: return {'count':0,'recommendations':[],'message':'No restaurants match all filters; relax one or more filters.'}
    # Content similarity to query + smoothed rating/popularity score.
    scores={i:0.0 for i in ids}
    if query:
        qv=model['vectorizer'].transform([query])
        sims=cosine_similarity(qv,model['matrix']).ravel()
        scores={i:float(sims[i]) for i in ids}
    rating=pd.to_numeric(df['rating'],errors='coerce').fillna(0)
    count=pd.to_numeric(df['rating_count'],errors='coerce').fillna(0)
    pop=(count/(count+20)).clip(0,1)
    for i in ids: scores[i] += 0.75*(rating[i]/5)+0.25*pop[i]
    ids=sorted(ids,key=lambda i:scores[i],reverse=True)[:limit]
    out=[]
    for i in ids:
        r=df.iloc[i].to_dict(); r['recommendation_score']=round(scores[i],4); out.append(r)
    return {'count':len(out),'recommendations':out}
