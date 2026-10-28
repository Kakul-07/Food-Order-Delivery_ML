"""FastAPI router using customer order-history preferences learned during training."""
from pathlib import Path
from typing import Optional
import joblib
from fastapi import APIRouter, HTTPException, Query

router=APIRouter(tags=['Previous-order recommendations'])
MODEL_PATH=Path(__file__).resolve().parent/'models'/'previous_order_model.joblib'

def load_model():
    if not MODEL_PATH.exists(): raise HTTPException(status_code=503,detail='Previous-order model is not trained. Run train_recommendation_models.py.')
    return joblib.load(MODEL_PATH)

@router.get('/restaurants/previous-orders')
def recommend_from_history(customer_id: str=Query(...), limit: int=10, cuisine: Optional[str]=None):
    if limit < 1 or limit > 100: raise HTTPException(400,detail='limit must be between 1 and 100')
    m=load_model(); uid=str(customer_id)
    if uid not in {k[0] for k in m['user_restaurant']}:
        # Cold-start: global most-ordered restaurants from the training data.
        scored=[(name,float(n),'popular_fallback') for name,n in m['restaurant_popularity'].items()]
    else:
        user_rest={r:n for (u,r),n in m['user_restaurant'].items() if u==uid}
        user_cuisine={c:n for (u,c),n in m['user_cuisine'].items() if u==uid}
        scored=[]
        for restaurant,pop in m['restaurant_popularity'].items():
            # Repeatedly ordered restaurants are boosted; cuisine preferences personalize discovery.
            cuisines={c:n for (r,c),n in m['restaurant_cuisine'].items() if r==restaurant}
            cuisine_score=sum(user_cuisine.get(c,0)*n for c,n in cuisines.items())
            score=4.0*user_rest.get(restaurant,0)+0.8*cuisine_score+0.05*pop
            if score>0: scored.append((restaurant,float(score),'personalized_history'))
    if cuisine: scored=[x for x in scored if cuisine.strip().strip('"\'').lower() in x[0].lower() or any(c.lower()==cuisine.strip().lower() for (r,c),n in m['restaurant_cuisine'].items() if r==x[0])]
    scored.sort(key=lambda x:x[1],reverse=True)
    result=[{'restaurant_name':r,'recommendation_score':round(s,4),'reason':reason} for r,s,reason in scored[:limit]]
    return {'customer_id':uid,'count':len(result),'recommendations':result,'note':'Recommendations are learned from the included historical order sample; cold-start customers receive popularity-based suggestions.'}
