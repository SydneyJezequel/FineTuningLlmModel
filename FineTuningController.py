from fastapi import FastAPI, HTTPException
from service.FineTuningService import FineTuningService






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn FineTuningController:app --reload --workers 1 --host 0.0.0.0 --port 8012






""" **************************************** Chargement de l'Api **************************************** """

app = FastAPI()
fine_tuning_service = FineTuningService()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






""" **************************** Fine Tuning du modèle LLM **************************** """

@app.get("/fine-tune-model", response_model=bool, status_code=200)
async def load_dataset():
    """ Méthode qui lance le Fine-Tuning """
    try:
        fine_tuning_service.fine_tune_model()
        print("Fine Tuning du modèle terminé")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


