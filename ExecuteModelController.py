from fastapi import FastAPI, HTTPException
from service.ExecuteModelService import ExecuteModelService
from BO.ExecuteModelParameters import ExecuteModelParameters






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn ExecuteModelController:app --reload --workers 1 --host 0.0.0.0 --port 8013






""" **************************************** Chargement de l'Api **************************************** """

app = FastAPI()
model_service_prediction = ExecuteModelService()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






""" **************************** Interroger le modèle **************************** """
@app.post("/execute-model", response_model=str, status_code=200)
async def load_dataset(input: ExecuteModelParameters):
    """ Méthode qui lance le Fine-Tuning """
    try:
        print("question : ", input.question)
        print("context : ", input.context)
        answer = model_service_prediction.question_answer(input.question, input.context)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

