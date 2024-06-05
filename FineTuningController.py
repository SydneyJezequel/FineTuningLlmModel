from fastapi import FastAPI, HTTPException

from BO.FineTuningParameters import FineTuningParameters
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

@app.post("/fine-tune-model", response_model=bool, status_code=200)
async def fineTuneModel(input: FineTuningParameters):
    """ Méthode qui lance le Fine-Tuning """
    print("Arguments passés au modèle : ")
    print(" Nombre d'Epoch : ", input.num_epochs)
    print(" Taille du dataset d'entrainement : ", input.train_dataset_size)
    print(" Taille du dataset de validation : ", input.validation_dataset_size)
    print(" Taille des lots d'entrainement : ", input.train_batch_size)
    print(" Taille des lots de validation: ", input.eval_batch_size)
    try:
        fine_tuning_service.fine_tune_model(input.num_epochs, input.train_dataset_size, input.validation_dataset_size, input.train_batch_size,
                                            input.eval_batch_size)
        print("Fine Tuning du modèle terminé")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


