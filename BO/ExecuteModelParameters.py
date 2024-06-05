from pydantic import BaseModel






class ExecuteModelParameters(BaseModel):
    """ Paramètres attendus par la méthode model_service_prediction::question_answer() """



    question: str
    context: str


