from pydantic import BaseModel






class ExecuteModelParameters(BaseModel):
    """ Classe qui exécute le modèle question/réponse """



    question: str
    context: str


