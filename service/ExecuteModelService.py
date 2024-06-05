from transformers import pipeline
import config






class ExecuteModelService:
    """ Méthode en charge de l'exécution du modèle """



    def __init__(self):
        self.question_answerer = pipeline(config.ACTIVITY_TYPE, model=config.MODEL_PATH)



    def question_answer(self, question, context):
        "Méthode en charge de répondre à une question utilisant un pipeline"
        result = self.question_answerer(question=question, context=context)
        print("result : ", result)
        answer = result['answer']
        print("answer : ", answer)
        return answer

