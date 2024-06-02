from transformers import AutoModelForQuestionAnswering






class Model:
    """ Classe du Modèle """



    def __init__(self, model_name):
        """ Constructeur """
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=True)


