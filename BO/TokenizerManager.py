from transformers import AutoTokenizer






class TokenizerManager:
    """ Classe du Gestionnaire de Tokenizer """



    def __init__(self, model_name):
        """ Constructeur """
        # Définition du Tokenizer pré-entraîné à partir du modèle spécifié :
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


