from transformers import AutoTokenizer






class Tokenizer:
    """ Classe du Tokenizer """



    def __init__(self, model_name):
        """ Constructeur """
        # Définition du Tokenizer pré-entraîné à partir du modèle spécifié :
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)



    def preprocess_function(self, examples):
        """ Méthode de préparation des données """
        # Chargement des questions :
        questions = [q.strip() for q in examples["question"]]
        contexts = [' '.join(context) for context in examples["context"]]
        # Tokenization des questions :
        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        # Extraction des balises qui indiquent le début et la fin des questions Tokenizées :
        offset_mapping = inputs.pop("offset_mapping")
        # Extraction des réponses :
        answers = examples["answer"]
        # Initialisation des positions :
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]

            # Assurer que 'answer' est bien un dictionnaire avec les clés 'answer_start' et 'text'
            if isinstance(answer, dict) and "answer_start" in answer and "text" in answer:
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
            else:
                # Gérer les cas où la structure des réponses est incorrecte
                start_positions.append(0)
                end_positions.append(0)
                continue
            # Identifier la partie de la séquence correspondant à la question et la partie correspondant au contexte :
            sequence_ids = inputs.sequence_ids(i)

            # On itère sur les identifiants de séquence jusqu'à ce que l'on trouve le premier identifiant correspondant au contexte :
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            # On continue d'itérer sur les identifiants de séquence jusqu'à ce que l'on rencontre un identifiant différent de celui du contexte :
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # On garantit que les positions de début et de fin de la réponse sont correctement déterminées dans le contexte tokenisé.
            # 1- Si la réponse n'est pas entièrement à l'intérieur du contexte, on l'étiquette (0, 0) ce qui signifie que la réponse n'est pas pertinente pour cette question donnée :
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 2- Sinon, on récupère les positions de début et de fin des tokens, pour identifier le début et la fin
                idx = context_start
                # Recherche de la position de début de la réponse dans le contexte :
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1    # On itère jusqu'à ce que nous trouvions un token dont la position de début est supérieure à l'indice de début de la réponse
                start_positions.append(idx - 1)
                # Recherche de la position de fin de la réponse dans le contexte :
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1    # On itère jusqu'à ce que nous trouvions un token dont la position de fin est inférieure à l'indice de fin de la réponse
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


