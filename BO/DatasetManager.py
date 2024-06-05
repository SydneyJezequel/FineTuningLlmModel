from datasets import load_dataset
import config






class DatasetManager:
    """ Classe du Gestionnaire de Dataset """



    def __init__(self, dataset_name, sub_dataset_name):
        """ Constructeur """
        self.dataset = load_dataset(dataset_name, sub_dataset_name)
        print("dataset : ", self.dataset)
        print("clés du dataset : ", self.dataset.keys())



    def prepare_dataset(self, dataset, dataset_size, type):
        """ Méthode qui structure les données des datasets avant la Tokenization """
        print("dataset : ", dataset)
        dataset = dataset[type]
        prepared_dataset = dataset.shuffle(seed=config.SEED_VALUE).select(range(dataset_size))
        return prepared_dataset



    def preprocess_function(self, examples, tokenizer):
        """ Méthode de Tokenization des datasets """
        questions = [q.strip() for q in examples["question"]]
        contexts = [' '.join(context) for context in examples["context"]]
        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answer"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            print(f"Answer: {answer}")

            # Transformer la réponse si ce n'est pas déjà un dictionnaire
            if isinstance(answer, str):
                answer = {"answer_start": [0], "text": [answer]}  # Assurez-vous d'avoir la position correcte

            # Assurer que 'answer' est bien un dictionnaire avec les clés 'answer_start' et 'text'
            if isinstance(answer, dict) and "answer_start" in answer and "text" in answer:
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
            else:
                # Gérer les cas où la structure des réponses est incorrecte
                print(f"Incorrect answer format: {answer}")
                start_positions.append(0)
                end_positions.append(0)
                continue
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

