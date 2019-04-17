from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.common.params import Params

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

from allennlp.data.token_indexers import TokenCharactersIndexer
import unicodedata, glob, os, string, random

# Methods from pytorch tutorial

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def findFiles(path): return glob.glob(path)

class Dataset_Reader(DatasetReader):
    
    # Implementing data reader from character RNN tutorial
    def __init__(self, token_indexers: Dict[str, SingleIdTokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Character indexers based on AllenNLP tutorial
        self.token_character_indexers = {"token_characters" : TokenCharactersIndexer()}
        
    def text_to_instance(self, tokens: List[str], tags: List[str] = None) -> Instance:
        name_tokens = [Token(name) for name in tokens]
        name_token_field = TextField(name_tokens, self.token_indexers)
        # Same logic for character tokens and fields as well
        fields = {"tokens": name_token_field}
        name_token_character_field = TextField(name_tokens, self.token_character_indexers)
        fields["token_characters"] = name_token_character_field
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=name_token_field)
            fields["labels"] = label_field

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        concatenated_pair = []
        for filename in findFiles(file_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = readLines(filename)
            category_lines[category] = lines
            concatenated_pair.extend([(word, category) for word in lines])
        # Tried using names in bigger (language based) splits, did not work out, will be reported in the report
        # Make smaller splits someway
        # Trying making splits of 30 names. for that shuffle the names, otherwise same problem as before
        np.random.shuffle(concatenated_pair)
        #print(concatenated_pair)
        # Code for chunks from "https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"
        for i in range(0, len(concatenated_pair), 30):
            shuffeled_split = concatenated_pair[i:i+30]
            yield self.text_to_instance([pair[0] for pair in shuffeled_split], [pair[1] for pair in shuffeled_split])

dataset_reader = Dataset_Reader()

category_lines = {}
all_categories = []
names_data = dataset_reader.read('data/names/*.txt')

class NamesClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                token_characters: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        # Had to change the input for embeddings, give it as an input of dictionary with tokens and token characters
        # use ** to unpack them
        embeddings = self.word_embeddings({**tokens, **token_characters})
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

# Since we don't have different files for validation, use sklearn to split the data
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(names_data, test_size=0.2)
vocab = Vocabulary.from_instances(names_data)

vocab

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder

# Define the model
WORD_EMBEDDING_DIM = 3
CHAR_EMBEDDING_DIM = 3
HIDDEN_DIM = 6
EMBEDDING_DIM = WORD_EMBEDDING_DIM + CHAR_EMBEDDING_DIM

# Embeddings for characters and then for names
character_encoder = PytorchSeq2VecWrapper(torch.nn.RNN(CHAR_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, batch_first=True))
token_character_embedding = Embedding(num_embeddings=vocab.get_vocab_size('token_characters'),
                            embedding_dim=WORD_EMBEDDING_DIM)
character_embeddings = TokenCharactersEncoder(token_character_embedding, character_encoder)

# Now names
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=WORD_EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding, "token_characters" : character_embeddings})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = NamesClassifier(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

# Train the model - 30 epochs seem to give a pretty good baseline accuracy - 0.7 val accuracy
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("tokens", "num_tokens"), ("token_characters", "num_token_characters")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_set,
                  validation_dataset=val_set,
                  patience=10,
                  num_epochs=2, 
                  cuda_device=cuda_device)
trainer.train()

# Manually test predictions
from allennlp.predictors import Predictor

class OwnPredictor(Predictor):
    # Takes as input model and dataset
    
    # define function for prediction, code similar to tutorial
    def predict_language(self, names):
        tag_logits = self.predict_instance(self._dataset_reader.text_to_instance(names)) 
        # No method to get instances other than using hidden variables
        tag_ids = np.argmax(tag_logits['tag_logits'], axis=-1)
        return [self._model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]

predictor = OwnPredictor(model, dataset_reader=dataset_reader)
print(predictor.predict_language(["Kuznetsov", "Schneider", "Washington", "Lindemann", "Müller"]))
