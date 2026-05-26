#
# Named-entity recognition using BERT
# Dataset: https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion
#
# Evaluation: token-level (accuracy / balanced accuracy / sklearn classification
# report) AND entity-level (precision / recall / F1 via seqeval).
#
# Note: seqeval requires:
#   pip install seqeval
#

# dependencies
import torch
import torch.optim as optim
from transformers import BertForTokenClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from tqdm.auto import tqdm  # modern, works in notebooks & scripts

# hyper-parameters
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5

# the path of the data files
base_path = '/conll003-englishversion/'

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read the data files
def load_sentences(filepath):

    sentences = []
    tokens = []
    pos_tags = []
    chunk_tags = []
    ner_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # sentence boundary
            if (line.startswith('-DOCSTART-') or line.strip() == ''):
                if len(tokens) > 0:
                    sentences.append({
                        'tokens': tokens,
                        'pos_tags': pos_tags,
                        'chunk_tags': chunk_tags,
                        'ner_tags': ner_tags
                    })
                    tokens = []
                    pos_tags = []
                    chunk_tags = []
                    ner_tags = []
            else:
                l = line.strip().split(' ')
                if len(l) >= 4:
                    tokens.append(l[0])
                    pos_tags.append(l[1])
                    chunk_tags.append(l[2])
                    ner_tags.append(l[3])
    # last sentence if file doesn't end with blank line
    if len(tokens) > 0:
        sentences.append({
            'tokens': tokens,
            'pos_tags': pos_tags,
            'chunk_tags': chunk_tags,
            'ner_tags': ner_tags
        })
    return sentences

print('loading data')
train_sentences = load_sentences(base_path + 'train.txt')
test_sentences = load_sentences(base_path + 'test.txt')
valid_sentences = load_sentences(base_path + 'valid.txt')

# build tag set and label mappings
all_tags = sorted({tag for s in train_sentences for tag in s['ner_tags']})
label2id = {tag: i for i, tag in enumerate(all_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(all_tags)
print('Tagset size:', num_labels)
print('Tags:', all_tags)

# load BERT tokenizer
bert_version = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(bert_version)

# map tokens and tags to token ids and label ids
def align_label(tokens, labels):
    """
    tokens: tokenizer output (BatchEncoding) with word_ids()
    labels: list of original word-level labels for a single sentence
    """
    word_ids = tokens.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            # special tokens
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # first subword of a word
            label_ids.append(label2id.get(labels[word_idx], -100))
        else:
            # subsequent subword of the same word (ignore for loss)
            label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def encode(sentence):
    encodings = tokenizer(
        sentence['tokens'],
        truncation=True,
        padding='max_length',
        is_split_into_words=True,
        return_tensors='pt'  # get tensors directly
    )
    labels = align_label(encodings, sentence['ner_tags'])

    return {
        'input_ids': encodings['input_ids'].squeeze(0),        # [seq_len]
        'attention_mask': encodings['attention_mask'].squeeze(0),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

print('encoding data')
train_dataset = [encode(sentence) for sentence in train_sentences]
valid_dataset = [encode(sentence) for sentence in valid_sentences]
test_dataset = [encode(sentence) for sentence in test_sentences]

# PyTorch Dataset wrapper for better compatibility with DataLoader
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = InputDataset(train_dataset)
valid_dataset = InputDataset(valid_dataset)
test_dataset = InputDataset(test_dataset)

# initialize the model including a classification layer with num_labels classes
print('initializing the model')
model = BertForTokenClassification.from_pretrained(
    bert_version,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(device)
optimizer = optim.AdamW(params=model.parameters(), lr=LR)

# prepare batches of data
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# evaluate the performance of the model
def EvaluateModel(model, data_loader):
    """
    Returns:
      Y_actual_flat, Y_preds_flat : 1D numpy arrays of label ids over all
        evaluated tokens (special/subword tokens excluded). Used for token-level
        accuracy, balanced accuracy and the sklearn classification report.
      y_true_tags, y_pred_tags    : lists of lists of BIO tag strings, one
        sub-list per sentence. Used for entity-level evaluation with seqeval.
    """
    model.eval()
    Y_actual_flat, Y_preds_flat = [], []
    y_true_tags, y_pred_tags = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # move the batch tensors to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]

            # iterate through the examples in the batch
            for idx in range(batch['labels'].size(0)):
                true_values_all = batch['labels'][idx]
                mask = (true_values_all != -100)

                true_values = true_values_all[mask]
                pred_values = preds[idx][mask]

                # accumulate flat tensors for token-level metrics
                Y_actual_flat.append(true_values)
                Y_preds_flat.append(pred_values)

                # accumulate per-sentence BIO tag strings for seqeval
                true_tags_sent = [id2label[i] for i in true_values.tolist()]
                pred_tags_sent = [id2label[i] for i in pred_values.tolist()]
                y_true_tags.append(true_tags_sent)
                y_pred_tags.append(pred_tags_sent)

    Y_actual_flat = torch.cat(Y_actual_flat).detach().cpu().numpy()
    Y_preds_flat = torch.cat(Y_preds_flat).detach().cpu().numpy()

    return Y_actual_flat, Y_preds_flat, y_true_tags, y_pred_tags


def report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags, split_name):
    """Print both token-level and entity-level metrics."""
    print(f"\n=== {split_name} — Token-level metrics ===")
    print("Accuracy          : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
    print("Balanced accuracy : {:.3f}".format(balanced_accuracy_score(Y_actual, Y_preds)))

    print(f"\n=== {split_name} — Entity-level metrics (seqeval) ===")
    print("Precision : {:.3f}".format(seqeval_precision(y_true_tags, y_pred_tags)))
    print("Recall    : {:.3f}".format(seqeval_recall(y_true_tags, y_pred_tags)))
    print("F1        : {:.3f}".format(seqeval_f1(y_true_tags, y_pred_tags)))


# train the model
print('training the model')
for epoch in range(EPOCHS):
    model.train()
    print(f'epoch {epoch + 1}/{EPOCHS}')
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate performance on validation set
    Y_actual, Y_preds, y_true_tags, y_pred_tags = EvaluateModel(model, valid_loader)
    report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags,
                   split_name=f"Validation (epoch {epoch + 1})")

print('\napplying the model to the test set')
Y_actual, Y_preds, y_true_tags, y_pred_tags = EvaluateModel(model, test_loader)

report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags, split_name="Test")

# detailed token-level classification report (per-tag, including 'O')
label_ids_sorted = list(range(num_labels))
target_names = [id2label[i] for i in label_ids_sorted]
print("\n=== Test — Token-level classification report (sklearn) ===")
print(
    classification_report(
        Y_actual,
        Y_preds,
        labels=label_ids_sorted,
        target_names=target_names,
        zero_division=0
    )
)

# detailed entity-level classification report (per entity type, no 'O')
print("=== Test — Entity-level classification report (seqeval) ===")
print(
    seqeval_report(
        y_true_tags,
        y_pred_tags,
        digits=3,
        zero_division=0
    )
)