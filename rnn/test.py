import sys
from sentencepiece import SentencePieceProcessor
from .data import load_dataset, SequenceProvider

train_set = sys.argv[1]
train_file = open(train_set, 'rb')
train_iter = load_dataset(train_file)

tokenizer = SentencePieceProcessor()
tokenizer.Init(model_file='data/tokenizer9-8k.model')

def filter_text(data):
    for _count, set_name, text in data:
        if set_name not in (
            'BookCorpus2', 'Books3', 'Enron Emails', 'Gutenberg (PG-19)',
            'HackerNews', 'OpenWebText2', 'Ubuntu IRC', 'Wikipedia (en)'
        ):
            continue

        yield text

seq_provider = SequenceProvider(
    n_sequences=2048,
    text_loader=filter_text(train_iter),
    tokenizer=tokenizer,
    short_ctx_len=8,
    target_seq_len=100,
)

while True:
    seq = seq_provider.next_sequence()
    print('\nseq:', tokenizer.Decode(seq), flush=False)
