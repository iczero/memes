import json
import multiprocessing
import warnings
from collections.abc import Iterable

import numpy as np
import tqdm
import zstandard
from sentencepiece import SentencePieceProcessor

TOKENIZE_BATCH_SIZE = 2048

def load_dataset(in_stream):
    dctx = zstandard.ZstdDecompressor()
    count = 0
    buffer = bytearray()
    for chunk in dctx.read_to_iter(in_stream):
        buffer += chunk
        lines = buffer.split(b'\n')
        buffer[:] = lines[-1]
        for line in lines[0:-1]:
            obj = json.loads(line)
            text = obj['text']
            set_name = obj['meta']['pile_set_name']
            if len(text) > 0:
                yield count, set_name, text
            count += 1

    if len(buffer) > 0:
        warnings.warn('dataset file did not end with newline')

class SequenceProvider:
    sequences: list[np.ndarray]
    tokenizer: SentencePieceProcessor
    short_ctx_len: int

    def __init__(self, tokenizer: SentencePieceProcessor, short_ctx_len: int):
        self.tokenizer = tokenizer
        self.short_ctx_len = short_ctx_len
        self.pad_token = self.tokenizer['<pad>']
        self.text_start_token = self.tokenizer['<s>']
        self.sequences = []

    def wrap_sequence(self, tokens: list[int]):
        pad_start = [self.pad_token] * (self.short_ctx_len - 1) + [self.text_start_token]
        return pad_start + tokens

    # TODO: maybe return numpy or torch array instead?
    def next_sequence(self):
        return self.wrap_sequence(self.sequences.pop().tolist())

class Preprocessor:
    tokenizer: SentencePieceProcessor
    target_seq_len: int

    pad_token: int
    text_start_token: int
    text_end_token: int

    tokens_count: int = 0
    sequences: list[np.ndarray]
    randgen: np.random.Generator

    def __init__(
        self,
        tokenizer: SentencePieceProcessor,
        target_seq_len: int,
        randseed: np.random.SeedSequence | None = None,
    ):
        self.tokenizer = tokenizer
        self.target_seq_len = target_seq_len

        self.pad_token = self.tokenizer['<pad>']
        self.text_start_token = self.tokenizer['<s>']
        self.text_end_token = self.tokenizer['</s>']

        self.sequences = []
        self.randgen = np.random.default_rng(randseed)

    def _load_old(self, text_loader: Iterable[str]):
        self.tokens_count = 0
        self.sequences.clear()
        sequences = self.sequences
        pbar = tqdm.tqdm(
            leave=True, disable=None, unit='token',
            desc='reading sequences'
        )

        doc_count = 0
        try:
            has_next = True
            while has_next:
                # load sequences
                batch_document = []
                for _ in range(TOKENIZE_BATCH_SIZE):
                    try:
                        document = next(text_loader)
                    except StopIteration:
                        has_next = False
                        break

                    batch_document.append(document)

                tokenized = self.tokenizer.Encode(batch_document)
                for doc in tokenized:
                    doc_count += 1
                    pos = 0
                    while pos < len(doc):
                        tokens = doc[pos : pos + self.target_seq_len]
                        if len(tokens) < self.target_seq_len:
                            tokens.append(self.text_end_token)

                        self.tokens_count += len(tokens)
                        sequences.append(np.array(tokens, dtype=np.uint16))
                        pbar.update(self.tokens_count - pbar.n)
                        pos += self.target_seq_len

            pbar.close()
            print(f'loaded {doc_count} documents, {self.tokens_count} tokens')
        except:
            # don't clear bar on error
            pbar.leave = True
            raise

        # shuffle
        self.randgen.shuffle(sequences)

    def load(self, text_loader: Iterable[str]):
        self.tokens_count = 0
        self.sequences.clear()
        sequences = self.sequences
        pbar = tqdm.tqdm(
            leave=True, disable=None, unit='token',
            desc='reading sequences'
        )

        doc_count = 0
        has_next = True
        while has_next:
            # load sequences
            batch_document = []
            for _ in range(TOKENIZE_BATCH_SIZE):
                try:
                    document = next(text_loader)
                except StopIteration:
                    has_next = False
                    break

                batch_document.append(document)

            tokenized = self.tokenizer.Encode(batch_document)
            for doc in tokenized:
                doc_count += 1
                pos = 0
                while pos < len(doc):
                    tokens = doc[pos : pos + self.target_seq_len]
                    if len(tokens) < self.target_seq_len:
                        tokens.append(self.text_end_token)

                    self.tokens_count += len(tokens)
                    sequences.append(np.array(tokens, dtype=np.uint16))
                    pbar.update(self.tokens_count - pbar.n)
                    pos += self.target_seq_len

        pbar.close()
        print(f'loaded {doc_count} documents, {self.tokens_count} tokens')

        # shuffle
        self.randgen.shuffle(sequences)

def filter_text(data):
    for _count, set_name, text in data:
        if set_name not in (
            'BookCorpus2', 'Books3', 'Enron Emails', 'Gutenberg (PG-19)',
            'HackerNews', 'OpenWebText2', 'Ubuntu IRC', 'Wikipedia (en)'
        ):
            continue

        yield text

def load_worker(queue: multiprocessing.Queue, filename: str):
    with open(filename, 'rb') as f:
        loader = filter_text(load_dataset(f))
        for entry in loader:
            queue.put(entry)

        queue.put(None)

def subprocess_loader(filename: str) -> Iterable[str]:
    queue = multiprocessing.Queue(TOKENIZE_BATCH_SIZE * 2)
    proc = multiprocessing.Process(target=load_worker, args=(queue, filename))
    proc.start()

    while True:
        out = queue.get()
        if out is None:
            break

        yield out

    proc.join()

if __name__ == '__main__':
    pass
