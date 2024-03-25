import json
import re
import warnings
from collections.abc import Iterable

import numpy as np
import zstandard
from sentencepiece import SentencePieceProcessor

END_PARAGRAPH = re.compile(r'\n(?:[-=~]*\n+)?')

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
    n_sequences: int
    states: list[Iterable[list[int]]]
    text_loader: Iterable[str]
    tokenizer: SentencePieceProcessor
    target_seq_len: int
    seq_len_high_threshold: int
    seq_len_low_threshold: int
    text_fragment_max_len: int

    pad_token: int
    text_start_token: int
    text_end_token: int

    def __init__(
        self,
        n_sequences: int,
        text_loader: Iterable[str],
        tokenizer: SentencePieceProcessor,
        short_ctx_len: int,
        target_seq_len: int,
    ):
        self.n_sequences = n_sequences
        self.text_loader = text_loader
        self.tokenizer = tokenizer
        self.short_ctx_len = short_ctx_len
        self.target_seq_len = target_seq_len
        self.states = [None] * n_sequences
        self.seq_len_high_threshold = int(target_seq_len * 1.1)
        self.seq_len_low_threshold = int(target_seq_len * 0.8)
        self.text_fragment_max_len = int(target_seq_len * 512)

        self.pad_token = self.tokenizer['<pad>']
        self.text_start_token = self.tokenizer['<s>']
        self.text_end_token = self.tokenizer['</s>']

    def next_document(self) -> str:
        try:
            return next(self.text_loader)
        except StopIteration:
            # TODO: deal with this somehow
            raise RuntimeError('loader exhausted')

    def wrap_sequence(self, tokens: list[int]):
        pad_start = [self.pad_token] * (self.short_ctx_len - 1) + [self.text_start_token]
        last = [self.text_end_token]
        return pad_start + tokens + last

    def make_sequence_slicer(self, text: str):
        current_pos = 0
        while len(text) - current_pos > 0:
            tokens = self.tokenizer.Encode(
                text[current_pos : current_pos + self.text_fragment_max_len]
            )
            if len(tokens) < self.seq_len_low_threshold:
                if current_pos == 0:
                    # send if it isn't something we've truncated
                    yield self.wrap_sequence(tokens)
                # otherwise discard
                return

            if len(tokens) < self.seq_len_high_threshold:
                yield self.wrap_sequence(tokens)
                return

            should_continue = False
            end_matches = END_PARAGRAPH.finditer(text, pos=current_pos)
            for end_match in end_matches:
                fragment = text[current_pos : end_match.start()]
                tokens = self.tokenizer.Encode(fragment)
                if len(tokens) < self.seq_len_low_threshold:
                    # don't send a fragment that's too short
                    continue

                should_continue = True
                current_pos = end_match.end()
                yield self.wrap_sequence(tokens)
                break

            if not should_continue:
                # either no paragraph separator or no good match
                yield self.wrap_sequence(tokens)
                return

    def next_sequence(self):
        index = np.random.randint(0, len(self.states))
        while True:
            current = self.states[index]
            try:
                if current is None:
                    raise StopIteration()

                return next(current)
            except StopIteration:
                self.states[index] = self.make_sequence_slicer(self.next_document())
