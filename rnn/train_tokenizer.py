import io
import json
import sys

import sentencepiece as spm
import tqdm
import zstandard


class TqdmFileWrapper:
    def __init__(self, filename: str):
        self.file = open(filename, 'rb')
        self.file.seek(0, io.SEEK_END)
        file_len = self.file.tell()
        self.file.seek(0, io.SEEK_SET)
        self.pbar = tqdm.tqdm(total=file_len)

    def read(self, n: int) -> bytes:
        buf = self.file.read(n)
        self.pbar.update(len(buf))
        return buf

def main():
    out_prefix = sys.argv[1]
    in_file = sys.argv[2]

    file_wrap = TqdmFileWrapper(in_file)
    dctx = zstandard.ZstdDecompressor()
    def generate_sentences():
        sentences = 0
        buffer = bytearray()
        for chunk in dctx.read_to_iter(file_wrap):
            buffer += chunk
            lines = buffer.split(b'\n')
            buffer[:] = lines[-1]
            for line in lines[0:-1]:
                obj = json.loads(line)
                set_name = obj['meta']['pile_set_name']
                if set_name not in (
                    'BookCorpus2', 'Books3', 'Enron Emails', 'Gutenberg (PG-19)',
                    'HackerNews', 'OpenWebText2', 'Ubuntu IRC', 'Wikipedia (en)'
                ):
                    continue
                yield obj['text']
                sentences += 1
                if sentences % 100 == 0:
                    file_wrap.pbar.set_description(f'sentences: {sentences}', refresh=False)

                if sentences >= 262144:
                    file_wrap.pbar.set_description(
                        f'sentences: {sentences}, input loading done', refresh=True
                    )
                    return

    spm.SentencePieceTrainer.Train(
        sentence_iterator=generate_sentences(),
        model_type='unigram',
        model_prefix=out_prefix,
        vocab_size=16000,
        max_sentence_length=1048576,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name='nfkc',
        byte_fallback=True,
        character_coverage=0.995,
        # force characters to exist, or character_coverage will disappear them
        required_chars=''.join([
            '0123456789',
            'abcdefghijklmnopqrstuvwxyz',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '`~!@#$%^&*-_=+|()[]{}<>.,:;/?"\'\\\n\t',
            # misc unicode chars that might be common
            '°‘’“”',
        ]),
        # enable all special tokens
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        control_symbols='<del>',
    )

if __name__ == '__main__':
    main()
