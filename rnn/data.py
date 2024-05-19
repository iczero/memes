import json
import warnings
import zstandard

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
