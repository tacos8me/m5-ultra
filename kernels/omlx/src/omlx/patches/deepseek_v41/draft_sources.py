"""Opt-in bounded draft sources for native DSML tool traffic.

Only proposals change. The target still verifies every token with L=2..5.
Hidden retrieval uses completed earlier responses with the same tool schemas.
"""
import hashlib
import json
import os
from collections import OrderedDict, deque
import numpy as np

MODE = os.environ.get('DS41_EXTRA_DRAFT', 'off')

class Bank:
    def __init__(self):
        self.rows = OrderedDict()

    def add(self, schema, tokens, features):
        for pos, feature in features:
            suffix = tokens[pos + 1:pos + 5]
            if len(suffix) != 4:
                continue
            key = (schema, tokens[pos])
            bucket = self.rows.setdefault(key, deque(maxlen=64))
            bucket.append((feature.copy(), suffix))
            self.rows.move_to_end(key)
            if len(self.rows) > 128:
                self.rows.popitem(last=False)

    def propose(self, schema, anchor, feature):
        entries = self.rows.get((schema, anchor))
        if not entries:
            return None
        scores = np.stack([row[0] for row in entries]) @ feature
        best = int(np.argmax(scores))
        return list(entries[best][1]) if scores[best] >= .95 else None

class Tracker:
    def __init__(self, tokenizer, schemas, bank, first_token, mode):
        self.mode, self.bank = mode, bank
        self.schema = hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).digest()
        self.tokens, self.features = [first_token], []
        self.templates = {}
        self.anchor, self.feature = first_token, None
        self.done = False
        self.tool_end = tokenizer.encode("</｜DSML｜ calls>", add_special_tokens=False)
        texts = ['<｜DSML｜ calls>\n<｜DSML｜ invoke name="']
        for schema in schemas:
            properties = schema.get('parameters', {}).get('properties', {})
            names = list(properties)
            headers = [f'<｜DSML｜ parameter name="{name}" string="{str(properties[name].get("type") == "string").lower()}">' for name in names]
            texts.append(f'<｜DSML｜ invoke name="{schema["name"]}">\n' + (headers[0] if headers else '</｜DSML｜ invoke>\n</｜DSML｜ calls>'))
            texts.extend('</｜DSML｜ parameter>\n' + header for header in headers[1:])
            texts.append('</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            for n in range(4, len(ids)):
                self.templates.setdefault(tuple(ids[n-4:n]), []).append((ids[:n], ids[n:n+4]))

    def append(self, ids, hidden):
        start = len(self.tokens)
        self.tokens.extend(ids[:max(0, 4096 - start)])
        self.anchor = ids[-1]
        if self.mode in ('1', 'both', 'semantic'):
            # Existing target taps are already materialized by verification.
            features = np.asarray(hidden[0, :, ::64].tolist(), dtype=np.float32)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            self.feature = features[-1]
            for n, feature in enumerate(features):
                if start + n < 4092:
                    self.features.append((start + n, feature))

    def propose(self, tail, budget):
        if budget < 1:
            return None, None
        if self.mode in ('1', 'both', 'schema'):
            found = []
            for prefix, suffix in self.templates.get(tuple(tail[-4:]), []):
                if len(prefix) <= len(tail) and tail[-len(prefix):] == prefix:
                    found.append((len(prefix), suffix))
            if found:
                width = max(n for n, _ in found)
                choices = [s for n, s in found if n == width]
                common = list(choices[0])
                for other in choices[1:]:
                    n = 0
                    while n < min(len(common), len(other)) and common[n] == other[n]:
                        n += 1
                    common = common[:n]
                if len(common) >= min(4, budget):
                    return common[:min(4, budget)], 'schema'
        if self.feature is not None:
            draft = self.bank.propose(self.schema, self.anchor, self.feature)
            if draft:
                return draft[:min(4, budget)], 'retrieval'
        return None, None

    def finish(self, emitted, *, tool_filter=False):
        tokens = self.tokens[:emitted]
        if tool_filter:
            # The API parser completes calls before the model emits EOS.
            # Ordinary cancellation/filtering must not publish partial calls.
            end = self.tool_end
            tail = tokens[-(len(end) + 8):]
            if not any(tail[i:i + len(end)] == end for i in range(len(tail))):
                return
        if not self.done:
            self.bank.add(self.schema, tokens, self.features)
            self.done = True


def create(host, copy):
    if MODE not in ('1', 'both', 'schema', 'semantic'):
        return None
    tokenizer = getattr(host, '_ds41_draft_tokenizer', None)
    if tokenizer is None:
        return None
    text = tokenizer.decode(copy._buf[:min(copy.n - 1, 16384)].tolist())
    if '### Available Tool Schemas\n' not in text:
        return None
    block = text.split('### Available Tool Schemas\n', 1)[1].split('\n\n', 1)[0]
    schemas = []
    for line in block.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get('name'), str) and isinstance(item.get('parameters'), dict):
            schemas.append(item)
    if not schemas:
        return None
    bank = getattr(host, '_ds41_draft_bank', None)
    if bank is None:
        bank = Bank()
        host._ds41_draft_bank = bank
    return Tracker(tokenizer, schemas, bank, int(copy._buf[copy.n - 1]), MODE)
