"""Lab 6 - Huffman compression (reference solution)."""
import heapq, math
from collections import Counter


class Node:
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq, self.sym, self.left, self.right = freq, sym, left, right


def build_tree(freqs):
    """Greedy rule: repeatedly merge the two least frequent nodes.
    The counter makes tie-breaking deterministic (older node first)."""
    heap, uid = [], 0
    for sym, f in sorted(freqs.items()):
        heap.append((f, uid, Node(f, sym))); uid += 1
    heapq.heapify(heap)
    if len(heap) == 1:                       # degenerate text with a single symbol
        f, _, only = heap[0]
        return Node(f, None, only, None)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, uid, Node(f1 + f2, None, a, b))); uid += 1
    return heap[0][2]


def make_codes(root):
    codes = {}
    def walk(node, prefix):
        if node.sym is not None:
            codes[node.sym] = prefix or "0"
            return
        if node.left:  walk(node.left, prefix + "0")
        if node.right: walk(node.right, prefix + "1")
    walk(root, "")
    return codes


def encode(text, codes):
    return "".join(codes[c] for c in text)


def decode(bits, root):
    out, node = [], root
    for b in bits:
        node = node.left if b == "0" else node.right
        if node.sym is not None:             # reached a leaf -> emit symbol, restart at root
            out.append(node.sym)
            node = root
    return "".join(out)


def is_prefix_free(codes):
    cs = sorted(codes.values())
    return all(not cs[i + 1].startswith(cs[i]) for i in range(len(cs) - 1))


def metrics(text, codes, bits_per_char=8):
    freqs = Counter(text)
    raw = len(text) * bits_per_char
    payload = sum(freqs[s] * len(codes[s]) for s in freqs)
    # simple metadata estimate: for each symbol store the symbol (8 bits) + its code length (8 bits)
    meta = len(codes) * 16
    n = len(text)
    entropy = -sum(f / n * math.log2(f / n) for f in freqs.values())
    return dict(chars=n, symbols=len(freqs), raw=raw, payload=payload, meta=meta,
                ratio=payload / raw, saving=(1 - payload / raw) * 100,
                total=payload + meta, ratio_total=(payload + meta) / raw,
                avg_len=payload / n, entropy=entropy)


def run(text):
    freqs = Counter(text)
    root = build_tree(freqs)
    codes = make_codes(root)
    bits = encode(text, codes)
    assert decode(bits, root) == text            # exact round trip
    assert is_prefix_free(codes)
    return freqs, codes, bits, metrics(text, codes)


def warmup():
    """A:40 B:25 C:20 D:10 E:5 -> list of merge steps, codes, cost."""
    freqs = {"A": 40, "B": 25, "C": 20, "D": 10, "E": 5}
    steps = []
    pool = [(f, s) for s, f in sorted(freqs.items(), key=lambda kv: kv[1])]
    while len(pool) > 1:
        pool.sort(key=lambda x: x[0])
        (f1, s1), (f2, s2) = pool[0], pool[1]
        steps.append((s1, f1, s2, f2, f1 + f2))
        pool = pool[2:] + [(f1 + f2, "(" + s1 + s2 + ")")]
    root = build_tree(freqs)
    codes = make_codes(root)
    cost = sum(freqs[s] * len(codes[s]) for s in freqs)
    return steps, codes, cost


def report():
    from display import banner, section, note, table, ci, figure, save, show_all
    banner("LAB 6 — Compression de Huffman")
    section("1. Échauffement : A:40  B:25  C:20  D:10  E:5")
    steps, codes, cost = warmup()
    table(["Étape", "Les 2 moins fréquents", "Nouveau nœud"], [[i + 1, f"{a} ({fa}) + {b} ({fb})", fa + fb] for i, (a, fa, b, fb, _) in enumerate(steps)])
    fr = {"A": 40, "B": 25, "C": 20, "D": 10, "E": 5}
    table(["Symbole", "Fréquence", "Code", "Longueur", "fréq × longueur"], [[s, fr[s], codes[s], len(codes[s]), fr[s] * len(codes[s])] for s in "ABCDE"] + [["Total", 100, "", "", cost]])
    note(f"Payload = {cost} bits (moyenne {cost/100:.2f} bits/symbole) vs 300 bits b code fixe 3 bits ⇒ économie 30 %.")
    section("2. Deux textes : aller-retour exact + métriques")
    texts = {"skewed": "aaaaaaaaaabbbbbcccd" * 50, "divers": "the quick brown fox jumps over the lazy dog " * 5}
    M = {}
    for k, t in texts.items():
        M[k] = run(t)[3]
    table(["Métrique", "Texte skewed", "Texte divers"], [
        ["caractères / symboles", f"{M['skewed']['chars']} / {M['skewed']['symbols']}", f"{M['divers']['chars']} / {M['divers']['symbols']}"],
        ["taille brute (8 bits/car)", ci(M["skewed"]["raw"]) + " bits", ci(M["divers"]["raw"]) + " bits"],
        ["payload Huffman", ci(M["skewed"]["payload"]) + " bits", ci(M["divers"]["payload"]) + " bits"],
        ["ratio payload/brut", f"{M['skewed']['ratio']:.3f}", f"{M['divers']['ratio']:.3f}"],
        ["longueur moyenne", f"{M['skewed']['avg_len']:.3f}", f"{M['divers']['avg_len']:.3f}"],
        ["entropie H (borne)", f"{M['skewed']['entropy']:.3f}", f"{M['divers']['entropy']:.3f}"],
        ["métadonnées estimées", f"{M['skewed']['meta']} bits", f"{M['divers']['meta']} bits"],
        ["ratio total (avec métadonnées)", f"{M['skewed']['ratio_total']:.3f}", f"{M['divers']['ratio_total']:.3f}"]])
    note("decode(encode(text)) == text vérifié (assert). Kol ma distribution skewed, kol ma H sghira w compression mlieha. Métadonnées yzidou overhead.")
    fig, ax = figure(1, 3, figsize=(14, 4))
    if fig:
        ax[0].bar(list("ABCDE"), [len(codes[s]) for s in "ABCDE"], color="#2A7F8E")
        for i, s in enumerate("ABCDE"): ax[0].text(i, len(codes[s]) + 0.05, f"{fr[s]}%", ha="center", fontsize=8)
        ax[0].set_ylabel("longueur du code (bits)"); ax[0].set_title("Fréquent ⇒ code court (warm-up)")
        names = list(M); x = range(len(names)); w = 0.27
        ax[1].bar([i - w for i in x], [M[k]["raw"] for k in names], w, label="brut (8 bits)")
        ax[1].bar(list(x), [M[k]["payload"] for k in names], w, label="payload Huffman")
        ax[1].bar([i + w for i in x], [M[k]["total"] for k in names], w, label="payload + métadonnées")
        ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names); ax[1].set_ylabel("bits"); ax[1].set_title("Tailles"); ax[1].legend(fontsize=8)
        ax[2].bar([i - 0.2 for i in x], [M[k]["entropy"] for k in names], 0.4, label="entropie H")
        ax[2].bar([i + 0.2 for i in x], [M[k]["avg_len"] for k in names], 0.4, label="longueur moyenne Huffman")
        ax[2].set_xticks(list(x)); ax[2].set_xticklabels(names); ax[2].set_ylabel("bits / symbole"); ax[2].set_title("H ≤ longueur moyenne < H + 1"); ax[2].legend(fontsize=8)
        save(fig, "lab06_huffman")
    show_all()


if __name__ == "__main__":
    report()
