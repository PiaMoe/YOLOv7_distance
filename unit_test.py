def encode(score, distance, heading):
    # Score auf 2 Nachkommastellen beschränken
    score = round(score * 100) / 100
    # Distance & Heading anhängen
    encoded = score + distance / 1e5 + heading / 1e8
    return float(encoded)  # float32-kompatibel

def truncate(num, decimals):
    factor = 10.0 ** decimals
    return int(num * factor) / factor


def decode(encoded):
    # Score (2 Dezimalstellen)
    score = truncate(encoded, 2)

    # Rest = Distance + Heading
    rest = encoded - score

    # Distance aus den ersten 3 Nachkommastellen
    distance = int(rest * 1e5)

    # Heading aus den nächsten 3 Nachkommastellen
    heading = int(truncate((rest * 1e8) % 1000,2))
    #heading = 0  # Heading wird aktuell nicht kodiert
    return score, distance, heading

tests = [
    (0.87, 923, 45),
    (1.00, 999, 359),
    (0.00, 0, 0),
    (0.95, 1000, 123)
]

for s, d, h in tests:
    enc = encode(s, d, h)
    dec = decode(enc)
    print(f"Input: s={s}, d={d}, h={h} | Encoded={enc} | Decoded={dec}")
