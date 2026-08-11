"""
Suffixarray + LCP som demo pa pulssekvenser (GMM-komponent per puls).
Ren stdlib. Naiv O(n^2 log n)-sortering - avsiktligt, for att koden ska
vara lasbar. Byt till pydivsufsort/SA-IS for skarp anvandning.
"""


def suffix_array(s):
    """Startpositioner sorterade efter suffix."""
    return sorted(range(len(s)), key=lambda i: s[i:])


def lcp_array(s, sa):
    """lcp[i] = antal gemensamma tecken mellan suffix sa[i] och sa[i-1]."""
    lcp = [0] * len(sa)
    for i in range(1, len(sa)):
        a, b = sa[i - 1], sa[i]
        n = 0
        while a + n < len(s) and b + n < len(s) and s[a + n] == s[b + n]:
            n += 1
        lcp[i] = n
    return lcp


def show_table(s, sa, lcp, width=28):
    print(f"  rank  SA  {'suffix':<{width}} LCP")
    print("  " + "-" * (width + 16))
    for rank, (pos, l) in enumerate(zip(sa, lcp)):
        suf = s[pos:]
        if len(suf) > width:
            suf = suf[: width - 1] + "~"
        mark = "  <--" if rank > 0 and l == max(lcp) else ""
        print(f"  {rank:>4}  {pos:>2}  {suf:<{width}} {l:>3}{mark}")


def repeats(s, min_len=3):
    """Maximala repeats: (monster, pos_a, pos_b, skift)."""
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    out = []
    for i in range(1, len(sa)):
        if lcp[i] >= min_len:
            a, b = sorted((sa[i - 1], sa[i]))
            out.append((s[a : a + lcp[i]], a, b, b - a))
    return sorted(out, key=lambda r: -len(r[0]))


def report(namn, s, min_len=3, table=False):
    print(f"\n=== {namn} ===")
    print(f"  S = {s}   (n={len(s)})")
    sa = suffix_array(s)
    if table:
        show_table(s, sa, lcp_array(s, sa))
    r = repeats(s, min_len)
    if not r:
        print(f"  Ingen repeat >= {min_len} tecken.")
        return
    monster, a, b, skift = r[0]
    print(f"  Langsta repeat : '{monster}' ({len(monster)} tecken)")
    print(f"  Positioner     : {a} och {b}")
    print(f"  Skift/period   : {skift}")
    # skift som aterkommer flera ganger = trolig cykellangd
    from collections import Counter
    c = Counter(x[3] for x in r)
    print(f"  Vanligaste skift: {c.most_common(3)}")


if __name__ == "__main__":
    # 1. Rent stagger-monster ABC, tre cykler
    report("Rent monster", "ABCABCAB", min_len=2, table=True)

    # 2. Langre sekvens, samma cykel
    report("Langre, brusfri", "ABCABCABCABCABC")

    # 3. Tappad puls: tva intervall slas ihop -> GMM ger fel komponent 'X'
    #    pa position 7. Monstret kapas i tva fragment.
    report("Med dropout (X)", "ABCABCAXCABCABC")

    # 4. Samma sekvens efter reparation: X kandes igen som ~2xPRI och
    #    splittades tillbaka till sina tva ursprungliga symboler.
    report("Efter reparation", "ABCABCABCABCABC")
