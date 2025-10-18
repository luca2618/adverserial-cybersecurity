ADJ_TAGS = {"JJ", "JJR", "JJS"}

def _match_casing(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl.title()
    return repl

def _degree_wrap(base_adj: str, tag: str) -> str:
    # Safer than trying to inflect irregulars: use "more/most" wrappers
    if tag == "JJR":
        return f"more {base_adj}"
    if tag == "JJS":
        return f"most {base_adj}"
    return base_adj

def attack_replace_adjectives(caption: str, replacement_adj: str = "") -> str:
    doc = nlp(caption)
    out = []
    for tok in doc:
        if tok.tag_ in ADJ_TAGS:
            repl = _degree_wrap(replacement_adj, tok.tag_)
            # If the replacement is multiword (e.g., "more fluffy"), match casing of the first word only
            parts = repl.split(" ")
            parts[0] = _match_casing(tok.text, parts[0])
            repl = " ".join(parts)
            # Preserve original whitespace
            out.append(repl + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    return "".join(out)