import spacy
import inflect

nlp = spacy.load("en_core_web_sm")
infl = inflect.engine()

ADJ_TAGS = {"JJ", "JJR", "JJS"}


def attack_replace_adjectives(caption: str, replacement_adj: str = "") -> str:
    doc = nlp(caption)
    out = []
    for tok in doc:
        if tok.tag_ in ADJ_TAGS:
            # If the replacement is multiword (e.g., "more fluffy"), match casing of the first word only
            parts = replacement_adj.split(" ")
            parts[0] = _match_casing(tok.text, parts[0])
            replacement_adj = " ".join(parts)
            # Preserve original whitespace
            out.append(replacement_adj + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    return "".join(out)