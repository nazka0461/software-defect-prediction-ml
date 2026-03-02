import radon.metrics as rm
from radon.raw import analyze
import radon.complexity as rc
import tokenize, io, token

def extract_metrics(code: str) -> dict:
    raw = analyze(code)
    hal = rm.h_visit(code)[0]
    cc  = rc.cc_visit(code)

    # ev(g) and iv(g) approximation
    ev_g = sum(1 for f in cc if f.complexity == 1)   # structured functions
    iv_g = sum(f.complexity for f in cc if f.is_method)

    # uniq_Op: count distinct operator tokens
    ops = set()
    try:
        for tok_type, tok_val, *_ in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok_type == token.OP:
                ops.add(tok_val)
    except:
        pass

    v = hal.volume
    d = hal.difficulty
    l = 1.0 / d if d not in (0, None) else 0.0          # level = 1 / difficulty
    i = v / d if d not in (0, None) else 0.0            # intelligence content = V / D
    e = hal.effort
    b = hal.bugs
    t = hal.time

    return {
        'loc'               : raw.loc,
        'ev(g)'             : ev_g,
        'iv(g)'             : iv_g,
        'v'                 : v,
        'l'                 : l,
        'd'                 : d,
        'i'                 : i,
        'e'                 : e,
        'b'                 : b,
        't'                 : t,
        'lOComment'         : raw.comments,
        'lOBlank'           : raw.blank,
        'locCodeAndComment' : raw.sloc + raw.comments,
        'uniq_Op'           : len(ops)
    }

code = open("src/data/load_promise_nasa.py").read()
metrics = extract_metrics(code)
print(metrics)