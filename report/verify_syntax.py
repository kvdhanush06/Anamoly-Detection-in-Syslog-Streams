import glob
import re

tex_files = glob.glob('report/**/*.tex', recursive=True)
print(f"Found {len(tex_files)} tex files.")

for f in sorted(tex_files):
    if 'verify_syntax' in f:
        continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Strip comments: only % that are NOT preceded by backslash
    no_comments = re.sub(r'(?<!\\)%.*', '', content)
    
    # Strip lstlisting and verbatim blocks
    no_code = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '', no_comments, flags=re.DOTALL)
    no_code = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', no_code, flags=re.DOTALL)
    
    # Remove escaped braces \{ and \}
    cleaned = re.sub(r'\\\{', '', no_code)
    cleaned = re.sub(r'\\\}', '', cleaned)
    
    open_b = cleaned.count('{')
    close_b = cleaned.count('}')
    
    begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', cleaned)
    ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', cleaned)
    
    env_diff = sorted(begins) != sorted(ends)
    
    if open_b != close_b or env_diff:
        print(f"File {f}:")
        if open_b != close_b:
            print(f"  Braces mismatch: {open_b} open vs {close_b} close (diff: {open_b - close_b})")
            # Track line by line
            lines = no_code.split('\n')
            balance = 0
            for idx, line in enumerate(lines, 1):
                c_line = re.sub(r'(?<!\\)%.*', '', line)
                c_line = re.sub(r'\\\{', '', c_line)
                c_line = re.sub(r'\\\}', '', c_line)
                o = c_line.count('{')
                c = c_line.count('}')
                balance += (o - c)
                if o != c:
                    print(f"    Line {idx} (o={o}, c={c}, bal={balance}): {line.strip()[:80]}")
        if env_diff:
            print(f"  Environments mismatch: begins={begins}, ends={ends}")
    else:
        print(f"File {f}: OK")

# Citation and Reference Check
bib_file = 'report/references.bib'
with open(bib_file) as f:
    bib_content = f.read()

bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))

labels = set()
cites = set()
refs = set()

for f in sorted(tex_files):
    if 'verify_syntax' in f:
        continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    no_comments = re.sub(r'(?<!\\)%.*', '', content)
    for l in re.findall(r'\\label\{([^}]+)\}', no_comments):
        labels.add(l)
    for l in re.findall(r'label=\{([^}]+)\}', no_comments):
        labels.add(l)
    for c in re.findall(r'\\cite\{([^}]+)\}', no_comments):
        for single_c in c.split(','):
            cites.add(single_c.strip())
    for r in re.findall(r'\\ref\{([^}]+)\}', no_comments):
        refs.add(r)

print(f"Total labels defined: {len(labels)}")
print(f"Total internal \\ref calls: {len(refs)}")
print(f"Total \\cite calls: {len(cites)}")

missing_cites = cites - bib_keys
if missing_cites:
    print("MISSING BIB KEYS:", missing_cites)
else:
    print("All citations exist in references.bib!")

missing_refs = refs - labels
if missing_refs:
    print("MISSING LABELS FOR REFS:", missing_refs)
else:
    print("All internal \\ref cross-references have matching \\label definitions!")
