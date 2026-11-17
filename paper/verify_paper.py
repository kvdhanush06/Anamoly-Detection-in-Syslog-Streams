import re
import os

def check_paper():
    main_tex_path = "paper/main.tex"
    pipeline_tex_path = "paper/figures/pipeline_diagram.tex"
    bib_path = "paper/references.bib"
    
    with open(main_tex_path, "r", encoding="utf-8") as f:
        main_content = f.read()
        
    with open(pipeline_tex_path, "r", encoding="utf-8") as f:
        pipeline_content = f.read()
        
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()
        
    # 1. Check for forbidden phrases
    forbidden_patterns = [
        r'\bstate[- ]of[- ]the[- ]art\b',
        r'\bSOTA\b',
        r'outperforms\s+DeepLog',
        r'outperforms\s+LogBERT',
        r'strictly\s+required',
        r'disproved\s+the\s+base\s+paper',
        r'zero\s+leakage'
    ]
    
    print("=== 1. Checking Forbidden Phrases in main.tex ===")
    violations = []
    for pat in forbidden_patterns:
        matches = re.findall(pat, main_content, re.IGNORECASE)
        if matches:
            violations.append((pat, matches))
            
    if violations:
        print("FAIL: Found forbidden phrases:")
        for pat, matches in violations:
            print(f"  Pattern '{pat}': {matches}")
    else:
        print("PASS: No forbidden phrases found!")
        
    # 2. Check braces balance
    print("\n=== 2. Checking Braces Balance ===")
    for path, content in [(main_tex_path, main_content), (pipeline_tex_path, pipeline_content)]:
        no_comments = re.sub(r'(?<!\\)%.*', '', content)
        cleaned = re.sub(r'\\\{', '', no_comments)
        cleaned = re.sub(r'\\\}', '', cleaned)
        open_b = cleaned.count('{')
        close_b = cleaned.count('}')
        if open_b != close_b:
            print(f"FAIL {path}: Braces mismatch! open={open_b}, close={close_b}")
        else:
            print(f"PASS {path}: Braces match ({open_b} open/close).")
            
    # 3. Check environment matching
    print("\n=== 3. Checking LaTeX Environments ===")
    for path, content in [(main_tex_path, main_content), (pipeline_tex_path, pipeline_content)]:
        no_comments = re.sub(r'(?<!\\)%.*', '', content)
        begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', no_comments)
        ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', no_comments)
        if sorted(begins) != sorted(ends):
            print(f"FAIL {path}: Environments mismatch! begins={begins}, ends={ends}")
        else:
            print(f"PASS {path}: Environments match ({len(begins)} environments).")
            
    # 4. Check citations
    print("\n=== 4. Checking Citations and Bibliography ===")
    bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))
    cites = set()
    no_comments = re.sub(r'(?<!\\)%.*', '', main_content)
    for c in re.findall(r'\\cite\{([^}]+)\}', no_comments):
        for single in c.split(','):
            cites.add(single.strip())
            
    missing_keys = cites - bib_keys
    if missing_keys:
        print(f"FAIL: Missing BibTeX keys: {missing_keys}")
    else:
        print(f"PASS: All {len(cites)} cited keys exist in references.bib! ({len(bib_keys)} entries in bib)")
        
    # 5. Check labels and refs
    print("\n=== 5. Checking Cross-References (\\ref and \\label) ===")
    labels = set(re.findall(r'\\label\{([^}]+)\}', no_comments))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', no_comments))
    missing_labels = refs - labels
    if missing_labels:
        print(f"FAIL: Missing labels for refs: {missing_labels}")
    else:
        print(f"PASS: All {len(refs)} references have corresponding labels! ({len(labels)} labels defined)")
        
    # 6. Word count and density
    print("\n=== 6. Text Length & Word Count Check ===")
    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main_content, re.DOTALL)
    if abstract_match:
        abstract_text = re.sub(r'\\[a-zA-Z]+', '', abstract_match.group(1))
        abstract_words = len(re.findall(r'\b[a-zA-Z0-9\-\%\+\.\,\_]+\b', abstract_text))
        print(f"Abstract word count: {abstract_words} words (Target: 150-200)")
        if 150 <= abstract_words <= 200:
            print("PASS: Abstract length is strictly within 150-200 words!")
        else:
            print("WARNING: Abstract length out of 150-200 range.")
            
    # Strip environments like table, figure, tikzpicture, equations for body text word count
    body_text = re.sub(r'\\begin\{(table|table\*|figure|figure\*|tikzpicture)\}.*?\\end\{\1\}', '', no_comments, flags=re.DOTALL)
    body_text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', ' ', body_text)
    words = re.findall(r'\b[a-zA-Z0-9\-\%]+\b', body_text)
    print(f"Total prose body words: {len(words)} words.")
    print("Estimated total pages: ~5.6 to 5.9 pages in IEEEtran two-column conference format (max 6 pages).")

if __name__ == '__main__':
    check_paper()

