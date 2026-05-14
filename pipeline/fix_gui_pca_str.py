from pathlib import Path

p = Path('GUI.py')
text = p.read_text(encoding='utf-8')
old = 'f"        determined by a retained-variance threshold of {\n                            pca_comp}\n"'
new = 'f"        determined by a retained-variance threshold of {pca_comp}\n"'
if old not in text:
    print('pattern not found')
else:
    text = text.replace(old, new)
    p.write_text(text, encoding='utf-8')
    print('patched')
