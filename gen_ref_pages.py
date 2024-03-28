'''Generate the code reference pages.'''

from pathlib import Path

import mkdocs_gen_files

src = 'VAE'
hide = {'_', '.'}

for path in sorted(Path(src).rglob('*.py')):
    if not any(part[0] in hide for part in path.parts):
        module_path = path.relative_to(src).with_suffix('')
        doc_path = path.relative_to(src).with_suffix('.md')
        full_doc_path = Path(src, doc_path)

        parts = list(module_path.parts)

        if parts[-1] == '__init__':
            parts = parts[:-1]
        elif parts[-1] == '__main__':
            continue

        parts = [src] + parts
        identifier = '.'.join(parts)
        full_doc_path = '.'.join(full_doc_path.parts)

        # virtually create the doc file
        with mkdocs_gen_files.open(full_doc_path, 'w') as f:
            print('::: ' + identifier, file=f)
