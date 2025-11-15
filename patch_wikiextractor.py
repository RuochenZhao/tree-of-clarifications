#!/usr/bin/env python3
"""Patch wikiextractor to work with Python 3.11+"""

import wikiextractor
import os
import re

# Find wikiextractor installation
extract_py = os.path.join(os.path.dirname(wikiextractor.__file__), 'extract.py')

print(f"Patching {extract_py}...")

# Read the file
with open(extract_py, 'r') as f:
    content = f.read()

# Fix first regex - move (?i) to the start
old_pattern_1 = r"'\[(((?i)' + '|'.join(wgUrlProtocols) + ')' + EXT_LINK_URL_CLASS + r'+)\s*([^\]\x00-\x08\x0a-\x1F]*?)\]'"
new_pattern_1 = r"'(?i)\[((' + '|'.join(wgUrlProtocols) + ')' + EXT_LINK_URL_CLASS + r'+)\s*([^\]\x00-\x08\x0a-\x1F]*?)\]'"

# Fix second regex - use regex substitution to handle any whitespace
# This will match the pattern even if it spans multiple lines
old_pattern_2_regex = re.compile(
    r'r"""(\^|\(\?i\)\^)(http://\|https://)\(\[\^\]\[<>"\\\x00-\\\x20\\\x7F\\\s\]\+\)\s*/\(\[A-Za-z0-9_\.,~%\\\-\+&;#\*\?\!=\(\)@\\\x80-\\\xFF\]\+\)\\\.\(\(\?i\)gif\|png\|jpg\|jpeg\)\$"""',
    re.MULTILINE | re.DOTALL
)

# Apply first replacement
if old_pattern_1 in content:
    content = content.replace(old_pattern_1, new_pattern_1)
    print("✅ Fixed first regex pattern")
else:
    print("ℹ️  First regex pattern already fixed or not found")

# Apply second replacement - simpler approach, just replace the specific problematic part
if '.((?i)gif|png|jpg|jpeg)$' in content:
    content = content.replace('.((?i)gif|png|jpg|jpeg)$', '.(gif|png|jpg|jpeg)$')
    # Now add (?i) at the start of the pattern
    content = content.replace(
        'r"""^(http://|https://)([^][<>"\x00-\x20\x7F\s]+)',
        'r"""(?i)^(http://|https://)([^][<>"\x00-\x20\x7F\s]+)'
    )
    print("✅ Fixed second regex pattern")
else:
    print("ℹ️  Second regex pattern already fixed or not found")

# Write back
with open(extract_py, 'w') as f:
    f.write(content)

print("✅ Successfully patched wikiextractor for Python 3.11+ compatibility!")
