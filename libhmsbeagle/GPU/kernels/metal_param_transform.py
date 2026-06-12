#!/usr/bin/env python3
"""
Transform CUDA/OpenCL kernel source to Metal-compatible kernel signatures.

Metal kernel functions cannot accept plain 'int' or 'float' scalar parameters.
They must use 'constant int&' and 'constant float&' respectively.
This script transforms scalar parameters only inside kernel function signatures,
leaving function bodies and non-kernel functions untouched.
"""

import sys
import re

def transform_params_in_signature(sig):
    """
    Given the text of a kernel parameter list (everything between the outer parens),
    replace bare scalar 'int name' and 'float name' with 'constant int& name'
    and 'constant float& name'.  Pointer params (int*, float*) are left alone.
    uint/uint3/etc. from KW_THREAD_ARGS are also left alone.
    """
    # Match a bare 'int' or 'float' type not followed by '*' (pointer stays as-is)
    # and not already preceded by 'constant' or another qualifier.
    # We use a negative lookbehind for 'constant ' and a negative lookahead for '*'.
    sig = re.sub(
        r'(?<![a-zA-Z0-9_])int(?!\s*[*&])(\s+)(?=[a-zA-Z_])',
        r'constant int&\1',
        sig
    )
    sig = re.sub(
        r'(?<![a-zA-Z0-9_])float(?!\s*[*&])(\s+)(?=[a-zA-Z_])',
        r'constant float&\1',
        sig
    )
    return sig

def transform(text):
    """
    Walk through the source text.  When we enter a kernel function signature
    (marked by 'KW_GLOBAL_KERNEL'), collect everything up to the matching '{'
    and transform scalar params in the parameter list portion.
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        # Look for the start of a kernel declaration.
        m = re.search(r'\bKW_GLOBAL_KERNEL\b', text[i:])
        if m is None:
            result.append(text[i:])
            break

        start = i + m.start()
        result.append(text[i:start])  # everything before this kernel

        # Find the opening '(' of the parameter list.
        paren_open = text.find('(', start)
        if paren_open == -1:
            result.append(text[start:])
            break

        # Find matching closing ')'.
        depth = 0
        j = paren_open
        while j < n:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        paren_close = j  # points at ')'

        # Emit up to and including '('
        result.append(text[start:paren_open + 1])

        # Transform scalar params between '(' and ')'
        params = text[paren_open + 1 : paren_close]
        result.append(transform_params_in_signature(params))

        # Emit ')'
        result.append(')')
        i = paren_close + 1

    return ''.join(result)

if __name__ == '__main__':
    src = sys.stdin.read()
    sys.stdout.write(transform(src))
