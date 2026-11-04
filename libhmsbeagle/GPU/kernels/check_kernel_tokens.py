#!/usr/bin/env python3
# Token-identity check of the GPU kernel texts between two source trees (fused-handoff PLAN section 4, D15).
#
#   python3 -I libhmsbeagle/GPU/kernels/check_kernel_tokens.py BASE HEAD
#
# BASE and HEAD are repository roots (e.g. a worktree at 88615ea4 and the fused_kernels clone). For every
# configuration below it preprocesses the kernel text of both trees with clang and compares the token streams:
#   OpenCL: each string of make_opencl_kernels.sh (standard) and make_opencl_spectral_kernels.sh (spectral),
#           x {SP, DP} x 9 state counts, under clang -E -P -x c with the build defines of GPUInterfaceOpenCL.cpp
#           for each device set, plus the TinyGPU AMD HIP defines (tinygpu_amd_compile.cpp);
#   CUDA:   kernels4.cu (4 states) or kernelsX.cu (others) as make_tinygpu_kernels.sh hands them to nvcc,
#           under clang -E -P -x c++ -I<root>, with {CUDA}, {CUDA, FW_TINYGPU}, {CUDA, FW_TINYGPU, CUDA_SPECTRAL}.
# Every differing token run is printed with context. Exit status 0 only if all configurations are identical.

import concurrent.futures, difflib, os, re, subprocess, sys

STATE_COUNTS = [4, 16, 32, 48, 64, 80, 128, 192, 256]
OPENCL_BASE = ['FW_OPENCL', 'OPENCL_KERNEL_BUILD', 'DLS_MACOS']
OPENCL_DEVICES = [[], ['FW_OPENCL_AMDGPU'], ['FW_OPENCL_AMDGPU', 'FW_OPENCL_APPLEAMDGPU'],
                  ['FW_OPENCL_INTELGPU', 'FW_OPENCL_APPLEINTELGPU'], ['FW_OPENCL_CPU'],
                  ['FW_OPENCL_CPU', 'FW_OPENCL_APPLECPU']]
HIP = ['FW_TINYGPU_AMD', 'FW_OPENCL', 'OPENCL_KERNEL_BUILD']
CUDA_SETS = [['CUDA'], ['CUDA', 'FW_TINYGPU'], ['CUDA', 'FW_TINYGPU', 'CUDA_SPECTRAL']]
TOKEN = re.compile(r'[A-Za-z_]\w*|\d[\w.]*(?:[eE][+-]\d+)?|\S')
CONTEXT = 12


def script_lines(path):
    # cat FILE | sed 's/$/\\n\\/': every line of the file, each ending in a newline
    text = open(path).read()
    lines = text.split('\n')
    if text.endswith('\n'):
        lines.pop()
    return ''.join(line + '\n' for line in lines)


def opencl_text(root, header, states, dp):
    gpu = os.path.join(root, 'libhmsbeagle', 'GPU')
    kernels = os.path.join(gpu, 'kernels')
    files = [os.path.join(gpu, 'GPUImplDefs.h'), os.path.join(kernels, 'kernelsAll.cu')]
    if states == 4:
        files += [os.path.join(kernels, 'kernels4.cu'), os.path.join(kernels, 'kernels4Derivatives.cu')]
        if header == 'spectral':
            files.append(os.path.join(kernels, 'kernelsSpectralIfDef4.cu'))
    else:
        files += [os.path.join(kernels, 'kernelsX.cu'), os.path.join(kernels, 'kernelsXDerivatives.cu')]
        if header == 'spectral':
            files.append(os.path.join(kernels, 'kernelsSpectralIfDef.cu'))
    text = '#define STATE_COUNT %d\n' % states
    if dp:
        text += '#define DOUBLE_PRECISION\n'
    return text + ''.join(script_lines(f) for f in files)


def preprocess(args, stdin=None):
    r = subprocess.run(['clang', '-E', '-P'] + args, input=stdin, capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr.strip()
    return TOKEN.findall(r.stdout), None


def opencl_job(root, header, states, dp, defs):
    return preprocess(['-x', 'c'] + ['-D' + d for d in defs] + ['-'], opencl_text(root, header, states, dp))


def cuda_job(root, states, dp, defs):
    source = os.path.join(root, 'libhmsbeagle', 'GPU', 'kernels', 'kernels4.cu' if states == 4 else 'kernelsX.cu')
    defs = defs + ['STATE_COUNT=%d' % states] + (['DOUBLE_PRECISION'] if dp else []) + ['HAVE_CONFIG_H']
    return preprocess(['-x', 'c++'] + ['-D' + d for d in defs] + ['-I' + root, source])


def configurations():
    for header in ['standard', 'spectral']:
        for dp in [0, 1]:
            for states in STATE_COUNTS:
                for devices in OPENCL_DEVICES + [None]:
                    defs = HIP if devices is None else OPENCL_BASE + devices
                    name = 'opencl %s %s %d %s' % (header, 'DP' if dp else 'SP', states,
                                                   'HIP' if devices is None else '{' + ','.join(devices) + '}')
                    yield name, opencl_job, (header, states, dp, defs)
    for dp in [0, 1]:
        for states in STATE_COUNTS:
            for defs in CUDA_SETS:
                name = 'cuda %s %d {%s}' % ('DP' if dp else 'SP', states, ','.join(defs))
                yield name, cuda_job, (states, dp, defs)


def differences(a, b):
    # trim the common prefix and suffix, then align the middle; yields (base_lo, base_hi, head_lo, head_hi)
    lo = 0
    while lo < len(a) and lo < len(b) and a[lo] == b[lo]:
        lo += 1
    hi = 0
    while hi < len(a) - lo and hi < len(b) - lo and a[-1 - hi] == b[-1 - hi]:
        hi += 1
    matcher = difflib.SequenceMatcher(None, a[lo:len(a) - hi], b[lo:len(b) - hi])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            yield lo + i1, lo + i2, lo + j1, lo + j2


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python3 -I %s BASE_ROOT HEAD_ROOT' % sys.argv[0])
    base, head = (os.path.abspath(p) for p in sys.argv[1:])
    configs = list(configurations())
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = [(name, pool.submit(job, base, *args), pool.submit(job, head, *args)) for name, job, args in configs]
        bad_configs = errors = diff_tokens = base_tokens = 0
        for name, fb, fh in futures:
            (tb, eb), (th, eh) = fb.result(), fh.result()
            if tb is None or th is None:
                errors += 1
                print('PREPROCESSOR ERROR %s\n  base: %s\n  head: %s' % (name, eb, eh))
                continue
            base_tokens += len(tb)
            if tb == th:
                continue
            bad_configs += 1
            print('DIFF %s (base %d tokens, head %d tokens)' % (name, len(tb), len(th)))
            for i1, i2, j1, j2 in differences(tb, th):
                diff_tokens += max(i2 - i1, j2 - j1)
                print('  at base token %d, head token %d:' % (i1, j1))
                print('    before: %s' % ' '.join(tb[max(0, i1 - CONTEXT):i1]))
                print('    base:   %s' % ' '.join(tb[i1:i2]))
                print('    head:   %s' % ' '.join(th[j1:j2]))
                print('    after:  %s' % ' '.join(tb[i2:i2 + CONTEXT]))
    print('%d configurations, %d differing, %d preprocessor errors, %d differing tokens, %d base tokens compared'
          % (len(configs), bad_configs, errors, diff_tokens, base_tokens))
    sys.exit(1 if bad_configs or errors else 0)


if __name__ == '__main__':
    main()
