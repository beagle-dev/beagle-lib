"""Device-free inspection of ptxas cubins, the facts the C++ program loader and the planned AOT cubin selection rely
on (TODO.md plan step C1): ELF header (e_type, e_flags = SM, OS ABI), size and zlib size, tinygrad elf_loader image
size, .text.* count, relocation types, addends and targets, and the EIATTR_REGCOUNT (0x2f) placement in .nv.info.
    python cubin_inspect.py [cubin ...]        # default: every cubin in $BEAGLE_TINYGPU_DATA/cubins
    python cubin_inspect.py --relocs cubin     # also list relocations with their symbols (raw ELF parse)"""
import os, sys, struct, zlib, collections, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.support.elf import elf_loader

def summary(path):
    b = open(path, "rb").read()
    e_type, = struct.unpack_from("<H", b, 0x10)
    e_flags, = struct.unpack_from("<I", b, 0x30)
    image, sections, relocs = elf_loader(b, force_section_align=128)
    names = [s.name for s in sections]
    texts = [n for n in names if n.startswith(".text.")]
    types = collections.Counter(r[2] for r in relocs)
    addends = collections.Counter(r[3] for r in relocs)
    def target(r):
        name = next((s.name for s in sections if s.header.sh_addr <= r[1] < s.header.sh_addr + max(s.header.sh_size, 1)
                     and s.header.sh_type in (1, 8)), "?")
        return ".text.<kernel>" if name.startswith(".text.") else name
    targets = collections.Counter(target(r) for r in relocs)
    bare, regcounts = collections.Counter(), set()
    for sh in sections:
        if sh.name != ".nv.info": continue
        c, off = bytes(sh.content), 0
        while off + 4 <= len(c):
            typ, param, sz = struct.unpack_from("BBH", c, off)
            bare[(hex(typ), hex(param))] += 1
            if param == 0x2f and typ == 4: regcounts.add(struct.unpack_from("II", c, off + 4)[1])
            off += (sz if typ == 4 else 0) + 4
    print(f"{os.path.basename(path)}: {len(b)} B (zlib-9 {len(zlib.compress(b, 9))} B), image {image.nbytes:#x}, "
          f"e_type={e_type} e_flags={e_flags:#x} osabi={b[7]} abiver={b[8]}")
    print(f"   .text.*={len(texts)}  reloc types={dict(types)}  addends={dict(addends)}  targets={dict(targets)}")
    print(f"   bare .nv.info REGCOUNT entries={bare[('0x4', '0x2f')]} values={sorted(regcounts)[:12]}")

def relocations(path):
    b = open(path, "rb").read()
    shoff, = struct.unpack_from("<Q", b, 0x28); shnum, shstrndx = struct.unpack_from("<HH", b, 0x3c)
    S = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * 64) for i in range(shnum)]
    def nm(tab, off): e = b.find(b"\0", tab + off); return b[tab + off:e].decode()
    names = [nm(S[shstrndx][4], s[0]) for s in S]
    symtab = next(i for i, s in enumerate(S) if s[1] == 2); strtab = S[S[symtab][6]][4]
    def sym(i):
        st_name, info, other, shndx, val, size = struct.unpack_from("<IBBHQQ", b, S[symtab][4] + i * 24)
        return nm(strtab, st_name) or f"[sec {names[shndx]}]", shndx, val
    kinds = collections.Counter()
    for i, s in enumerate(S):
        if s[1] not in (4, 9): continue
        ent = 24 if s[1] == 4 else 16
        for k in range(s[5] // ent):
            o = s[4] + k * ent
            r_off, r_info = struct.unpack_from("<QQ", b, o)
            add = struct.unpack_from("<q", b, o + 16)[0] if s[1] == 4 else 0
            sn, shndx, val = sym(r_info >> 32)
            kinds[(names[i], r_info & 0xffffffff, "kernel*" if sn.startswith("kernel") else sn, add != 0)] += 1
    print("   relocations (section, type, symbol, nonzero addend): count")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]): print(f"     {k}: {v}")

args = sys.argv[1:]
show_relocs = "--relocs" in args
paths = [a for a in args if a != "--relocs"] or sorted(glob.glob(str(tgpaths.DATA / "cubins" / "*.cubin")))
for p in paths:
    summary(p)
    if show_relocs: relocations(p)
