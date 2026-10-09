/*
 * tinygpu_amd_compile.cpp
 *
 * The build-time AMD kernel compiler (TODO.md plan step A1j): tinygrad's compile_hip
 * (tinygrad/runtime/support/compiler_amd.py:36-83 at a9830e2b4) in C++, calling the comgr library it calls (dlopen'd, comgr
 * 3: its enum values, as tinygrad's autogen comgr_3.py), on the source the AMD daemon compiles at run time: the three
 * defines amd_compile_helper.compile_hip prepends, then KERNELS_STRING_<precision>_<padded state count> from
 * BeagleOpenCL_kernels.h. The same actions, options and option splitting, so the HSACOs are tinygrad's byte for byte
 * (golden_amd_hsaco.py); the plugin embeds them (kernels/BeagleTinyGPU_hsaco.S) and the daemon no longer compiles.
 * A variant fails, and so the build, if any of its kernels spills registers to scratch (the code object's metadata).
 *   tinygpu_amd_compile <libamd_comgr path> <arch> <out dir> <variant (SP_4 ... DP_256)>...
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"

namespace {

struct Handle { uint64_t handle; };   // amd_comgr_data_t, amd_comgr_data_set_t, amd_comgr_action_info_t, amd_comgr_metadata_node_t
typedef int Status;                   // amd_comgr_status_t: 0 is success
enum : uint32_t {                     // comgr 3 (tinygrad/runtime/autogen/comgr_3.py)
    LANGUAGE_HIP = 3, KIND_SOURCE = 1, KIND_LOG = 5, KIND_EXECUTABLE = 8,
    ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 12, ACTION_CODEGEN_BC_TO_RELOCATABLE = 4, ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7,
};

struct Comgr {
    void (*get_version)(size_t*, size_t*);
    Status (*status_string)(Status, const char**);
    Status (*create_action_info)(Handle*);
    Status (*action_info_set_language)(Handle, uint32_t);
    Status (*action_info_set_isa_name)(Handle, const char*);
    Status (*action_info_set_logging)(Handle, bool);
    Status (*action_info_set_option_list)(Handle, const char**, size_t);
    Status (*create_data_set)(Handle*);
    Status (*create_data)(uint32_t, Handle*);
    Status (*set_data)(Handle, size_t, const char*);
    Status (*set_data_name)(Handle, const char*);
    Status (*data_set_add)(Handle, Handle);
    Status (*do_action)(uint32_t, Handle, Handle, Handle);
    Status (*action_data_get_data)(Handle, uint32_t, size_t, Handle*);
    Status (*get_data)(Handle, size_t*, char*);
    Status (*release_data)(Handle);
    Status (*destroy_data_set)(Handle);
    Status (*destroy_action_info)(Handle);
    Status (*get_data_metadata)(Handle, Handle*);
    Status (*metadata_lookup)(Handle, const char*, Handle*);
    Status (*get_metadata_string)(Handle, size_t*, char*);
    Status (*get_metadata_list_size)(Handle, size_t*);
    Status (*index_list_metadata)(Handle, size_t, Handle*);
    Status (*destroy_metadata)(Handle);
};

Comgr g;
bool load(const char* path) {
    void* lib = dlopen(path, RTLD_NOW);
    if (!lib) { fprintf(stderr, "tinygpu_amd_compile: %s\n", dlerror()); return false; }
    bool ok = true;
    auto sym = [&](auto& f, const char* name) { *(void**)&f = dlsym(lib, name); if (!f) { fprintf(stderr, "tinygpu_amd_compile: no %s\n", name); ok = false; } };
    sym(g.get_version, "amd_comgr_get_version"); sym(g.status_string, "amd_comgr_status_string");
    sym(g.create_action_info, "amd_comgr_create_action_info"); sym(g.action_info_set_language, "amd_comgr_action_info_set_language");
    sym(g.action_info_set_isa_name, "amd_comgr_action_info_set_isa_name"); sym(g.action_info_set_logging, "amd_comgr_action_info_set_logging");
    sym(g.action_info_set_option_list, "amd_comgr_action_info_set_option_list"); sym(g.create_data_set, "amd_comgr_create_data_set");
    sym(g.create_data, "amd_comgr_create_data"); sym(g.set_data, "amd_comgr_set_data"); sym(g.set_data_name, "amd_comgr_set_data_name");
    sym(g.data_set_add, "amd_comgr_data_set_add"); sym(g.do_action, "amd_comgr_do_action");
    sym(g.action_data_get_data, "amd_comgr_action_data_get_data"); sym(g.get_data, "amd_comgr_get_data");
    sym(g.release_data, "amd_comgr_release_data"); sym(g.destroy_data_set, "amd_comgr_destroy_data_set");
    sym(g.destroy_action_info, "amd_comgr_destroy_action_info");
    sym(g.get_data_metadata, "amd_comgr_get_data_metadata"); sym(g.metadata_lookup, "amd_comgr_metadata_lookup");
    sym(g.get_metadata_string, "amd_comgr_get_metadata_string"); sym(g.get_metadata_list_size, "amd_comgr_get_metadata_list_size");
    sym(g.index_list_metadata, "amd_comgr_index_list_metadata"); sym(g.destroy_metadata, "amd_comgr_destroy_metadata");
    return ok;
}

struct Fail { std::string what; };
void check(Status s) {   // compiler_amd.check
    if (s == 0) return;
    const char* str = "";
    g.status_string(s, &str);
    throw Fail{"comgr fail " + std::to_string(s) + ", " + str};
}
std::string get_data(Handle set, uint32_t kind) {   // _get_comgr_data
    Handle d;
    size_t n = 0;
    check(g.action_data_get_data(set, kind, 0, &d));
    check(g.get_data(d, &n, nullptr));
    std::string out(n, '\0');
    check(g.get_data(d, &n, &out[0]));
    check(g.release_data(d));
    return out;
}
Status set_options(Handle info, const std::string& options) {   // set_options: split on ' ', as bytes.split(b' ')
    std::vector<std::string> parts;
    size_t at = 0;
    while (true) {
        size_t sp = options.find(' ', at);
        parts.push_back(options.substr(at, sp == std::string::npos ? std::string::npos : sp - at));
        if (sp == std::string::npos) break;
        at = sp + 1;
    }
    std::vector<const char*> argv;
    for (const std::string& p : parts) argv.push_back(p.c_str());
    return g.action_info_set_option_list(info, argv.data(), argv.size());
}

std::string compile_hip(const std::string& prg, const std::string& arch) {
    Handle info, src_set, bc_set, reloc_set, exec_set, src;
    check(g.create_action_info(&info));
    check(g.action_info_set_language(info, LANGUAGE_HIP));
    check(g.action_info_set_isa_name(info, ("amdgcn-amd-amdhsa--" + arch).c_str()));
    check(g.action_info_set_logging(info, true));
    for (Handle* s : {&src_set, &bc_set, &reloc_set, &exec_set}) check(g.create_data_set(s));
    check(g.create_data(KIND_SOURCE, &src));
    check(g.set_data(src, prg.size(), prg.data()));
    check(g.set_data_name(src, "<null>"));
    check(g.data_set_add(src_set, src));
    const char* options[] = {"-O3", "-mcumode", "--hip-version=6.0.32830", "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=0",
                             "-DHIP_VERSION_PATCH=32830", "-D__HIPCC_RTC__", "-std=c++14", "-nogpuinc", "-Wno-gnu-line-marker",
                             "-Wno-missing-prototypes", nullptr, "-I/opt/rocm/include", "-Xclang -disable-llvm-passes", "-Xclang -aux-triple",
                             "-Xclang x86_64-unknown-linux-gnu"};
    std::string joined;
    for (const char* o : options) {
        if (!joined.empty()) joined += ' ';
        joined += o ? std::string(o) : "--offload-arch=" + arch;
    }
    check(set_options(info, joined));
    if (g.do_action(ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, info, src_set, bc_set) != 0) {
        fprintf(stderr, "%s\n", get_data(bc_set, KIND_LOG).c_str());
        throw Fail{"compile failed"};
    }
    check(set_options(info, "-O3 -mllvm -amdgpu-internalize-symbols"));
    check(g.do_action(ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set));
    check(set_options(info, ""));
    check(g.do_action(ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set, exec_set));
    std::string out = get_data(exec_set, KIND_EXECUTABLE);
    check(g.release_data(src));
    for (Handle s : {src_set, bc_set, reloc_set, exec_set}) check(g.destroy_data_set(s));
    check(g.destroy_action_info(info));
    return out;
}

// The kernels of an HSACO that spill registers to scratch, from its metadata's .vgpr_spill_count and .sgpr_spill_count:
// " name (v VGPRs, s SGPRs)" each, "" if none. Spilled registers live in scratch memory, in VRAM: GPUImplDefs.h's
// KW_NO_UNROLL says how the 64-state kernels came to spill.
std::string spills(const std::string& hsaco) {
    Handle data, meta, kernels;
    check(g.create_data(KIND_EXECUTABLE, &data));
    check(g.set_data(data, hsaco.size(), hsaco.data()));
    check(g.get_data_metadata(data, &meta));
    check(g.metadata_lookup(meta, "amdhsa.kernels", &kernels));
    size_t n = 0;
    check(g.get_metadata_list_size(kernels, &n));
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        Handle k;
        check(g.index_list_metadata(kernels, i, &k));
        auto field = [&](const char* key) {
            Handle v;
            if (g.metadata_lookup(k, key, &v) != 0) throw Fail{std::string("no ") + key + " in a kernel's metadata"};
            size_t len = 0;   // with the terminating NUL
            check(g.get_metadata_string(v, &len, nullptr));
            std::string s(len, '\0');
            check(g.get_metadata_string(v, &len, &s[0]));
            check(g.destroy_metadata(v));
            return std::string(s.c_str());
        };
        const std::string vgpr = field(".vgpr_spill_count"), sgpr = field(".sgpr_spill_count");
        if (vgpr != "0" || sgpr != "0") out += " " + field(".name") + " (" + vgpr + " VGPRs, " + sgpr + " SGPRs)";
        check(g.destroy_metadata(k));
    }
    check(g.destroy_metadata(kernels));
    check(g.destroy_metadata(meta));
    check(g.release_data(data));
    return out;
}

const char* variant_source(const std::string& v) {   // GPUInterfaceTinyGPUAMD.cpp amd_opencl_kernel_source
#define TG_V(P, N) if (v == #P "_" #N) return KERNELS_STRING_##P##_##N;
    TG_V(SP, 4) TG_V(SP, 16) TG_V(SP, 32) TG_V(SP, 48) TG_V(SP, 64) TG_V(SP, 80) TG_V(SP, 128) TG_V(SP, 192) TG_V(SP, 256)
    TG_V(DP, 4) TG_V(DP, 16) TG_V(DP, 32) TG_V(DP, 48) TG_V(DP, 64) TG_V(DP, 80) TG_V(DP, 128) TG_V(DP, 192) TG_V(DP, 256)
#undef TG_V
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <libamd_comgr> <arch> <out dir> <variant>...\n", argv[0]); return 2; }
    if (!load(argv[1])) return 1;
    size_t major = 0, minor = 0;
    g.get_version(&major, &minor);
    if (major < 3) { fprintf(stderr, "tinygpu_amd_compile: comgr %zu.%zu; these enum values are comgr 3's\n", major, minor); return 1; }
    const std::string arch = argv[2], dir = argv[3];
    for (int i = 4; i < argc; ++i) {
        const char* src = variant_source(argv[i]);
        if (!src) { fprintf(stderr, "tinygpu_amd_compile: no variant %s\n", argv[i]); return 1; }
        try {
            const std::string hsaco = compile_hip(std::string("#define FW_TINYGPU_AMD 1\n#define FW_OPENCL 1\n#define OPENCL_KERNEL_BUILD 1\n") + src, arch);
            const std::string spilled = spills(hsaco);
            if (!spilled.empty()) { fprintf(stderr, "tinygpu_amd_compile: %s: kernels spill registers to scratch:%s\n", argv[i], spilled.c_str()); return 1; }
            const std::string path = dir + "/" + argv[i] + "_" + arch + ".hsaco";
            FILE* f = fopen(path.c_str(), "wb");
            if (!f || fwrite(hsaco.data(), 1, hsaco.size(), f) != hsaco.size() || fclose(f) != 0) { fprintf(stderr, "tinygpu_amd_compile: cannot write %s\n", path.c_str()); return 1; }
            printf("%s: %zu bytes (comgr %zu.%zu)\n", path.c_str(), hsaco.size(), major, minor);
        } catch (const Fail& e) {
            fprintf(stderr, "tinygpu_amd_compile: %s: %s\n", argv[i], e.what.c_str());
            return 1;
        }
    }
    return 0;
}
