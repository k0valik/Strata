// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

<<<<<<< this fork
constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "STRATA-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);

#ifndef _WIN32
// bytes in free blocks of 2 MiB or more (/proc/buddyinfo, order >= 9): what MADV_HUGEPAGE can get without compaction
uint64_t free_huge_bytes() {
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            const char* q = z + 4;
            while (*q == ' ') ++q;
            while (*q != '\0' && *q != ' ') ++q;   // the zone's name
            unsigned long long c[11] = {0};
            int n = 0;
            for (; n < 11; ++n) {
                char* e = nullptr;
                const unsigned long long v = std::strtoull(q, &e, 10);
                if (e == q) break;
                c[n] = v;
                q = e;
            }
            for (int o = 9; o < n; ++o) total += c[o] * (4096ull << o);
        }
        std::fclose(f);
    }
    return total;
}
#endif

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
// A non-empty shared_file instead maps one file whose first 4 KiB identify the pack and whose remaining bytes
// are the resident arena.  This shared-file layout is intended for tmpfs (/dev/shm); hugetlbfs would need
// hugepage-aligned file size and arena offset rather than the 4 KiB header layout used here.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note, const std::string& shared_file,
              uint64_t shared_pack_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;

=======
// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#ifdef _WIN32
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    if (large > 0) {
        void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        note = "large pages refused (GetLargePageMinimum=" + std::to_string((unsigned long long) large) +
               ", VirtualAlloc error " + std::to_string((unsigned long long) GetLastError()) +
               " - needs SeLockMemoryPrivilege); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    return p;
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
    if (p != MAP_FAILED) {
        got = PageBacking::LargePages;
        note = "hugetlb 2 MB pages";
        return p;
    }
    note = "MAP_HUGETLB unavailable (no hugetlb pool configured?); using 4 KB pages";
    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
<<<<<<< this fork
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    // #771: with THP "always" the kernel already backs this mapping with huge pages where it can; the extra
    // MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
    // machine stretched a 25 s start to 432 s.  So it is not asked for there.  STRATA_NO_ARENA_THP=1 skips it
    // anywhere (the request alone; STRATA_NO_LARGEPAGES also skips the hugetlb try).
    std::string thp_mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) thp_mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    const bool skip_thp_request = std::getenv("STRATA_NO_ARENA_THP") != nullptr || thp_mode == "always";
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && skip_thp_request) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += std::getenv("STRATA_NO_ARENA_THP") != nullptr
                    ? "; transparent huge pages skipped (STRATA_NO_ARENA_THP)"
                    : "; transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && std::getenv("STRATA_ARENA_THP") == nullptr &&
               free_huge_bytes() < bytes / 2) {
        // #2126: THP "madvise" + defrag "madvise" makes every fault of a MADV_HUGEPAGE mapping reclaim and compact
        // until it finds a 2 MiB block.  When less than half of the arena could be backed by blocks that are free
        // now (a 62 GB box holding 47 GB of anonymous memory: 148K failed compactions, 100% system time, the load
        // never ended), that is minutes of kernel work per GiB for a TLB gain.  Said, and STRATA_ARENA_THP=1 asks for it anyway.
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages NOT requested: only " + std::to_string(free_huge_bytes() >> 30) +
                " GiB of free 2 MiB blocks for a " + std::to_string(bytes >> 30) +
                " GiB arena, compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway) (#2126)";
        std::fprintf(stderr, "strata: expert arena: %s\n", note.c_str());   // said now: the load that follows can take minutes
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
=======
    return p == MAP_FAILED ? nullptr : p;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds) : capacity(bytes) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note);

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        const cudaError_t e = cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && clear_error()) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (cudaHostRegister((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            note = "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); " +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (env == nullptr || std::string(env) != "0") {
                    const strata::platform::LockResult lr =
                        strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            if (env == nullptr || std::string(env) != "0") {
                const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}

<<<<<<< this fork
PinnedArena::PinnedArena(uint64_t bytes, ByNeed, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                         uint64_t max_pinned_bytes, const std::string& shared_file,
                         uint64_t shared_pack_hash) : capacity(bytes) {
    by_need = true;
    if (bytes == 0) return;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    if (base == nullptr) return;
    // Register BEFORE any page is touched (see the first constructor).  Same caps as there: the engine's
    // `max_pinned_bytes`, else STRATA_ARENA_PIN_GIB=N (> 0); "auto" (-2) is the sliced pin's and does not apply.
    const int env_gib = arena_pin_cap_gib();
    uint64_t cap = max_pinned_bytes;
    if (cap == 0 && env_gib > 0) cap = (uint64_t) env_gib << 30;
#ifdef _WIN32
    // WDDM: the pin must leave margin (the budget is ~RAM/2 and later cudaMallocs need room).  Unset: 29 GiB;
    // "auto" (-2): the sliced pin's shared-memory-budget limit (29 GiB when it cannot be worked out);
    // STRATA_ARENA_PIN_GIB=0 ("the whole arena"): no cap, up to the driver's refusal.
    if (cap == 0 && env_gib != 0) {
        uint64_t lim = 0;
        std::string why;
        cap = env_gib == -2 && sliced_pin_limit(lim, why) && lim > 0 ? lim : (29ull << 30);
    }
#endif
    constexpr uint64_t kPage = 4096;
    std::map<uint64_t, uint64_t> done;   // registered [start, end), by start: ranges never overlap
    uint64_t total = 0;
    int refused = 0;
    bool capped = false;
    for (const auto& rg : ranges) {
        if (rg.second == 0 || rg.first >= bytes) continue;
        // INWARD: a registered page holds no byte of an unpinned neighbour blob (a copy that STARTS in a registered page
        // and runs past the registration is rejected by CUDA); the boundary pages stay unregistered
        uint64_t a = (rg.first + kPage - 1) & ~(kPage - 1);
        uint64_t b = std::min<uint64_t>(rg.first + rg.second, bytes) & ~(kPage - 1);
        // a page shared with a range registered before (blobs smaller than a page, or odd offsets) is already
        // registered: registering it twice is an error, so trim to what is left; a range inside one is skipped
        auto it = done.upper_bound(a);
        if (it != done.begin()) {
            auto p = std::prev(it);
            if (p->second > a) a = p->second;
        }
        if (it != done.end() && it->first < b) b = it->first;
        if (a >= b) continue;
        if (cap > 0 && total + (b - a) > cap) { capped = true; break; }
        if (cudaHostRegister((uint8_t*) base + a, (size_t) (b - a), cudaHostRegisterPortable | cudaHostRegisterMapped) !=
            cudaSuccess) {
            (void) cudaGetLastError();
            refused = 1;
            break;
        }
        done[a] = b;
        total += b - a;
    }
    for (const auto& d : done) pin_ranges.emplace_back(d.first, d.second - d.first);
    registered_bytes = total;
    registered_slices = (int) pin_ranges.size();
    // the holes stay resident through the working-set lock, as the sliced fallback does for its tail
    std::vector<std::pair<void*, uint64_t>> holes;
    uint64_t cur = 0, hole_bytes = 0;
    for (const auto& r : pin_ranges) {
        if (r.first > cur) { locked_ranges.emplace_back(cur, r.first - cur); hole_bytes += r.first - cur; }
        cur = r.first + r.second;
    }
    if (cur < bytes) { locked_ranges.emplace_back(cur, bytes - cur); hole_bytes += bytes - cur; }
    note = "pin by need: " + std::to_string(pin_ranges.size()) + " ranges registered (" + std::to_string(total >> 20) +
           " MiB)" + (refused ? ", the driver refused the next one" : capped ? ", the cap was reached" : "") +
           "; " + note;
    const char* env = std::getenv("STRATA_ARENA_LOCK");
    if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
        note = "large pages are resident without a lock; " + note;
        locked_ranges.clear();
    } else if (env != nullptr && std::string(env) == "0") {
        note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
        locked_ranges.clear();
    } else {
        for (const auto& r : locked_ranges) holes.emplace_back((uint8_t*) base + r.first, r.second);
        const strata::platform::LockResult lr = strata::platform::lock_resident_ranges(holes);
        locked_bytes = lr.locked_bytes;
        note = lr.note + " (" + std::to_string(hole_bytes >> 20) + " MiB of holes); " + note;
        // a partial lock is unlocked region by region in the destructor; what it did not reach was never locked
        // (VirtualUnlock/munlock of an unlocked page is harmless), so the list is kept whole
    }
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0 || bounds.size() < 2) return;
    base = reserve(bytes, backing, note, std::string{}, 0, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    slice_bytes = 1;   // sliced: one registration per layer
}

void PinnedArena::register_slices(std::atomic<int>& ready) {
    const int n = (int) bounds_.size() - 1;
    int i = 0;
    for (; base != nullptr && i < n; ++i) {
        const uint64_t off = bounds_[(size_t) i], len = bounds_[(size_t) i + 1] - off;
        if (cudaHostRegister((uint8_t*) base + off, (size_t) len, cudaHostRegisterPortable | cudaHostRegisterMapped) !=
            cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        slice_starts.push_back(off);
        registered_bytes = off + len;
        ++registered_slices;
        ready.store(i + 1, std::memory_order_release);
    }
    note = "cudaHostRegister per layer, pipelined with the load: " + std::to_string(registered_slices) + " of " +
           std::to_string(n) + " slices pinned; " + note;
    if (i < n && base != nullptr) {   // the rest stays resident through the working-set lock, as the sliced fallback does
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {           // #779: large pages cannot be paged out: nothing to lock
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr =
                strata::platform::lock_resident((uint8_t*) base + registered_bytes, capacity - registered_bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
    ready.store(INT_MAX, std::memory_order_release);   // every slice done (or given up on)
}

=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
PinnedArena::~PinnedArena() {
    if (base && by_need) {
        for (const auto& r : locked_ranges) strata::platform::unlock_resident((uint8_t*) base + r.first, r.second);
        for (const auto& r : pin_ranges) cudaHostUnregister((uint8_t*) base + r.first);
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
        return;
    }
    if (base) {
        if (locked_bytes) strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else {
            cudaHostUnregister(base);
        }
        release(base, capacity);
        base = nullptr;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk) {
    std::vector<uint8_t*> layer_dst;
    for (uint64_t off : layer_off) layer_dst.push_back(dst + off);
    return load_experts_ranges(path, layer_dst, layer_off, layer_bytes, threads, chunk);
}

<<<<<<< this fork
LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    return load_experts_ranges(path, dst, layer_off, layer_bytes, threads, chunk, ready, nullptr);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready, const PadMap* pad) {
=======
LoadStats load_experts_ranges(const std::string& path, const std::vector<uint8_t*>& layer_dst,
                              const std::vector<uint64_t>& layer_off, const std::vector<uint64_t>& layer_bytes, int threads,
                              uint64_t chunk, const std::function<void(uint64_t)>& done) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // one handle per thread, seeked once per layer: a shared handle would need a lock around the seek
            // and would serialise the very thing the threads are here to parallelise
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "cannot open " + path;
                return;
            }
            f.seekg((std::streamoff) off);
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                f.read((char*) buf.data(), (std::streamsize) n);
                if ((uint64_t) f.gcount() != n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L);
                    return;
                }
<<<<<<< this fork
                const auto t_copy = std::chrono::steady_clock::now();
                if (pad == nullptr) {
                    std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                } else {   // the padded layout: a chunk may span blobs, each lands at its own page-aligned start
                    const uint64_t B = pad->blob[(size_t) L], S = pad->stride[(size_t) L];
                    for (uint64_t q = 0; q < n;) {
                        const uint64_t p = pos + q, bi = p / B, in = p % B;
                        const uint64_t m = std::min<uint64_t>(n - q, B - in);
                        std::memcpy(dst + pad->dst_off[(size_t) L] + bi * S + in, buf.data() + q, (size_t) m);
                        q += m;
                    }
                }
=======
                std::memcpy(layer_dst[(size_t) L] + pos, buf.data(), (size_t) n);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
                h = fnv1a64(buf.data(), n, h);
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
            if (done) done(L);
        }
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        return st;
    }
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace strata::core
