#include "ll/api/memory/Hook.h"
#include "ll/api/mod/RegisterHelper.h"
#include "ll/api/mod/NativeMod.h"
#include "mc/client/renderer/game/ItemInHandRenderer.h"
#include "mc/world/item/ItemStack.h"
#include <glm/matrix.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <set>
#include <string>

namespace inspector {
using Clock = std::chrono::steady_clock;
using Mat = glm::mat4x4;
static std::atomic_bool active{false};
static std::mutex fileMutex;
static std::ofstream output;
static Clock::time_point started;
static std::map<std::string, std::pair<Clock::time_point, unsigned>> samples;
static unsigned total = 0;
static bool limitReported = false;
static std::set<std::string> observed;
static constexpr unsigned maxTotal = 50000;
static constexpr unsigned maxPerKey = 32;

// Thread-local context; never store entity pointers or item references.
struct Context { int flags = -1; int mainHand = -1; const char* source = "unknown"; };
static thread_local Context current;
struct Scope {
    Context previous;
    explicit Scope(Context value) noexcept : previous(current) { current = value; }
    ~Scope() { current = previous; }
};

static std::string targetName(ItemStack const& item) {
    if (!item.mCount) return {};
    auto name = item.getTypeName();
    if (name.empty() || name == "air" || name == "minecraft:air") return {};
    if (name.find(':') == std::string::npos) name = "minecraft:" + name;
    return name;
}
static std::optional<Mat> snapshot(MatrixStack::MatrixStackRef const& ref) {
    // v26.60.9 declares mat as Matrix* (scalar TypedStorage aliases the pointer).
    if (!ref.mat) return std::nullopt;
    return ref.mat->_m.get();
}
static bool finite(Mat const& m) {
    for (int c=0; c<4; ++c) for (int r=0; r<4; ++r)
        if (!std::isfinite(m[c][r])) return false;
    return true;
}
static void printMatrix(std::ostream& out, const char* label, Mat const& m) {
    out << label << " (rows; GLM storage m[column][row])\n";
    for (int r=0; r<4; ++r) {
        for (int c=0; c<4; ++c) out << (c ? " " : "  ") << m[c][r];
        out << '\n';
    }
}
static const char* view(int flags, bool held) {
    if (flags < 0) return "UNKNOWN";
    if (flags & 4) return "UI";
    if (flags & 1) return "FIRST_PERSON_PASS";
    if ((flags & 2) && (held || (flags & 32))) return "WORLD_HELD_CANDIDATE";
    return "OTHER";
}
static void record(std::string const& item, Context ctx, bool held, bool json, float scale,
                   Mat const& before, Mat const& after) {
    if (!finite(before) || !finite(after)) return;
    if (ctx.flags >= 0 && (ctx.flags & 4)) return;
    if (!held && !(ctx.flags >= 0 && (ctx.flags & (1|32)))) return;
    auto now = Clock::now();
    std::lock_guard guard(fileMutex);
    if (!active || !output) return;
    if (total >= maxTotal) {
        if (!limitReported) {
            output << "SESSION_SAMPLE_LIMIT reached; restart game for a new session\n" << std::flush;
            limitReported = true;
        }
        return;
    }
    std::string key = item + ":" + std::to_string(ctx.flags) + ":" + std::to_string(ctx.mainHand)
                    + ":" + std::to_string(json) + ":" + std::to_string(held);
    auto& entry = samples[key];
    if (entry.second >= maxPerKey || (entry.second && now - entry.first < std::chrono::milliseconds(250))) return;
    entry = {now, entry.second+1};
    output << "\nSAMPLE " << ++total << " elapsed_ms="
           << std::chrono::duration_cast<std::chrono::milliseconds>(now-started).count()
           << " item=" << item << " view=" << view(ctx.flags, held)
           << " flags=" << ctx.flags << " context=" << ctx.source << " main_hand=" << ctx.mainHand
           << " is_in_hand=" << held << " json_transform=" << json << " held_item_scale=" << scale << '\n';
    printMatrix(output,"BEFORE_DEFAULT",before);
    printMatrix(output,"AFTER_DEFAULT",after);
    float det = glm::determinant(before);
    if (std::isfinite(det) && std::abs(det) > 1e-12f) {
        Mat local = glm::inverse(before) * after;
        if (finite(local)) printMatrix(output,"CANDIDATE_POSTMULTIPLY_DELTA",local);
        Mat pre = after * glm::inverse(before);
        if (finite(pre)) printMatrix(output,"CANDIDATE_PREMULTIPLY_DELTA",pre);
    } else output << "DELTA_UNAVAILABLE singular_before\n";
    output << "END_SAMPLE\n" << std::flush;
}

// An item seen at the entry point is not proof that it reaches DefaultTransform.
static void observe(ItemStack const& item, Context ctx) {
    if (!active.load(std::memory_order_relaxed) || (ctx.flags & 4)) return;
    auto name = targetName(item);
    if (name.empty()) return;
    std::string key = name + ":" + std::to_string(ctx.flags) + ":" + std::to_string(ctx.mainHand);
    std::lock_guard guard(fileMutex);
    if (!active || !output || observed.size() >= maxTotal) return;
    if (observed.insert(key).second) {
        output << "OBSERVED_RENDER_ITEM item=" << name << " flags=" << ctx.flags
               << " main_hand=" << ctx.mainHand << '\n' << std::flush;
    }
}

// These three signatures are from the v26.60.9 CLIENT headers.
// All hooks call the unmodified original exactly once, even if logging fails.
LL_TYPE_INSTANCE_HOOK(FirstPersonContext, ItemInHandRenderer, ll::memory::HookPriority::Normal,
    &ItemInHandRenderer::renderFirstPerson, void,
    BaseActorRenderContext& context, Matrix const& prevProj, ItemContextFlags flags) {
    Scope s({static_cast<int>(flags) | 1, -1, "renderFirstPerson"});
    origin(context, prevProj, flags);
}
LL_TYPE_INSTANCE_HOOK(ItemContext, ItemInHandRenderer, ll::memory::HookPriority::Normal,
    &ItemInHandRenderer::renderItem, void,
    BaseActorRenderContext& context, Actor& entity, ItemStack const& item, bool fromJson,
    ItemContextFlags flags, bool useMatrixAsIs, bool renderingMainHand) {
    Scope s({static_cast<int>(flags) | (current.flags >= 0 ? (current.flags & 1) : 0), renderingMainHand ? 1 : 0, "renderItem"});
    try { observe(item, current); } catch (...) { /* diagnostic only */ }
    origin(context, entity, item, fromJson, flags, useMatrixAsIs, renderingMainHand);
}
LL_TYPE_INSTANCE_HOOK(DefaultTransform, ItemInHandRenderer, ll::memory::HookPriority::Normal,
    &ItemInHandRenderer::_applyDefaultItemTransforms, void,
    MatrixStack::MatrixStackRef& worldMatrix, ItemStack const& item, bool isInHandItem,
    BlockType const* blockType, BlockShape blockShape, ItemRenderCall const* renderObjectCall,
    float heldItemScale, bool posAndRotSetByJSON) {
    std::string name;
    std::optional<Mat> before;
    Context ctx = current;
    if (active.load(std::memory_order_relaxed)) {
        try {
            name = targetName(item);
            if (!name.empty()) before = snapshot(worldMatrix);
        } catch (...) { before.reset(); }
    }
    origin(worldMatrix, item, isInHandItem, blockType, blockShape, renderObjectCall, heldItemScale, posAndRotSetByJSON);
    if (before) {
        try {
            auto after = snapshot(worldMatrix);
            if (after) record(name, ctx, isInHandItem, posAndRotSetByJSON, heldItemScale, *before, *after);
        } catch (...) { /* Diagnostic failures must not change rendering. */ }
    }
}

class Inspector {
    ll::mod::NativeMod& self = *ll::mod::NativeMod::current();
    bool firstHook=false, itemHook=false, defaultHook=false;
public:
    static Inspector& instance() { static Inspector value; return value; }
    bool load() { return true; }
    bool enable() {
        try {
            auto dir = self.getModDir();
            std::filesystem::create_directories(dir);
            auto path = dir / "ItemTransformDump.txt";
            {
                std::lock_guard guard(fileMutex);
                output.open(path, std::ios::out | std::ios::app);
                if (!output) { self.getLogger().error("Cannot open ItemTransformDump.txt"); return false; }
                output.imbue(std::locale::classic());
                output << std::setprecision(9);
                started = Clock::now(); samples.clear(); observed.clear(); total=0; limitReported=false;
                output << "\n=== BE Item Transform Inspector 0.1.2 | SDK 26.60.9 | new session ===\n"
                       << "All item IDs accepted; held default-transform probe only, not a full hand-bone or mesh capture.\n"
                       << "flags: first=1 world=2 UI=4 item2d=8 item3d=16 inhand=32 glint=64 tint=128; -1 unknown\n"
                       << "Both matrix multiplication hypotheses are logged. No Euler angles inferred.\n"
                       << "OBSERVED_RENDER_ITEM is entry-only, not a successful matrix sample. Limits: 32/key, 50000/session.\n" << std::flush;
            }
            int a = FirstPersonContext::hook(); firstHook = a == 0;
            int b = ItemContext::hook(); itemHook = b == 0;
            int c = DefaultTransform::hook(); defaultHook = c == 0;
            { std::lock_guard guard(fileMutex);
              output << "HOOK_STATUS first=" << a << " item=" << b << " default=" << c << '\n' << std::flush; }
            if (!defaultHook) { self.getLogger().error("Default transform hook failed; see ItemTransformDump.txt"); disable(); return false; }
            active.store(true);
            self.getLogger().info("Item inspector enabled; output: {}", path.string());
            return true;
        } catch (...) { disable(); return false; }
    }
    bool disable() {
        active.store(false);
        if (defaultHook) { DefaultTransform::unhook(); defaultHook=false; }
        if (itemHook) { ItemContext::unhook(); itemHook=false; }
        if (firstHook) { FirstPersonContext::unhook(); firstHook=false; }
        std::lock_guard guard(fileMutex);
        if (output.is_open()) { output << "SESSION_END samples=" << total << '\n'; output.close(); }
        return true;
    }
};
} // namespace inspector
LL_REGISTER_MOD(inspector::Inspector, inspector::Inspector::instance());
