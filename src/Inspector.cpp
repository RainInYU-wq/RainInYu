#include "ll/api/memory/Hook.h"
#include "ll/api/mod/RegisterHelper.h"
#include "ll/api/mod/NativeMod.h"
#include "mc/client/renderer/game/ItemInHandRenderer.h"
#include "mc/world/item/ItemStack.h"
#include "mc/client/renderer/BaseActorRenderContext.h"
#include "mc/client/gui/screens/ScreenContext.h"
#include "mc/deps/renderer/Camera.h"
#include "mc/client/model/models/DataDrivenModel.h"
#include "mc/client/model/models/SkinnedMesh.h"
#include "mc/client/model/geom/ModelPart.h"
#include "mc/client/model/geom/Cube.h"
#include "mc/client/model/geom/PolygonQuad.h"
#include "mc/client/renderer/Tessellator.h"
#include "mc/deps/minecraft_renderer/game/ItemRenderCall.h"
#include "mc/deps/minecraft_renderer/resources/MERSUniformData.h"
#include "mc/world/actor/RenderParams.h"
#include <glm/matrix.hpp>
#include <vector>
#include <cstdint>
#include <string_view>
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
static constexpr unsigned maxTotal = 2048;
static constexpr unsigned maxPerKey = 24;
static std::map<std::string, std::pair<Clock::time_point, unsigned>> stages;
static unsigned stageTotal = 0;
static std::string captureTarget = "minecraft:iron_pickaxe";
static bool want(std::string const& name) { return name == captureTarget || captureTarget == "*"; }

// Thread-local context; never store entity pointers or item references.
struct Context { int flags = -1; int mainHand = -1; const char* source = "unknown"; };
static thread_local Context current;
static thread_local std::string_view currentItem;
static thread_local int attachDepth = 0;
struct ItemScope {
    std::string_view previous;
    explicit ItemScope(std::string_view name) noexcept : previous(currentItem) { currentItem = name; }
    ~ItemScope() { currentItem = previous; }
};
struct AttachScope { AttachScope() { ++attachDepth; } ~AttachScope() { --attachDepth; } };
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
    std::unique_lock guard(fileMutex, std::try_to_lock);
    if (!guard.owns_lock() || !active || !output) return;
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
    if (name.empty() || !want(name)) return;
    std::string key = name + ":" + std::to_string(ctx.flags) + ":" + std::to_string(ctx.mainHand);
    std::unique_lock guard(fileMutex, std::try_to_lock);
    if (!guard.owns_lock() || !active || !output || observed.size() >= maxTotal) return;
    if (observed.insert(key).second) {
        output << "OBSERVED_RENDER_ITEM item=" << name << " flags=" << ctx.flags
               << " main_hand=" << ctx.mainHand << '\n' << std::flush;
    }
}

// These three signatures are from the v26.60.9 CLIENT headers.
// All hooks call the unmodified original exactly once, even if logging fails.
LL_TYPE_INSTANCE_HOOK(FirstPersonContext, ll::memory::HookPriority::Normal, ItemInHandRenderer,
    &ItemInHandRenderer::renderFirstPerson, void,
    BaseActorRenderContext& context, Matrix const& prevProj, ItemContextFlags flags) {
    Scope s({static_cast<int>(flags) | 1, -1, "renderFirstPerson"});
    origin(context, prevProj, flags);
}
LL_TYPE_INSTANCE_HOOK(ItemContext, ll::memory::HookPriority::Normal, ItemInHandRenderer,
    &ItemInHandRenderer::renderItem, void,
    BaseActorRenderContext& context, Actor& entity, ItemStack const& item, bool fromJson,
    ItemContextFlags flags, bool useMatrixAsIs, bool renderingMainHand) {
    Scope s({static_cast<int>(flags) | (current.flags >= 0 ? (current.flags & 1) : 0), renderingMainHand ? 1 : 0, "renderItem"});
    std::string name;
    if (active.load(std::memory_order_relaxed)) {
        try { name = targetName(item); observe(item, current); } catch (...) { name.clear(); }
    }
    ItemScope itemScope(name);
    origin(context, entity, item, fromJson, flags, useMatrixAsIs, renderingMainHand);
}
LL_TYPE_INSTANCE_HOOK(DefaultTransform, ll::memory::HookPriority::Normal, ItemInHandRenderer,
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
            if (!name.empty() && want(name)) before = snapshot(worldMatrix);
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


// All stage labels describe raw observations, not a certified common coordinate frame.
static bool reserveStage(std::string const& key, unsigned limit = 24, int intervalMs = 150) {
    std::unique_lock lock(fileMutex, std::try_to_lock);
    if (!lock.owns_lock() || !active || !output || stageTotal >= 2048) return false;
    auto now = Clock::now();
    auto& e = stages[key];
    if (e.second >= limit || (e.second && now-e.first < std::chrono::milliseconds(intervalMs))) return false;
    e = {now, e.second+1}; ++stageTotal;
    return true;
}
static void writeStage(std::string const& data) {
    std::unique_lock lock(fileMutex, std::try_to_lock);
    if (lock.owns_lock() && active && output) output << data << std::flush;
}
static void initStream(std::ostringstream& out) { out.imbue(std::locale::classic()); out << std::setprecision(9); }
static const char* screenView(ScreenContext const& sc) {
    if (sc.isDrawingUI) return "UI";
    return sc.isDrawingFirstPersonObjects || (current.flags >= 0 && (current.flags & 1))
        ? "FIRST_PERSON_CANDIDATE" : "WORLD_CANDIDATE";
}
static void dumpCamera(std::ostream& out, ScreenContext const& sc) {
    auto const& cam = sc.camera;
    if (!cam.worldMatrixStack->stack->empty()) printMatrix(out,"CAMERA_WORLD_STACK_TOP",cam.worldMatrixStack->top()._m.get());
    if (!cam.viewMatrixStack->stack->empty()) printMatrix(out,"CAMERA_VIEW_STACK_TOP",cam.viewMatrixStack->top()._m.get());
    // Projection is intentionally excluded; common pre-projection space still needs verification.
}
static void stageHeader(std::ostream& out, const char* stage, std::string_view name, ScreenContext const& sc) {
    out << "\nSTAGE " << stage << " item=" << name << " view=" << screenView(sc)
        << " flags=" << current.flags << " attach_depth=" << attachDepth
        << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-started).count() << '\n';
}
static bool stageTarget(std::string_view n) { return !n.empty() && (n == captureTarget || captureTarget == "*"); }
static std::string stageKey(const char* label, std::string_view name, ScreenContext const& sc) {
    return std::string(label)+":"+std::string(name)+":"+screenView(sc);
}
static void cameraStage(const char* label, std::string_view name, ScreenContext const& sc, int slot=-1) {
    if (sc.isDrawingUI || !stageTarget(name) || !reserveStage(stageKey(label,name,sc))) return;
    std::ostringstream out; initStream(out); stageHeader(out,label,name,sc);
    out << "slot=" << slot << '\n'; dumpCamera(out,sc); out << "END_STAGE\n"; writeStage(out.str());
}

LL_TYPE_INSTANCE_HOOK(NativeDraw, ll::memory::HookPriority::Normal, ItemInHandRenderer,
    &ItemInHandRenderer::renderObject, void,
    BaseActorRenderContext& ctx, ItemRenderCall const& call, dragon::RenderMetadata const& metadata, ItemContextFlags flags) {
    if (active.load(std::memory_order_relaxed) && stageTarget(currentItem)) {
        try {
            auto const& sc=ctx.mScreenContext;
            if (!sc.isDrawingUI && reserveStage(stageKey("NATIVE_RENDER_OBJECT_ENTRY",currentItem,sc))) {
                std::ostringstream out; initStream(out); stageHeader(out,"NATIVE_RENDER_OBJECT_ENTRY",currentItem,sc);
                out << "call_session_address=" << reinterpret_cast<std::uintptr_t>(&call) << " object_flags=" << static_cast<int>(flags) << '\n';
                dumpCamera(out,sc); out << "END_STAGE\n"; writeStage(out.str());
            }
        } catch (...) {}
    }
    origin(ctx,call,metadata,flags);
}

// Scoped texture-mesh input vertices. These are Tessellator inputs, NOT GPU readback.
struct VertexSample {
    float x,y,z,u,v; bool hasUV, applyTransform;
    Mat transform;
    glm::vec3 offset,scale;
};
static thread_local std::vector<VertexSample>* vertexBuffer = nullptr;
struct VertexScope {
    std::vector<VertexSample>* previous;
    explicit VertexScope(std::vector<VertexSample>* buffer) : previous(vertexBuffer) { vertexBuffer=buffer; }
    ~VertexScope() { vertexBuffer=previous; }
};
static std::atomic_bool vertexHookInstalled{false};
LL_TYPE_INSTANCE_HOOK(VertexInput, ll::memory::HookPriority::Normal, Tessellator,
    &Tessellator::vertex, void, float x, float y, float z) {
    if (active.load(std::memory_order_relaxed) && vertexBuffer && vertexBuffer->size() < 8192) {
        try {
            auto const& uv=mNextUV[0].get();
            auto const& o=mPostTransformOffset.get(); auto const& s=mPostTransformScale.get();
            vertexBuffer->push_back({x,y,z,uv?uv->x:0.f,uv?uv->y:0.f,uv.has_value(),mApplyTransform,
                mApplyTransform ? mTransformMatrix.get() : Mat(1.0f),{o.x,o.y,o.z},{s.x,s.y,s.z}});
        } catch (...) {}
    }
    origin(x,y,z);
}
LL_TYPE_INSTANCE_HOOK(NativeMeshBuild, ll::memory::HookPriority::Normal, ItemInHandRenderer,
    &ItemInHandRenderer::getRenderCallAtFrame, ItemRenderCall const&,
    BaseActorRenderContext& ctx, ItemStack const& item, int frame) {
    std::string name;
    bool capture=false;
    if (active.load(std::memory_order_relaxed) && vertexHookInstalled && !vertexBuffer) {
        try { name=targetName(item); capture=!name.empty() && want(name) && reserveStage("MESH_ATTEMPT:"+name,48,500); } catch (...) {}
    }
    std::vector<VertexSample> vertices;
    if (capture) { try { vertices.reserve(8192); } catch (...) { capture=false; } }
    VertexScope scope(capture ? &vertices : vertexBuffer);
    auto const& result=origin(ctx,item,frame);
    if (capture && !vertices.empty() && reserveStage("MESH_OUTPUT:"+name,4,0)) {
        try {
            std::ostringstream out; initStream(out); stageHeader(out,"NATIVE_TESSELLATOR_INPUT",name,ctx.mScreenContext);
            out << "frame=" << frame << " call_session_address=" << reinterpret_cast<std::uintptr_t>(&result) << " vertices=" << vertices.size() << " truncated_possible=" << (vertices.size()==8192) << '\n';
            out << "columns: x y z has_uv u v apply_transform offset_xyz scale_xyz matrix_rows_16\n";
            for (auto const& v:vertices) {
                out << "V " << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.hasUV << ' ' << v.u << ' ' << v.v << ' ' << v.applyTransform
                    << ' ' << v.offset.x << ' ' << v.offset.y << ' ' << v.offset.z << ' ' << v.scale.x << ' ' << v.scale.y << ' ' << v.scale.z;
                for(int r=0;r<4;++r) for(int c=0;c<4;++c) out << ' ' << v.transform[c][r];
                out << '\n';
            }
            out << "END_STAGE\n"; writeStage(out.str());
        } catch (...) {}
    }
    return result;
}

// Keep item identity only on the calling thread and only while the original runs.
LL_TYPE_INSTANCE_HOOK(AttachEntry, ll::memory::HookPriority::Normal, DataDrivenModel,
    &DataDrivenModel::renderAttachable, void,
    ItemStack const& item, AttachableSlotIndex const& slot, RenderParams& params, Actor& actor) {
    std::string name;
    if (active.load(std::memory_order_relaxed)) {
        try { name=targetName(item); if (params.mBaseActorRenderContext) cameraStage("ATTACH_ENTRY",name,params.mBaseActorRenderContext->mScreenContext,static_cast<int>(slot)); } catch (...) {}
    }
    ItemScope itemScope(name); AttachScope attach;
    origin(item,slot,params,actor);
}
LL_TYPE_INSTANCE_HOOK(AttachInner, ll::memory::HookPriority::Normal, DataDrivenModel,
    &DataDrivenModel::renderAttachableNoChecks, void,
    ItemStack const& item, AttachableSlotIndex const& slot, RenderParams& params, Actor& actor) {
    std::string name;
    if (active.load(std::memory_order_relaxed)) {
        try { name=targetName(item); if (params.mBaseActorRenderContext) cameraStage("ATTACH_INNER_ENTRY",name,params.mBaseActorRenderContext->mScreenContext,static_cast<int>(slot)); } catch (...) {}
    }
    ItemScope itemScope(name); AttachScope attach;
    origin(item,slot,params,actor);
}
static void dumpVec(std::ostream& out, Vec3 const& v) { out << v.x << ' ' << v.y << ' ' << v.z; }
static void dumpPart(std::ostream& out, ModelPart const& p, bool geometry) {
    out << "PART name=" << p.mName->getString() << " part_index=" << p.mPartIndex << " mesh_index=" << p.mMeshIndex
        << " orientation_index=" << p.mBoneOrientationIndex << " visible=" << p.mVisible << " pos="; dumpVec(out,p.mPos.get());
    out << " rot="; dumpVec(out,p.mRot.get()); out << " scale="; dumpVec(out,p.mScale.get());
    out << " offset="; dumpVec(out,p.mOffset.get()); out << '\n';
    if (!geometry) return;
    auto const& cubes=p.mCubes.get();
    out << "CUBE_COUNT " << cubes.size() << '\n';
    for (size_t i=0;i<cubes.size() && i<512;++i) {
        auto const& c=cubes[i]; out << "CUBE " << i << " origin="; dumpVec(out,c.mOrigin.get());
        out << " size="; dumpVec(out,c.mSize.get()); out << " rotation="; dumpVec(out,c.mRotation.get());
        out << " pivot="; dumpVec(out,c.mCubePivot.get()); out << " mirrored=" << c.mMirrored << '\n';
        for (size_t f=0;f<c.mFaceData->size();++f) {
            auto const& a=(*c.mFaceData)[f]; auto const& uv=a.mUV.get(); auto const& sz=a.mUVSize.get();
            out << "FACE " << f << " valid=" << a.mFaceValid << " uv=" << uv.x << ' ' << uv.y << " uvsize=" << sz.x << ' ' << sz.y
                << " uvrotation=" << static_cast<int>(a.mUvRotation) << '\n';
        }
    }
}
LL_TYPE_INSTANCE_HOOK(AttachDraw, ll::memory::HookPriority::Normal, SkinnedMesh,
    &SkinnedMesh::draw, void,
    ScreenContext& sc, dragon::RenderMetadata const& metadata, RenderController const* controller,
    uint layer, MaterialVariants const& materials, ClientPBRTextureData const& textures,
    SortedMeshDrawList* sorted, std::optional<glm::mat4x4> const& previous,
    std::optional<MERSUniformData> const& mers, ShadowContext shadow, bool dithering) {
    if (active.load(std::memory_order_relaxed) && attachDepth && stageTarget(currentItem) && !sc.isDrawingUI) {
        try {
            // Separate mesh/layer keys; don't conflate outline and base draws.
            auto key=stageKey("ATTACH_SKINNED_DRAW",currentItem,sc)+":"+std::to_string(reinterpret_cast<std::uintptr_t>(this))+":"+std::to_string(layer);
            if (reserveStage(key,24,150)) {
                std::ostringstream out; initStream(out); stageHeader(out,"ATTACH_SKINNED_DRAW",currentItem,sc);
                out << "mesh_session_id=" << reinterpret_cast<std::uintptr_t>(this) << " layer=" << layer
                    << " shadow_context=" << static_cast<int>(shadow) << " use_bones=" << mUseBones << " vertex_count=" << mVertexCount << " total_vertices=" << mTotalVertices << '\n';
                dumpCamera(out,sc);
                auto const& mats=mBoneMatrices.get(); out << "BONE_MATRIX_COUNT " << mats.size() << '\n';
                for (size_t i=0;i<mats.size() && i<256;++i) {
                    auto label="SKIN_BONE_MATRIX_"+std::to_string(i); printMatrix(out,label.c_str(),mats[i]);
                }
                auto const& parts=mSkinnedParts.get(); out << "PART_COUNT " << parts.size() << '\n';
                bool geometry=reserveStage("GEOMETRY:"+key,1,0);
                for (size_t i=0;i<parts.size() && i<256;++i) if (parts[i]) dumpPart(out,*parts[i],geometry);
                out << "END_STAGE\n"; writeStage(out.str());
            }
        } catch (...) {}
    }
    origin(sc,metadata,controller,layer,materials,textures,sorted,previous,mers,shadow,dithering);
}

class Inspector {
    ll::mod::NativeMod& self = *ll::mod::NativeMod::current();
    bool firstHook=false, itemHook=false, defaultHook=false, nativeDrawHook=false, meshBuildHook=false, attachEntryHook=false, attachInnerHook=false, attachDrawHook=false;
public:
    static Inspector& instance() { static Inspector value; return value; }
    bool load() { return true; }
    bool enable() {
        try {
            auto dir = self.getModDir();
            std::filesystem::create_directories(dir);
            // Unique per process/session; never overwrite previous capture.
            auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            auto path=dir / ("ItemTransformDump_v020_"+std::to_string(stamp)+".txt");
            for (unsigned i=1; std::filesystem::exists(path);++i) path=dir/("ItemTransformDump_v020_"+std::to_string(stamp)+"_"+std::to_string(i)+".txt");
            captureTarget="minecraft:iron_pickaxe";
            { std::ifstream config(dir/"capture_target.txt"); std::string value; if (config>>value) captureTarget=value; }
            std::string label="unspecified";
            { std::ifstream config(dir/"capture_label.txt"); std::getline(config,label); if(label.empty()) label="unspecified"; }
            {
                std::lock_guard guard(fileMutex);
                output.open(path, std::ios::out | std::ios::app);
                if (!output) { self.getLogger().error("Cannot open ItemTransformDump.txt"); return false; }
                output.imbue(std::locale::classic());
                output << std::setprecision(9);
                started = Clock::now(); samples.clear(); observed.clear(); stages.clear(); total=0; stageTotal=0; limitReported=false;
                output << "\n=== BE Item Transform Inspector 0.2.0-probe1 | SDK 26.60.9 | new session ===\n"
                       << "capture_label=" << label << " target=" << captureTarget << '\n'
                       << "Raw stage observations only; no automatic landmark pairing or final common matrix asserted.\n"
                       << "flags: first=1 world=2 UI=4 item2d=8 item3d=16 inhand=32 glint=64 tint=128; -1 unknown\n"
                       << "Both matrix multiplication hypotheses are logged. No Euler angles inferred.\n"
                       << "OBSERVED_RENDER_ITEM is entry-only, not a successful matrix sample. Limits: 24/key, 2048 default + 2048 stage reservations/session.\n" << std::flush;
            }
            int a = FirstPersonContext::hook(); firstHook = a == 0;
            int b = ItemContext::hook(); itemHook = b == 0;
            int c = DefaultTransform::hook(); defaultHook = c == 0;
            int d=NativeDraw::hook(); nativeDrawHook=d==0;
            int e=AttachEntry::hook(); attachEntryHook=e==0;
            int f=AttachInner::hook(); attachInnerHook=f==0;
            int g=AttachDraw::hook(); attachDrawHook=g==0;
            // BTG pass does not need to intercept the high-frequency tessellator.
            bool captureVertices = label.rfind("BTG",0)!=0 && label.rfind("btg",0)!=0;
            { std::ifstream config(dir/"capture_vertices.txt"); int value; if(config>>value) captureVertices=value!=0; }
            int h=-999, j=-999;
            if(captureVertices) { h=VertexInput::hook(); vertexHookInstalled=h==0;
                if(vertexHookInstalled) { j=NativeMeshBuild::hook(); meshBuildHook=j==0; }
            }
            { std::lock_guard guard(fileMutex);
              output << "HOOK_STATUS first=" << a << " item=" << b << " default=" << c << '\n'
                     << "HOOK_NOTE 0=success; -999=not_attempted\n"
                     << "HOOK_STATUS native_draw=" << d << " attach_entry=" << e << " attach_inner=" << f
                     << " attach_draw=" << g << " vertex=" << h << " mesh_build=" << j << '\n' << std::flush; }
            if (!defaultHook && !nativeDrawHook && !attachDrawHook) { self.getLogger().error("No matrix/draw hooks available"); disable(); return false; }
            active.store(true);
            self.getLogger().info("Item inspector enabled; output: {}", path.string());
            return true;
        } catch (...) { disable(); return false; }
    }
    bool disable() {
        active.store(false);
        // No file mutex held while unhooking. Never unhook from within render callbacks.
        if (meshBuildHook) { NativeMeshBuild::unhook(); meshBuildHook=false; }
        if (vertexHookInstalled) { VertexInput::unhook(); vertexHookInstalled=false; }
        if (attachDrawHook) { AttachDraw::unhook(); attachDrawHook=false; }
        if (attachInnerHook) { AttachInner::unhook(); attachInnerHook=false; }
        if (attachEntryHook) { AttachEntry::unhook(); attachEntryHook=false; }
        if (nativeDrawHook) { NativeDraw::unhook(); nativeDrawHook=false; }
        if (defaultHook) { DefaultTransform::unhook(); defaultHook=false; }
        if (itemHook) { ItemContext::unhook(); itemHook=false; }
        if (firstHook) { FirstPersonContext::unhook(); firstHook=false; }
        std::lock_guard guard(fileMutex);
        if (output.is_open()) { output << "SESSION_END samples=" << total << " stage_reservations=" << stageTotal << '\n' << std::flush; output.close(); }
        return true;
    }
};
} // namespace inspector
LL_REGISTER_MOD(inspector::Inspector, inspector::Inspector::instance());
