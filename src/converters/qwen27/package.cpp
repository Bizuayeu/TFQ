// Zero-dependency fast C++17 residual packager for Qwen3.8-27B TFQ3 checkpoints.
// Accepts BOTH official HF BF16 (Qwen/Qwen3.8-27B) AND MLX 8-bit/64 as source,
// and emits standalone model.safetensors + config.json + tokenizer metadata with
// either --residual bf16 (pure BF16 base, no quantization block!) or --residual q8.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static inline float bf16_to_f32(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float f; memcpy(&f, &w, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t w; memcpy(&w, &f, 4);
    uint32_t lsb = (w >> 16) & 1u;
    return (uint16_t)((w + 0x7FFFu + lsb) >> 16);
}

struct SafeReader {
    struct Meta { uint64_t off, len; std::string dtype; std::vector<int64_t> shape; };
    struct Entry {
        int fd = -1; uint8_t* ptr = nullptr; size_t len = 0; uint64_t base = 0;
        std::map<std::string, Meta> tensors;
    };
    std::string dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

    void init(const std::string& d) {
        dir = d;
        std::ifstream f(d + "/model.safetensors.index.json");
        if (f.good()) {
            std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t pos = txt.find("\"weight_map\"");
            pos = txt.find('{', pos);
            size_t end = txt.find('}', pos);
            std::string body = txt.substr(pos + 1, end - pos - 1);
            size_t i = 0;
            while (i < body.size()) {
                size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
                size_t q2 = body.find('"', q1 + 1);
                size_t q3 = body.find('"', q2 + 1);
                size_t q4 = body.find('"', q3 + 1);
                weight_map[body.substr(q1 + 1, q2 - q1 - 1)] = body.substr(q3 + 1, q4 - q3 - 1);
                i = q4 + 1;
            }
            return;
        }
        Entry& e = get_shard("model.safetensors");
        for (const auto& kv : e.tensors) weight_map[kv.first] = "model.safetensors";
    }

    bool has(const std::string& name) const { return weight_map.count(name) > 0; }

    Entry& get_shard(const std::string& fn) {
        auto it = shards.find(fn);
        if (it != shards.end()) return it->second;
        Entry e;
        std::string full = dir + "/" + fn;
        e.fd = open(full.c_str(), O_RDONLY);
        if (e.fd < 0) { fprintf(stderr, "Cannot open %s\n", full.c_str()); exit(1); }
        struct stat st{}; fstat(e.fd, &st); e.len = st.st_size;
        e.ptr = (uint8_t*)mmap(nullptr, e.len, PROT_READ, MAP_PRIVATE, e.fd, 0);
        uint64_t hlen = 0; memcpy(&hlen, e.ptr, 8);
        e.base = 8 + hlen;
        std::string hdr((const char*)e.ptr + 8, hlen);
        size_t p = 0;
        while (true) {
            size_t dk = hdr.find("\"data_offsets\"", p);
            if (dk == std::string::npos) break;
            size_t open_brace = hdr.rfind('{', dk);
            size_t close_brace = hdr.find('}', dk);
            size_t colon = hdr.rfind(':', open_brace);
            size_t q2 = hdr.rfind('"', colon);
            size_t q1 = hdr.rfind('"', q2 - 1);
            std::string tname = hdr.substr(q1 + 1, q2 - q1 - 1);
            std::string obj = hdr.substr(open_brace, close_brace - open_brace + 1);
            size_t lb = hdr.find('[', dk), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            uint64_t o0 = std::stoull(hdr.substr(lb + 1, cm - lb - 1));
            uint64_t o1 = std::stoull(hdr.substr(cm + 1, rb - cm - 1));
            std::string dtype;
            size_t dp = obj.find("\"dtype\"");
            if (dp != std::string::npos) {
                size_t d1 = obj.find('"', obj.find(':', dp) + 1);
                size_t d2 = obj.find('"', d1 + 1);
                dtype = obj.substr(d1 + 1, d2 - d1 - 1);
            }
            std::vector<int64_t> sh;
            size_t sp = obj.find("\"shape\"");
            if (sp != std::string::npos) {
                size_t slb = obj.find('[', sp), srb = obj.find(']', slb);
                std::string sb = obj.substr(slb + 1, srb - slb - 1);
                std::stringstream ss(sb); std::string tok;
                while (std::getline(ss, tok, ',')) if (!tok.empty()) sh.push_back(std::stoll(tok));
            }
            e.tensors[tname] = {o0, o1 - o0, dtype, std::move(sh)};
            p = close_brace + 1;
        }
        shards[fn] = std::move(e);
        return shards[fn];
    }

    const uint8_t* get(const std::string& name, Meta& meta) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) { fprintf(stderr, "Missing tensor %s\n", name.c_str()); exit(1); }
        Entry& sh = get_shard(it->second);
        meta = sh.tensors.at(name);
        return sh.ptr + sh.base + meta.off;
    }

    ~SafeReader() {
        for (auto& kv : shards) { if (kv.second.ptr) munmap(kv.second.ptr, kv.second.len); if (kv.second.fd >= 0) close(kv.second.fd); }
    }
};

struct OutTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

int main(int argc, char** argv) {
    std::string src_dir, out_dir, residual = "bf16";
    int layers = 64;
    bool keep_head = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--src" && i + 1 < argc) src_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--residual" && i + 1 < argc) residual = argv[++i];
        else if (a == "--layers" && i + 1 < argc) layers = atoi(argv[++i]);
        else if (a == "--keep-head") keep_head = true;
    }
    if (src_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --src <dir> --out <dir> [--residual bf16|q8] [--keep-head]\n", argv[0]);
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);

    SafeReader rd;
    rd.init(src_dir);
    std::string src_pfx, head_src;
    if (rd.has("model.language_model.embed_tokens.weight")) { src_pfx = "model.language_model."; head_src = "lm_head"; }
    else if (rd.has("language_model.model.embed_tokens.weight")) { src_pfx = "language_model.model."; head_src = "language_model.lm_head"; }
    else if (rd.has("model.embed_tokens.weight")) { src_pfx = "model."; head_src = "lm_head"; }
    else { fprintf(stderr, "Unrecognized Qwen3.8-27B namespace in %s\n", src_dir.c_str()); return 1; }

    SafeReader::Meta l0m;
    const uint16_t* l0p = (const uint16_t*)rd.get(src_pfx + "layers.0.input_layernorm.weight", l0m);
    double mean = 0.0;
    size_t n0 = l0m.len / 2;
    for (size_t i = 0; i < n0; ++i) mean += bf16_to_f32(l0p[i]);
    mean /= (double)n0;
    bool add_one = (mean < 0.5);
    printf("Packaging residual=%s from %s (pfx='%s', add_one_norms=%d)\n", residual.c_str(), src_dir.c_str(), src_pfx.c_str(), (int)add_one);

    std::vector<OutTensor> outs;
    auto emit_scalar = [&](const std::string& s_leaf, const std::string& d_name, bool norm_add, bool conv_fix, bool keep_f32 = false) {
        SafeReader::Meta m;
        const uint8_t* ptr = rd.get(src_pfx + s_leaf, m);
        OutTensor ot; ot.name = d_name; ot.dtype = m.dtype; ot.shape = m.shape;
        if (conv_fix && ot.shape.size() == 3 && ot.shape[1] == 1 && ot.shape[2] == 4) {
            ot.shape[1] = 4; ot.shape[2] = 1;
        }
        if (m.dtype == "F32" && !keep_f32) {
            ot.dtype = "BF16";
            size_t n = m.len / 4;
            ot.data.resize(n * 2);
            const float* src = (const float*)ptr;
            uint16_t* dst = (uint16_t*)ot.data.data();
            float delta = (norm_add && add_one) ? 1.0f : 0.0f;
            for (size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16(src[i] + delta);
        } else {
            ot.data.resize(m.len);
            if (norm_add && add_one && m.dtype == "BF16") {
                const uint16_t* src = (const uint16_t*)ptr;
                uint16_t* dst = (uint16_t*)ot.data.data();
                for (size_t i = 0; i < m.len / 2; ++i) dst[i] = f32_to_bf16(bf16_to_f32(src[i]) + 1.0f);
            } else {
                memcpy(ot.data.data(), ptr, m.len);
            }
        }
        outs.push_back(std::move(ot));
    };

    auto emit_linear = [&](const std::string& s_mod, const std::string& d_mod, int64_t rows, int64_t cols) {
        bool is_q8 = rd.has(s_mod + ".scales");
        if (residual == "bf16") {
            OutTensor ot; ot.name = d_mod + ".weight"; ot.dtype = "BF16"; ot.shape = {rows, cols};
            ot.data.resize((size_t)rows * (size_t)cols * 2ULL);
            if (is_q8) {
                SafeReader::Meta wm, sm, bm;
                const uint8_t* wp = rd.get(s_mod + ".weight", wm);
                const uint16_t* sp = (const uint16_t*)rd.get(s_mod + ".scales", sm);
                const uint16_t* bp = (const uint16_t*)rd.get(s_mod + ".biases", bm);
                uint16_t* dst = (uint16_t*)ot.data.data();
                int64_t ng = cols / 64;
                for (int64_t r = 0; r < rows; ++r) {
                    const uint8_t* wr = wp + (size_t)r * (size_t)cols;
                    const uint16_t* sr = sp + (size_t)r * (size_t)ng;
                    const uint16_t* br = bp + (size_t)r * (size_t)ng;
                    uint16_t* dr = dst + (size_t)r * (size_t)cols;
                    for (int64_t g = 0; g < ng; ++g) {
                        float sc = bf16_to_f32(sr[g]), bi = bf16_to_f32(br[g]);
                        for (int64_t j = 0; j < 64; ++j) dr[g * 64 + j] = f32_to_bf16((float)wr[g * 64 + j] * sc + bi);
                    }
                }
            } else {
                SafeReader::Meta wm;
                const uint8_t* wp = rd.get(s_mod + ".weight", wm);
                if (wm.dtype != "BF16" || wm.len != ot.data.size()) { fprintf(stderr, "Bad BF16 %s\n", s_mod.c_str()); exit(1); }
                memcpy(ot.data.data(), wp, wm.len);
            }
            outs.push_back(std::move(ot));
        } else {
            if (!is_q8) { fprintf(stderr, "--residual q8 needs MLX q8 source for %s\n", s_mod.c_str()); exit(1); }
            for (const char* suf : {".weight", ".scales", ".biases"}) {
                SafeReader::Meta m;
                const uint8_t* ptr = rd.get(s_mod + suf, m);
                OutTensor ot; ot.name = d_mod + suf; ot.dtype = m.dtype; ot.shape = m.shape;
                ot.data.assign(ptr, ptr + m.len);
                outs.push_back(std::move(ot));
            }
        }
    };

    emit_linear(src_pfx + "embed_tokens", "model.embed_tokens", 248320, 5120);
    emit_scalar("norm.weight", "model.norm.weight", true, false);
    if (keep_head) emit_linear(head_src, "lm_head", 248320, 5120);

    for (int l = 0; l < layers; ++l) {
        std::string p = "layers." + std::to_string(l) + ".";
        std::string dp = "model.layers." + std::to_string(l) + ".";
        emit_scalar(p + "input_layernorm.weight", dp + "input_layernorm.weight", true, false);
        emit_scalar(p + "post_attention_layernorm.weight", dp + "post_attention_layernorm.weight", true, false);
        if ((l + 1) % 4 != 0) {
            emit_scalar(p + "linear_attn.conv1d.weight", dp + "linear_attn.conv1d.weight", false, true);
            emit_scalar(p + "linear_attn.A_log", dp + "linear_attn.A_log", false, false, true);
            emit_scalar(p + "linear_attn.dt_bias", dp + "linear_attn.dt_bias", false, false);
            emit_scalar(p + "linear_attn.norm.weight", dp + "linear_attn.norm.weight", false, false);
            emit_linear(src_pfx + p + "linear_attn.in_proj_a", dp + "linear_attn.in_proj_a", 48, 5120);
            emit_linear(src_pfx + p + "linear_attn.in_proj_b", dp + "linear_attn.in_proj_b", 48, 5120);
        } else {
            emit_scalar(p + "self_attn.q_norm.weight", dp + "self_attn.q_norm.weight", true, false);
            emit_scalar(p + "self_attn.k_norm.weight", dp + "self_attn.k_norm.weight", true, false);
        }
    }

    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        if (i) hdr += ",";
        hdr += "\"" + outs[i].name + "\":{\"dtype\":\"" + outs[i].dtype + "\",\"shape\":[";
        for (size_t j = 0; j < outs[i].shape.size(); ++j) {
            if (j) hdr += ",";
            hdr += std::to_string(outs[i].shape[j]);
        }
        uint64_t end = off + outs[i].data.size();
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(end) + "]}";
        off = end;
    }
    hdr += "}";
    while (hdr.size() % 8 != 0) hdr.push_back(' ');

    std::string out_st = out_dir + "/model.safetensors";
    FILE* fo = fopen(out_st.c_str(), "wb");
    if (!fo) { fprintf(stderr, "Cannot write %s\n", out_st.c_str()); return 1; }
    uint64_t hlen = hdr.size();
    fwrite(&hlen, 8, 1, fo);
    fwrite(hdr.data(), 1, hlen, fo);
    for (const auto& ot : outs) fwrite(ot.data.data(), 1, ot.data.size(), fo);
    fclose(fo);
    printf("Wrote %s (%zu tensors, %.3f GiB)\n", out_st.c_str(), outs.size(), (double)(8 + hlen + off) / (double)(1ULL << 30));

    auto copy_if_exists = [&](const std::string& fn) {
        std::ifstream in(src_dir + "/" + fn, std::ios::binary);
        if (!in.good()) return;
        std::ofstream out(out_dir + "/" + fn, std::ios::binary);
        out << in.rdbuf();
    };
    for (const char* fn : {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"}) {
        copy_if_exists(fn);
    }

    // Copy config.json; for --residual bf16, strip any top-level MLX quantization block cleanly.
    std::ifstream cf(src_dir + "/config.json");
    if (cf.good()) {
        std::string cfg_txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        if (residual == "bf16") {
            for (const char* key : {"\"quantization\"", "\"quantization_config\""}) {
                while (true) {
                    size_t kp = cfg_txt.find(key);
                    if (kp == std::string::npos) break;
                    size_t col = cfg_txt.find(':', kp + strlen(key));
                    if (col == std::string::npos) break;
                    size_t v0 = cfg_txt.find_first_not_of(" \t\r\n", col + 1);
                    size_t ve = v0;
                    if (v0 != std::string::npos && cfg_txt[v0] == '{') {
                        int depth = 0; bool in_str = false;
                        for (ve = v0; ve < cfg_txt.size(); ++ve) {
                            char ch = cfg_txt[ve];
                            if (ch == '"' && (ve == 0 || cfg_txt[ve - 1] != '\\')) in_str = !in_str;
                            if (!in_str) {
                                if (ch == '{') depth++;
                                else if (ch == '}' && --depth == 0) { ve++; break; }
                            }
                        }
                    } else {
                        ve = cfg_txt.find_first_of(",}\n", v0);
                    }
                    // Consume either trailing comma or preceding comma.
                    size_t after = cfg_txt.find_first_not_of(" \t\r\n", ve);
                    size_t del_start = kp, del_end = ve;
                    if (after != std::string::npos && cfg_txt[after] == ',') {
                        del_end = after + 1;
                    } else {
                        size_t before = cfg_txt.find_last_not_of(" \t\r\n", kp ? kp - 1 : 0);
                        if (before != std::string::npos && cfg_txt[before] == ',') del_start = before;
                    }
                    cfg_txt.erase(del_start, del_end - del_start);
                }
            }
        }
        std::ofstream co(out_dir + "/config.json");
        co << cfg_txt;
        printf("Wrote %s/config.json (residual=%s)\n", out_dir.c_str(), residual.c_str());
    }
    return 0;
}
