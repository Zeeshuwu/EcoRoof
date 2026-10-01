# REPORT — Revisi Reviewer 2 (R2-2 & R2-3)
EcoPower Roof / SAM3 hierarchical patch inference — ICARES

Semua angka di dokumen ini berasal dari file di `revisi_r2/results/`; nama file sumber disebut di setiap tabel. Nilai dalam persen (%). "PAPER_CONFIG" = pipeline otomatis yang bisa direproduksi dan paling dekat dengan paper (definisinya di §1.1).

---

## 1. Ringkasan temuan

### 1.1 Rekonsiliasi angka paper (Fase 0)
Sumber: `phase0_reproduction.csv`, `phase0_all_sources_per_site.csv`, `per_site_metrics.csv`, `phase0_pv_revised_edits.csv`, `phase0_trace_9208.txt`.

1. **Cara agregasi di paper = rata-rata tak berbobot per site (bukan pooled piksel), dan hanya pada 3 site PV** (JKT_PV, JOG_PV, SBY_PV). Satu-satunya prediksi yang tersimpan dan cocok dengan paper adalah ekspor dashboard di `input/01 DATA/Results/*_REVISED/`, dan ekspor itu hanya ada untuk ketiga site tersebut. Padahal teks paper menyebut "six sites".
2. **Roof:** IoU (76.51) dan Precision (96.60) di paper **tereproduksi tepat**: 76.48 / 96.61. Mask roof `_REVISED` identik piksel dengan PAPER_CONFIG. Yang **tidak cocok** adalah Recall 75.23 dan F1 83.89 (terukur 78.53 / 86.56). Keduanya juga tidak konsisten dengan IoU dan Precision paper sendiri: dari IoU = 76.51 dan P = 96.60 diperoleh R ≈ 78.6 dan F1 ≈ 86.7. Kemungkinan salah salin atau berasal dari run lain. Accuracy juga berbeda (90.96 vs 90.16).
3. **PV:** IoU, P, R, F1 paper (84.71 / 87.06 / 96.95 / 91.69) cocok dengan `_REVISED` (84.74 / 87.06 / 96.98 / 91.71). **Namun `_REVISED` PV = ekspor otomatis dengan beberapa poligon dihapus secara manual.** JKT_PV: 8 dari 19 poligon (219 m²) dihapus, JOG_PV: 3 dari 23, SBY_PV: 1 dari 3. Semua poligon yang dihapus punya overlap PV-GT = 0, sementara FP lain tetap dipertahankan, jadi penghapusan tidak mengikuti aturan otomatis. Tanpa penyuntingan itu (PAPER_CONFIG), PV = **IoU 77.48, P 79.35, R 96.98, F1 87.27**.
4. **"80.61 % IoU"** = (76.51 + 84.71) / 2, yaitu rata-rata IoU dua kelas, sama dengan rata-rata 6 baris site×kelas `_REVISED`. **"92.08 % Precision" tidak dapat ditelusuri.** Nilai yang konsisten adalah (96.60 + 87.06) / 2 = **91.83**. Pencarian brute-force atas semua subset site×kelas, semua sumber, rata-rata maupun pooled, tidak menemukan sumber natural. Rekomendasi: ganti dengan 91.83 atau hapus kalimat itu, dan jangan menyebut "across all six sites".
5. Ambang confidence di `AdvancedConfig` (0.20/0.22/0.28) dan argumen `--confidence` **tidak berpengaruh**: `Sam3Processor` sudah membuang instance dengan skor ≤ 0.5. Ambang efektif = 0.5.
6. Teks paper menyebut NMS untuk menggabungkan level. Kode sebenarnya memakai **union piksel + consensus blending antar patch yang overlap (0.25 / 0.40) + validasi spasial L3 terhadap L1**, tanpa NMS. Bagian Methods perlu diselaraskan.
7. `post_process_mask` memuat filter geometri yang dirancang untuk jalan (membuang blob besar yang kompak), sehingga **atap besar ikut terbuang**. Mematikan filter ini saja menaikkan roof IoU 75.33 → 79.20 (`grid_full_data.csv`).
8. Minor: paper menyebut "30–280 rooftops per site", sedangkan GT berisi 20–87 poligon atap per site (`timing.csv`, kolom `n_gt_buildings`).

**PAPER_CONFIG** = `output_inference/*_mask.tif` (batch SAM3, prompt dashboard, skor > 0.5, `post_process_mask` default: min object 50 px, filter geometri ON, smoothing ON) + ekspor shapefile dashboard (connectivity 4, buang < 1 m², simplify 0.5 m) yang di-rasterisasi kembali. Tanpa penyuntingan manual. Roof dievaluasi pada 6 site dan PV pada 3 site (site ROOF tidak punya GT PV).

| PAPER_CONFIG vs paper (site-mean) | IoU | P | R | F1 | Acc |
|---|---|---|---|---|---|
| Roof, 6 site | 75.33 | 96.19 | 77.75 | 85.79 | 89.49 |
| Roof, 3 site PV | 76.48 | 96.61 | 78.53 | 86.56 | 90.16 |
| Paper roof | 76.51 | 96.60 | 75.23 | 83.89 | 90.96 |
| PV, 3 site | 77.48 | 79.35 | 96.98 | 87.27 | 99.41 |
| Paper PV (dengan edit manual) | 84.71 | 87.06 | 96.95 | 91.69 | 99.30 |

### 1.2 Cache & determinisme (Fase 1)
Sumber: `determinism.csv`, `timing.csv`, `cache_levels_log.txt`.
- Re-run 12 kombinasi (6 site × roof/pv) **identik 0 piksel** dengan `output_inference/*_mask.tif` dan `*_mask_prepost.tif`. Rakit ulang offline dari cache instance juga 0 piksel untuk semua 12. Artinya inferensi SAM3 deterministik pada mesin ini, dan analisis di atas cache setara dengan menjalankan ulang pipeline.
- Ditemukan detail implementasi: `generate_patches` menghasilkan patch duplikat di tepi citra, dan patch itu diproses dua kali serta ikut dihitung di consensus blending. Perilaku ini direproduksi apa adanya.
- **Waktu:** 118–129 s per kelas per tile 200 × 200 m (4 ha) di RTX 5060 Ti. L3 (256 px) memakan 74 % waktu, L2 (512 px) sekitar 28 s, L1 1–7 s. Kedua kelas: 239–251 s per tile (≈ 1.0 menit/ha). Per bangunan GT: 2.9–12.0 s (kedua kelas). Peak VRAM 4.3–8.7 GB. Run asli (manifest): 119–143 s per kelas.

### 1.3 R2-2 — variabilitas (Fase 2)
Sumber: `summary_table.csv/.md`, `mc_iterations.csv`, `bootstrap.csv`, `loso.csv`, `loco.csv`, `mc_selected_params.csv`, `grid_full_data.csv`, `block_inventory.csv`.
- Unit sampling: blok 25 × 25 m (64 per site). Blok tanpa GT dan tanpa prediksi apa pun dibuang: roof 357 dari 384 blok valid (27 dibuang), PV 72 dari 192 (120 dibuang).
- **Tanpa tuning (PAPER_CONFIG, bootstrap blok 1000×):** roof IoU 75.33, 95 % CI [72.76, 77.78]; F1 [83.97, 87.40]. PV IoU 77.48 [64.61, 84.03]; P [65.67, 86.32]. Angka roof paper (IoU/P) berada di dalam CI. **Angka PV paper (IoU 84.71, P 87.06, F1 91.69) berada di luar CI pipeline otomatis.**
- **Monte Carlo CV (100 × 70/30, parameter dikalibrasi ulang):** roof IoU **82.97 ± 1.12** (95 % CI [80.92, 85.03]), F1 **90.54 ± 0.69**. PV IoU **75.64 ± 11.06**, F1 **83.20 ± 10.79**. Dengan agregasi pooled: PV IoU 83.85 ± 5.51.
- Kalibrasi roof **selalu** (100/100) memilih τ = 0.35, min size 300 px, filter geometri OFF. Pemilihannya stabil, tapi τ berada di batas bawah cache. PV terbagi antara τ 0.55–0.70 dan min size 50/300, filter geometri selalu ON.
- **Generalisasi spasial:** LOSO roof IoU 83.13 ± 6.82 (tuned) vs 75.33 ± 6.72 (PAPER_CONFIG). LOCO roof 83.13 ± 5.99. PV LOSO 76.76 ± 13.81 (tuned) vs 77.48 ± 4.22 (PAPER_CONFIG). Untuk PV, tuning antar site **tidak** membantu: SBY_PV memilih τ = 0.70 dan IoU-nya turun ke 61.96.

### 1.4 R2-3 — hierarki (Fase 3)
Sumber: `ablation.csv`, `ablation_stats.csv`, `baseline_hierarchical.csv`, `baseline_slic_rules.txt`, `baseline_slic_timing.csv`, `conceptual_comparison.md`.
- **Roof:** hierarki penuh L1+L2+L3 terbaik (IoU 75.33) dan **lebih baik secara signifikan dari semua 6 subset** (Wilcoxon per blok, Holm, p ≤ 5 × 10⁻⁵). Level tunggal: L1 62.99, L2 70.44, L3 63.76. Dua level: 71.5–73.6.
- **PV:** hierarki penuh **bukan** yang terbaik. L2 saja memberi IoU 86.06 (P 93.82) vs penuh 77.48 (P 79.35), dan signifikan per blok (r = +0.71, p_Holm = 0.0016). Menambahkan L1 (full image) menambah FP PV. Pada blok ber-GT saja (sensitivitas), L2+L3 > penuh (p_Holm = 0.004).
- **Baseline hierarchical merging klasik (SLIC + RAG, 6 site):** versi terkalibrasi, pada blok test yang sama, roof IoU 68.86 vs SAM3 PAPER_CONFIG 73.59 vs SAM3 terkalibrasi 83.47; PV 33.97 vs 76.68. Batas atas oracle (label GT per segmen) turun tajam saat hierarki digabung makin kasar: roof 88.86 pada ΔE 5 (±1 244 segmen) → 62.26 pada ΔE 15 (±220) → 20.20 pada ΔE 30 (±19).

---

## 2. Draft teks paper (English, ready to paste)

### 2.1 Methods — new subsection "Statistical Evaluation Protocol"

> **E. Statistical Evaluation Protocol**
>
> Because SAM3 is applied zero-shot, no model parameters are trained. We therefore interpret the train/test split requested for variability analysis as a *calibration/test* split. The calibration set is used only to select post-processing parameters; the test set is used only for evaluation. Three parameters were calibrated without re-running SAM3: the instance confidence threshold τ ∈ {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70}, the minimum object size ∈ {50, 150, 300} px, and the geometric blob filter (on/off). This gives 42 combinations centred on the configuration used for the reported results (τ = 0.50, 50 px, filter on). All SAM3 instance masks with score > 0.35 were cached once per site and class, so every configuration is re-assembled offline with the same cross-level fusion and post-processing as the production pipeline. The re-assembled masks were verified to be pixel-identical to the pipeline output.
>
> To limit the influence of spatial autocorrelation, the resampling unit is a 25 × 25 m spatial block (64 blocks per 200 × 200 m site), not a pixel. Blocks containing neither reference nor predicted pixels under any configuration were excluded (27 of 384 roof blocks and 120 of 192 PV blocks). For every evaluation set, TP, FP, FN and TN are pooled over the blocks of each site, metrics are computed per site, and the unweighted mean over sites is reported (six sites for rooftops, the three PV sites for PV panels).
>
> (i) *Monte Carlo cross-validation.* In N = 100 iterations (seeds 42–141), the blocks of each site were randomly split 70/30 into calibration and test sets (stratified by site). The configuration with the highest calibration F1 was selected and evaluated on the test blocks. We report the mean ± standard deviation and the 2.5th–97.5th percentile interval over iterations.
> (ii) *Block bootstrap.* The fixed reported configuration was evaluated on 1,000 bootstrap resamples of blocks (with replacement, stratified by site) to obtain percentile 95 % confidence intervals without any tuning.
> (iii) *Spatial generalisation.* Leave-one-site-out (six folds for rooftops, three for PV) and leave-one-city-out (three folds) experiments calibrated on the remaining sites or cities and tested on the held-out one.
> (iv) *Hierarchy ablation.* Every non-empty subset of the three levels (full image L1, 512 px tiles L2, 256 px tiles L3) was fused and post-processed exactly as in the full pipeline. Each subset was compared with L1+L2+L3 using a two-sided Wilcoxon signed-rank test on paired per-block IoU and F1 (blocks without reference and prediction score 1), Holm-corrected across the 12 comparisons per class. Effect sizes are reported as the matched-pairs rank-biserial correlation r and the median per-block difference with a 2,000-sample bootstrap 95 % CI.
> All random procedures use a fixed base seed (42).

*(Also correct Section III.C: the three levels are fused by pixel-wise union, overlap-consensus blending of adjacent tiles (thresholds 0.25 for multiply covered pixels, 0.40 otherwise) and spatial validation of the 256 px level against the full-image level. The code contains no Non-Maximum Suppression.)*

### 2.2 Results — paragraphs and captions

**Variability (R2-2).**
> Table X summarises the variability of the segmentation metrics. Evaluated without any tuning, the reported configuration reaches a site-mean rooftop IoU of 75.33 % (95 % block-bootstrap CI 72.76–77.78 %), Precision 96.19 % (95.28–96.90 %), Recall 77.75 % (75.14–80.24 %) and F1 85.79 % (83.97–87.40 %). For PV panels it reaches IoU 77.48 % (64.61–84.03 %), Precision 79.35 % (65.67–86.32 %), Recall 96.98 % (95.90–98.17 %) and F1 87.27 % (77.24–91.26 %). When the post-processing parameters are calibrated on 70 % of the blocks and tested on the remaining 30 % (100 Monte Carlo iterations), rooftop performance rises to IoU 82.97 ± 1.12 % and F1 90.54 ± 0.69 %, with a narrow spread across iterations. PV performance is 75.64 ± 11.06 % IoU and 83.20 ± 10.79 % F1. The much larger PV spread reflects the small number of PV objects per test fold (one Surabaya site contains only two installations), not instability of the segmentation itself. Rooftop calibration selected the same setting in all 100 iterations (lower confidence threshold, geometric filter disabled). Leave-one-site-out and leave-one-city-out experiments give 83.13 ± 6.82 % and 83.13 ± 5.99 % rooftop IoU respectively, indicating that parameters calibrated in some cities transfer to unseen cities.

> **Table X.** Segmentation performance with variability estimates (%, site mean). Monte Carlo: 100 random 70/30 calibration/test splits of 25 × 25 m blocks, stratified by site; bootstrap: 1,000 block resamples of the fixed configuration; LOSO/LOCO: leave-one-site/city-out, mean ± SD over folds. *(Insert rows from `results/summary_table.md`.)*

> **Fig. X.** Distribution of test-block metrics over 100 Monte Carlo calibration/test splits for rooftops (6 sites) and PV panels (3 sites). Diamonds mark the values reported in the original submission. *(figures/fig_mc_distribution.png)*

> **Fig. Y.** Per-site metrics of the reported configuration with 95 % block-bootstrap confidence intervals. *(figures/fig_per_site_metrics.png)*

> **Fig. Z.** Frequency with which each post-processing setting was selected on the calibration blocks across 100 Monte Carlo iterations; * marks the setting used in the original submission. *(figures/fig_param_stability.png)*

**Per site.**
> Rooftop IoU ranges from 66.83 % (Jakarta, conventional-roof site) to 83.03 % (Yogyakarta, PV site), and PV IoU from 72.83 % (Yogyakarta) to 81.06 % (Jakarta). The lowest rooftop scores occur in dense residential fabric with tree occlusion and narrow gaps between adjacent roofs.

**Hierarchy ablation (R2-3).**
> Table Y and Fig. W isolate the contribution of each hierarchy level. For rooftops, every single level is clearly inferior to the full hierarchy (IoU 62.99 %, 70.44 % and 63.76 % for the full-image, 512 px and 256 px levels, versus 75.33 % for L1+L2+L3). Every two-level combination is also inferior (71.53–73.56 %). All six differences are significant at the block level after Holm correction (p ≤ 5 × 10⁻⁵; rank-biserial r between −0.28 and −0.77; median per-block IoU loss 0.07–4.88 pp). The levels are therefore complementary for large, heterogeneous roof surfaces. For PV panels the pattern differs: the 512 px level alone achieves the highest IoU (86.06 % vs 77.48 %, r = +0.71, Holm-adjusted p = 0.002) because the full-image level adds false positives on small, bright roof structures. This indicates that the hierarchy should be configured per class, with the full-image level reserved for large objects.

> **Table Y.** Hierarchy-level ablation (%, site mean, 95 % block-bootstrap CI). L1 = full image, L2 = 512 × 512 px tiles, L3 = 256 × 256 px tiles. *(results/ablation.csv; tests in results/ablation_stats.csv)*

> **Fig. W.** IoU and F1 of each hierarchy-level combination for rooftops and PV panels; error bars show 95 % block-bootstrap confidence intervals, shading marks the full hierarchy. *(figures/fig_ablation.png)*

**Classical hierarchical baseline.**
> As a classical bottom-up hierarchical baseline, we applied SLIC superpixels followed by agglomerative region-adjacency-graph merging on CIELAB colour (scikit-image `merge_hierarchical`) to all six sites. The dendrogram was cut at ΔE ∈ {5, 10, 15, 20, 30}. Because this method produces unlabelled regions, the regions were labelled by a depth-3 colour/size decision rule fitted on the calibration blocks (Monte Carlo split, seed 42), and the hierarchy cut was selected on the same blocks. On the common test blocks, this calibrated baseline reached 68.86 % rooftop IoU and 33.97 % PV IoU, compared with 73.59 % and 76.68 % for the reported SAM3 configuration and 83.47 % rooftop IoU for SAM3 with calibrated post-processing. Even when each region is labelled with the reference data (oracle), bottom-up merging degrades quickly as the hierarchy is cut coarser (rooftop IoU 88.86 % with ~1,240 regions per site at ΔE = 5, 62.26 % with ~220 regions at ΔE = 15, 20.20 % with ~19 regions at ΔE = 30). Colour homogeneity alone therefore does not delimit rooftops in RGB imagery, whereas the object-level masks of the foundation model do.

> **Table Z.** Comparison with a classical hierarchical merging baseline on identical test blocks (%, site mean). Oracle = regions labelled with reference data (upper bound, not a deployable method). *(results/baseline_hierarchical.csv)*

> **Fig. V.** Qualitative comparison for a Yogyakarta and a Surabaya site: orthomosaic, reference, SAM3 hierarchical patch inference and calibrated SLIC-RAG hierarchical merging. *(figures/fig_baseline_visual.png)*

### 2.3 Related Works / Discussion — hierarchical segmentation

> Hierarchical segmentation has a long history in remote sensing. Most approaches build the hierarchy bottom-up: pixels or superpixels are merged agglomeratively according to spectral similarity, and one level of the resulting tree is selected. Prades *et al.* [NN] follow this idea for hyperspectral imagery. After dimensionality reduction, an initial K-means partition is merged by model-based agglomerative clustering (with cluster densities estimated by independent component analysis), and a validation criterion selects the level whose clusters correspond to distinct materials. This yields an estimate of the number of endmembers and their spectra. Such methods are unsupervised and physically interpretable. However, they operate on spectral clusters rather than objects, they rely on the rich spectral signal of hyperspectral sensors, and they provide no semantic class. The hierarchy used in this work is different in kind. It is a top-down, multi-scale spatial decomposition (full scene, 512 px and 256 px tiles) in which one foundation model proposes class-specific object instances at each scale, and the scales are fused by union, overlap consensus and cross-scale validation. Its levels correspond to receptive fields rather than to merge thresholds. Our experiments on RGB orthomosaics show why this distinction matters. A classical bottom-up colour hierarchy (SLIC + region-adjacency merging) reaches at most 68.86 % rooftop IoU after calibration, whereas the multi-scale foundation-model hierarchy reaches 73.59 % without tuning and 83.47 % with calibrated post-processing on the same test blocks. The ablation further shows that the levels are complementary for rooftops but not for PV panels, which motivates class-specific level selection.
>
> [NN] J. Prades, G. Safont, A. Salazar, and L. Vergara, "Estimation of the number of endmembers in hyperspectral images using agglomerative clustering," *Remote Sens.*, vol. 12, no. 21, Art. no. 3585, 2020, doi: 10.3390/rs12213585.

---

## 3. Draft response to reviewer (English)

**Comment R2-2** (variability / Monte Carlo with random train/test splits)

> We thank the reviewer for this suggestion. Our framework applies SAM3 zero-shot, so no network weights are learned. We therefore implemented the requested experiment as a *calibration/test* protocol: the "training" portion is used only to select post-processing parameters (confidence threshold, minimum object size and a geometric filter; 42 combinations), and the "test" portion is used only for evaluation. To avoid the optimistic bias of pixel-level resampling caused by spatial autocorrelation, the sampling unit is a 25 × 25 m spatial block. We ran (i) 100 Monte Carlo iterations with 70/30 block splits stratified by site, (ii) a 1,000-sample block bootstrap of the reported configuration, and (iii) leave-one-site-out and leave-one-city-out experiments. The new subsection "Statistical Evaluation Protocol" (Section III.E), Table X and Figs. X–Z report the results.
>
> Without any tuning, rooftop IoU is 75.33 % (95 % CI 72.76–77.78 %) and PV IoU is 77.48 % (64.61–84.03 %). With calibrated post-processing, rooftop IoU is 82.97 ± 1.12 % across the 100 test splits, and 83.13 ± 5.99 % in leave-one-city-out evaluation. The larger spread for PV panels (IoU 75.64 ± 11.06 %) reflects the small number of PV installations in each test fold, and we now state this explicitly as a limitation. While preparing this analysis we re-derived all metrics from the stored predictions and corrected the aggregate figures in the manuscript (see the revised Section IV.A). *(Adapt this last sentence to the decision in §5.)*

**Comment R2-3** (comparison with hierarchical segmentation, e.g. Prades et al. 2020)

> We thank the reviewer for pointing us to this line of work. We have added a paragraph to the Related Works section that contrasts our approach with bottom-up hierarchical segmentation, including the agglomerative clustering of Prades *et al.* for hyperspectral endmember estimation. That method builds a bottom-up hierarchy over spectral clusters to estimate the number of materials. Ours builds a top-down multi-scale hierarchy over image tiles in which a foundation model proposes class-specific object instances. We also provide two new experiments.
>
> (1) An ablation of the hierarchy levels (Table Y, Fig. W). For rooftops, the full three-level hierarchy (IoU 75.33 %) significantly outperforms every single level (62.99–70.44 %) and every pair of levels (71.53–73.56 %) (Wilcoxon signed-rank on spatial blocks, Holm-corrected p ≤ 5 × 10⁻⁵). For PV panels, the 512 px level alone performs best (86.06 %), which we now discuss as a motivation for class-specific level selection.
>
> (2) A classical hierarchical merging baseline (SLIC superpixels + agglomerative region-adjacency-graph merging) on the same RGB orthomosaics and test blocks (Table Z, Fig. V). Calibrated in the same way, it reaches 68.86 % rooftop IoU and 33.97 % PV IoU, versus 73.59 % and 76.68 % for our method. Even with reference-labelled regions (oracle), its accuracy falls sharply as the hierarchy is cut coarser. A direct quantitative comparison with Prades *et al.* is not possible because their method requires hyperspectral data and produces material clusters rather than rooftop or PV masks. A conceptual comparison table is provided in the Supplementary Material.

---

## 4. File yang dihasilkan & cara reproduksi

### 4.1 Figur (`revisi_r2/figures/`, 300 dpi)
| File | Isi |
|---|---|
| `fig_mc_distribution.png` | Distribusi metrik MC (100 iterasi), Roof vs PV, penanda nilai paper |
| `fig_per_site_metrics.png` | Metrik per site PAPER_CONFIG + 95 % CI bootstrap blok |
| `fig_param_stability.png` | Frekuensi parameter terpilih pada kalibrasi MC |
| `fig_ablation.png` | IoU/F1 tiap kombinasi level + 95 % CI |
| `fig_baseline_visual.png` | Citra, GT, SAM3, SLIC-RAG (JOG_PV, SBY_ROOF) |

### 4.2 Tabel (`revisi_r2/results/`)
| File | Isi |
|---|---|
| `phase0_reproduction.csv` | Agregat semua sumber vs paper (pooled & site-mean), delta pp |
| `phase0_all_sources_per_site.csv` | Metrik per site × kelas untuk semua sumber (A–E) |
| `per_site_metrics.csv` | PAPER_CONFIG per site (6 roof, 3 PV) |
| `phase0_pv_revised_edits.csv` | Poligon PV yang dihapus manual di `_REVISED` |
| `phase0_trace_9208.txt` | Pencarian asal 92.08 / 80.61 |
| `hardware.txt` | GPU, VRAM, CPU, RAM, torch/CUDA |
| `determinism.csv` | Re-run vs `output_inference`, rakit ulang offline |
| `timing.csv` | Waktu per level, per site, per bangunan, VRAM |
| `block_inventory.csv` | Jumlah blok, blok valid, blok kosong yang dibuang |
| `grid_full_data.csv` | Metrik 42 konfigurasi grid pada semua blok valid |
| `mc_iterations.csv` | 100 iterasi × 2 kelas: parameter terpilih + metrik test |
| `mc_selected_params.csv` | Frekuensi parameter terpilih |
| `mc_per_site_paperconfig.csv` | Metrik per site per iterasi (cek fold tanpa GT) |
| `bootstrap.csv`, `bootstrap_per_site.csv` | 1000 resample blok |
| `loso.csv`, `loco.csv` | Leave-one-site/city-out |
| `summary_table.csv`, `summary_table.md` | Tabel ringkasan R2-2 |
| `per_block_metrics.csv` | **Untuk R2-1 (Raffi):** metrik per blok PAPER_CONFIG (site, city, block_id, class, TP, FP, FN, TN, IoU, P, R, F1, valid) |
| `ablation.csv`, `ablation_stats.csv` | Ablation level + uji Wilcoxon/Holm/efek |
| `baseline_hierarchical.csv`, `baseline_slic_rules.txt`, `baseline_slic_timing.csv` | Baseline SLIC-RAG |
| `conceptual_comparison.md` | Tabel konseptual Prades et al. vs SLIC-RAG vs metode kita |

### 4.3 Urutan command (dari root repo, PowerShell)
```powershell
.\venv\Scripts\python.exe revisi_r2\scripts\phase0_reproduce.py        # Fase 0 (+ hardware.txt)
.\venv\Scripts\python.exe revisi_r2\scripts\phase0b_paper_config.py
.\venv\Scripts\python.exe revisi_r2\scripts\phase0c_pv_manual_edits.py
.\venv\Scripts\python.exe revisi_r2\scripts\phase0d_trace_9208.py > revisi_r2\results\phase0_trace_9208.txt
.\venv\Scripts\python.exe revisi_r2\scripts\cache_levels.py            # Fase 1, GPU, ~25 menit, resumable
.\venv\Scripts\python.exe revisi_r2\scripts\verify_reassembly.py
.\venv\Scripts\python.exe revisi_r2\scripts\timing.py
.\venv\Scripts\python.exe revisi_r2\scripts\precompute_blocks.py       # ~4 menit, CPU
.\venv\Scripts\python.exe revisi_r2\scripts\montecarlo.py              # Fase 2
.\venv\Scripts\python.exe revisi_r2\scripts\mc_per_site_check.py
.\venv\Scripts\python.exe revisi_r2\scripts\grid_full_data.py
.\venv\Scripts\python.exe revisi_r2\scripts\ablation.py                # Fase 3A
.\venv\Scripts\python.exe revisi_r2\scripts\baseline_slic.py           # Fase 3B (~6 menit pertama kali)
```
Skrip bisa dijalankan dari root repo (semua path absolut terhadap repo, lewat `common.py`). Hasil yang sudah ada di-skip; hapus file terkait di `cache\`/`results\` untuk menghitung ulang.

Perubahan kode di luar `revisi_r2`: hanya `sam3_advanced/inference.py`, bersifat aditif (`sink=None`, `return_levels=False`; perilaku default tidak berubah; backup di `revisi_r2/backup/inference.py.orig`). Paket baru: `statsmodels`.

---

## 5. Keterbatasan & keputusan yang perlu diambil manusia

**Keputusan (penting, sebelum mengirim revisi):**
1. **Angka PV di paper bergantung pada penghapusan manual poligon FP** (`_REVISED`). Pilihan: (a) ganti dengan angka otomatis (IoU 77.48 / P 79.35 / R 96.98 / F1 87.27) — *rekomendasi*; (b) pertahankan tetapi nyatakan eksplisit bahwa ada koreksi manual pasca-deteksi dan laporkan angka otomatis berdampingan; (c) terapkan kalibrasi τ PV (τ ≈ 0.60 memberi IoU 82.76 / P 87.57 pada seluruh data, `grid_full_data.csv`) dan laporkan dengan protokol calibration/test. Menyimpan angka lama tanpa pengungkapan berisiko karena reviewer kini meminta protokol yang bisa direproduksi.
2. **Klaim "six sites" dan agregat 92.08 / 80.61.** Angka paper berasal dari 3 site PV. Putuskan apakah tabel utama memakai 6 site roof (75.33 IoU) atau 3 site (76.48), dan perbaiki kalimat 92.08 (→ 91.83, atau hapus).
3. **Recall/F1 roof (75.23 / 83.89) tidak konsisten** dengan IoU/P paper. Perlu dikoreksi (78.53 / 86.56 untuk 3 site; 77.75 / 85.79 untuk 6 site).
4. **Apakah konfigurasi terkalibrasi (roof: τ 0.35, filter geometri OFF) diadopsi sebagai hasil utama?** Hasilnya jauh lebih baik (IoU ≈ 83), tetapi mengubah angka utama paper. Alternatif: tetap laporkan konfigurasi asli sebagai hasil utama dan konfigurasi terkalibrasi sebagai analisis tambahan.
5. **Deskripsi NMS di Section III.C** tidak sesuai dengan kode (union + consensus + validasi). Perlu diperbaiki.
6. **Temuan PV-L2.** Untuk PV, hierarki penuh bukan yang terbaik. Putuskan apakah ini ditulis sebagai keterbatasan/arah kerja lanjutan (disarankan), atau pipeline diubah (hanya bisa dilakukan jika paper boleh mengganti hasil).

**Keterbatasan teknis:**
- τ roof terpilih di batas bawah cache (0.35). Nilai lebih rendah butuh inferensi SAM3 ulang dengan ambang koleksi lebih rendah.
- PV hanya 3 site dengan sedikit objek (SBY_PV: 2 panel GT) → CI PV lebar. 2/100 fold test SBY_PV tanpa PV GT.
- Accuracy pada tabel MC/bootstrap dihitung pada blok valid saja (berbeda dari accuracy seluruh tile). IoU/P/R/F1 tidak terpengaruh.
- Per-blok IoU/F1 memakai konvensi blok kosong-kosong = 1. Hasil uji sensitivitas pada blok ber-GT saja ada di `ablation_stats.csv` (`block_set = gt_positive_blocks`). Kesimpulan roof tidak berubah; untuk PV, efek L2/L3 menjadi borderline setelah Holm (p ≈ 0.06–0.08), kecuali L2+L3 (p = 0.004).
- Pada ablation tanpa L1, validasi L3 terhadap L1 tidak bisa dilakukan, sehingga L3 masuk tanpa validasi (didokumentasikan di `hierarchy.merge_levels`).
- Baseline SLIC-RAG: segmen yang melintasi batas blok calibration/test bisa berkontribusi ke training dan evaluasi (kebocoran kecil). Aturan pelabelan sengaja sederhana (decision tree depth 3). Baseline oracle adalah batas atas, bukan metode.
- Perbandingan dengan Prades et al. bersifat konseptual. Metode mereka butuh data hyperspectral dan menghasilkan cluster material, jadi tidak bisa diterapkan langsung pada ortomosaik RGB.
- Waktu diukur pada re-run dengan ambang koleksi 0.35 (sedikit overhead penyimpanan instance). Waktu run asli (manifest) disertakan untuk pembanding.
