# Dyn3D: Monocular Dynamic Scene Reconstruction (Human-Centric)

**Goal:** Given a single handheld RGB video of a person moving/deforming in a scene
(no depth sensor, no multi-camera rig), reconstruct a temporally consistent 3D
representation of the scene that can be rendered from novel viewpoints and novel times.

This is a scoped, buildable version of the general "dynamic non-rigid 3D reconstruction"
problem — full generality (arbitrary topology, arbitrary objects, long video) is still
open research. Scoping to a single articulated human subject and short clips (3-10s)
makes this achievable in weeks, not years, while still touching every hard part of the
underlying problem: pose-from-dynamic-video, non-rigid deformation, long-term
correspondence, and temporal consistency.

---

## 1. Problem Statement

**Input:** Monocular RGB video, 3-10 seconds, handheld or static camera, one primary
moving/deforming subject (human), static or slowly-varying background.

**Output:**
- A canonical 3D representation of the subject + background (3D Gaussian Splats)
- A per-frame deformation field mapping canonical space -> observed space
- Camera poses per frame
- Ability to render the scene from novel camera angles at any timestep in the clip
- Ability to interpolate between observed timesteps (slow-motion / temporal upsampling)

**Non-goals (explicitly out of scope for v1):**
- Multiple independently moving subjects
- Topology changes (removing clothing, cutting objects, object handoff)
- Minute-long video / online streaming reconstruction
- Full scene relighting

---

## 2. Why This Is Hard (and what each module fights)

| Difficulty | Module that addresses it |
|---|---|
| Camera motion is entangled with subject motion | Static-background-masked SfM / pose estimation |
| No multi-view triangulation available | Monocular depth prior (relative, needs scale/shift alignment) |
| Frame-to-frame depth is inconsistent (flicker) | Canonical space + learned deformation field |
| Occlusion / self-occlusion of the subject | Long-term point tracking (not just adjacent-frame optical flow) |
| No ground truth 3D for real handheld video | Self-supervised photometric + flow + tracking losses |
| No agreed evaluation metric | Multi-metric eval: novel-view PSNR/SSIM/LPIPS on held-out frames + reprojection consistency |

---

## 3. Architecture

```
                         ┌─────────────────────┐
  RGB video frames  ───▶ │  Preprocessing        │
                         │  - frame extraction    │
                         │  - human mask (SAM2)   │
                         │  - monocular depth      │
                         │    (Depth Anything V2)  │
                         │  - optical flow (RAFT)  │
                         │  - long-term tracks      │
                         │    (CoTracker3)          │
                         └─────────┬────────────┘
                                   │
                         ┌─────────▼────────────┐
                         │  Camera Pose Estimation│
                         │  - mask out subject     │
                         │  - COLMAP / DROID-SLAM  │
                         │    on background only   │
                         └─────────┬────────────┘
                                   │
                         ┌─────────▼────────────────────┐
                         │  Canonical Representation       │
                         │  - 3D Gaussians (init from       │
                         │    depth-unprojected point cloud) │
                         │  - split: static-bg gaussians +   │
                         │    subject gaussians              │
                         └─────────┬────────────────────┘
                                   │
                         ┌─────────▼────────────────────┐
                         │  Deformation Field (MLP)        │
                         │  - input: (xyz_canonical, t)      │
                         │  - output: (Δxyz, Δrot, Δscale)    │
                         │  - conditioned on per-frame latent │
                         └─────────┬────────────────────┘
                                   │
                         ┌─────────▼────────────────────┐
                         │  Differentiable Rasterizer       │
                         │  (gsplat / diff-gaussian-raster)  │
                         └─────────┬────────────────────┘
                                   │
                         ┌─────────▼────────────────────┐
                         │  Losses                          │
                         │  - photometric (L1 + SSIM)        │
                         │  - depth prior (scale/shift-inv)   │
                         │  - flow-consistency                 │
                         │  - long-term track consistency       │
                         │  - as-rigid-as-possible (ARAP) reg    │
                         │  - opacity / scale regularizers        │
                         └────────────────────────────────┘
```

---

## 4. Tech Stack

- **Language:** Python 3.10+, PyTorch 2.x
- **Rendering:** `gsplat` (fast, well-maintained differentiable 3DGS rasterizer)
- **Depth prior:** Depth Anything V2 (monocular relative depth)
- **Optical flow:** RAFT (torchvision or official repo)
- **Long-term tracking:** CoTracker3 (Meta) — critical, this is what separates this
  project from a naive per-frame dynamic NeRF
- **Segmentation:** SAM2 for subject/background masking
- **Camera pose:** COLMAP (background-only) or DROID-SLAM as a learnable alternative
- **Experiment tracking:** Weights & Biases or TensorBoard
- **Config:** Hydra / YAML

---

## 5. Repository Layout

```
dyn3d/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── data/
│   │   └── dataset.py          # video -> frames, mask, depth, flow caching
│   ├── geometry/
│   │   ├── camera.py           # pose estimation (masked SfM)
│   │   └── depth_flow.py       # depth + flow model wrappers
│   ├── tracking/
│   │   └── point_tracker.py    # CoTracker wrapper, long-term correspondences
│   ├── model/
│   │   ├── deformable_gaussians.py  # canonical gaussians + deformation MLP
│   │   └── losses.py           # all training losses
│   ├── train.py                # main training loop
│   └── eval/
│       └── metrics.py          # PSNR / SSIM / LPIPS / reprojection error
└── scripts/
    ├── preprocess.sh
    └── run_train.sh
```

---

## 6. Datasets

**For development / sanity checks (synthetic, has ground truth):**
- D-NeRF dataset (synthetic dynamic scenes, canonical GT available)
- HyperNeRF dataset (real, multi-camera rig — use as pseudo-GT for eval)

**For the real target use case (monocular, in-the-wild):**
- Record your own clips (phone video of a person moving) — cheapest, most relevant
- Neural3DVideo / DyCheck iPhone dataset — designed exactly for monocular
  dynamic evaluation, includes held-out validation cameras

Start with DyCheck. It exists specifically because monocular dynamic reconstruction
had no good benchmark before it — the authors provide extra validation camera views
withheld from training specifically to test what naive methods get catastrophically
wrong (they overfit to training-camera-trajectory and fail on novel views).

---

## 7. Milestones / Roadmap

### Phase 0 — Environment & Preprocessing (Week 1)
- [ ] Set up gsplat, Depth Anything V2, RAFT, CoTracker3, SAM2
- [ ] Build preprocessing pipeline: video -> frames -> masks -> depth -> flow -> tracks
- [ ] Cache all precomputed signals to disk (this will be your biggest iteration-speed win)

### Phase 1 — Static Baseline (Week 2)
- [ ] Get a *static* 3D Gaussian Splatting pipeline working end-to-end on a single frame
  set (treat it as a normal multi-view/single-view-with-depth-prior reconstruction)
- [ ] Validate the rasterizer, camera pose pipeline, and loss functions in isolation
  before adding any dynamics — this is the single highest-leverage debugging step

### Phase 2 — Add Time (Weeks 3-4)
- [ ] Implement the deformation MLP conditioned on per-frame latent codes
- [ ] Add photometric + depth-prior losses across all frames jointly
- [ ] Get *something* rendering through time, even if blurry/wrong — establishes the loop

### Phase 3 — Fix the Hard Part: Consistency (Weeks 5-6)
- [ ] Integrate long-term point tracks as an explicit supervision signal
  (this is what prevents drift/flicker that plagues naive per-frame approaches)
- [ ] Add ARAP (as-rigid-as-possible) regularization on the deformation field
- [ ] Add flow-consistency loss between rendered and estimated optical flow

### Phase 4 — Evaluation & Ablation (Week 7)
- [ ] Held-out novel-view PSNR/SSIM/LPIPS on DyCheck
- [ ] Ablate: with/without tracking loss, with/without ARAP, with/without depth prior
- [ ] Document failure modes explicitly (fast motion, self-occlusion, loose clothing)

### Phase 5 — Polish (Week 8+)
- [ ] Real-time-ish interactive viewer (gsplat has a viewer you can extend)
- [ ] Write up results, failure case gallery, and a short report/blog post
- [ ] (Stretch) Extend to two interacting subjects, or longer clips with re-tracking

---

## 8. Evaluation Metrics

- **Novel-view synthesis quality:** PSNR, SSIM, LPIPS on held-out camera views (DyCheck
  provides these) — this is the metric that actually matters, since it can't be gamed
  by overfitting to the training camera trajectory
- **Temporal consistency:** flicker metric (frame-to-frame pixel variance in static
  regions), reprojection error of tracked points
- **Geometric plausibility:** no ground-truth 3D exists for real video, so this stays
  qualitative — render from extreme novel angles and visually check for artifacts

---

## 9. Known Hard Failure Modes (be upfront about these)

- Fast motion → optical flow and tracking degrade → deformation field gets noisy
  supervision → blur/ghosting in renders
- Loose or textureless clothing → tracking has few good features to lock onto
- Heavy self-occlusion (arms crossing torso) → canonical space assignment ambiguous
- Long clips → per-frame latent codes stop generalizing, need periodic re-anchoring

Documenting these clearly is not optional — this is genuinely unsolved territory,
and a good project write-up is honest about where the method breaks.

---

## 10. Key References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (2023)
- Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering* (2024)
- Yang et al., *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene
  Reconstruction* (2024)
- Gao et al., *DyCheck: Monocular Dynamic View Synthesis: A Reality Check* (2022) —
  read this one first, it explains exactly why naive approaches fail
- Karaev et al., *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling
  Real Videos* (2024)
- Yang et al., *Depth Anything V2* (2024)

---

## 11. Getting Started

```bash
cd dyn3d
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Preprocess a video into frames/masks/depth/flow/tracks
bash scripts/preprocess.sh /path/to/video.mp4 data/my_clip

# 2. Train
python src/train.py --config configs/default.yaml --data data/my_clip

# 3. Evaluate
python src/eval/metrics.py --run outputs/my_clip --data data/my_clip
```
