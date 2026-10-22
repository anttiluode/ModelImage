# ModelImage — a teachable sieve image assembler

**[Open the live lab once GitHub Pages is enabled](https://anttiluode.github.io/ModelImage/)** · [index.html](index.html) · [core.js](core.js) · [tests](tests/)

> Break an image into fragments. Teach the system complete examples. Watch fragments of a **different** image settle into the arrangement preferred by what it learned.

This is a **working, interpretable assembly prototype**, not yet a generative foundation model. It uses no server, pretrained network, dependency, or API. The browser runs all learning, searching, pixel data, and rendering locally. In particular, it does not use the original positions of test tiles when solving. Their original positions are retained only for evaluation.

## Try it

1. Open \`index.html\` through a local HTTP server or GitHub Pages. From the repository directory: \`python -m http.server 8000\` → http://localhost:8000. The ES module import does not work reliably with \`file://\`.
2. Choose **Portrait**, **Garden**, **City**, or **Abstract field**; press **Teach 20**. The two left images show the most recently taught image and the learned, deliberately low-detail spatial memory.
3. Press **New unseen example**, then **Assemble**. All 36 tiles are available but shuffled. The computer chooses an arrangement from learned location and neighbor expectations plus nonlearned edge continuity. Watch it animate into place.
4. Set **Memory pull = 0** and assemble again, or **Erase learning**. The same task is now an ordinary raw-pixel edge-matching jigsaw, providing a key control.
5. Click **Compare learned vs untrained (3 tests)** for fresh held-out examples. It reports both exact tile placement and **pixel reconstruction error**, because swapping homogeneous sky or background tiles can be visually harmless while counting as incorrect.
6. **Teach my photo** uses one of your own images as a lesson. Repeat with *different photographs of similar composition*. **Test my photo** supplies a held-out image; uploading the same photograph as teaching data is not a legitimate generalization test. Uploaded photos are center-cropped to 96×96 and never transmitted.

The model can save numerical learned parameters in your local browser storage. Uploaded image pixels are **not** stored there. The browser only uses local computation.

## Architecture, in transparent equations

The image is resized to 96×96 and cut into a **6×6 grid of 16×16 tiles**.

### 1. Read: a simple sieve descriptor

For each tile at spatial site \(p\), the feature \(f_p\) contains:

- 3 average RGB values (coarse/color core)
- 9 means from a 3×3 luminance grid (internal low-frequency structure)
- 2 spatial luminance slopes
- 1 luminance standard deviation

This is a **small, explicitly defined sieve-inspired descriptor**. It is *not* a claim to be the complete transform used in earlier Sihti projects, and it is not a learned deep encoder.

### 2. Learn: local online prototypes and relations

Each complete teaching image updates a prototype \(\mu_p\) at every spatial site and an expected feature difference \(\delta_{p,q}\) for each of the 60 directed neighboring pairs (right and down):

\[
\mu_p^{(n)}=\mu_p^{(n-1)}+\frac{f_p-\mu_p^{(n-1)}}{n}
\]

\[
\delta_{p,q}^{(n)}=\delta_{p,q}^{(n-1)}+
\frac{(f_q-f_p)-\delta_{p,q}^{(n-1)}}{n}.
\]

All 36 locations and all 60 links are known in advance; the model learns their **values**, not which arbitrary object parts should connect. It learns neither cross-object relationships nor 3-piece parity at this stage.

### 3. Solve: competing full-image placements

The search receives only the shuffled tile pixels, the learned model, and its fixed hyperparameters. For an assignment \(\pi\) mapping board sites to tiles, it minimizes:

\[
E(\pi)=c_n\left[\alpha\,\frac{1}{36}\sum_p\Vert f_{\pi(p)}-\mu_p\Vert^2+
\beta\,\frac{1}{60}\sum_{p\rightarrow q}\Vert(f_{\pi(q)}-f_{\pi(p)})-\delta_{p,q}\Vert^2\right]
+\gamma\,E_{\mathrm{seam}}(\pi).
\]

Here \(c_n=\min(1,n/8)\) grows with the number of lessons; the distances in the implementation use per-feature normalization; \(E_{\mathrm{seam}}\) is **not learned**—it penalizes discontinuity between immediately adjacent tile-edge pixels. \(\alpha,\beta,\gamma\) are user-controlled weights (the relation weight is tied to memory pull in this first GUI).

A Hungarian assignment gives a globally one-to-one starting point for the learned positional term; subsequent simulated-annealing swap proposals minimize the full energy with local links and seams. Multiple restarts are used. To create a readable animation, the interface shows swaps from the initial scramble into the **already computed** final board. The animation is *not* a simulation of dendritic firing or physically moving neural activity.

### 4. Inspect and test

The app shows exact positions, true-image pixel error, learned parameters rendered as a blurry image, total energy, and a comparison against an untrained edge-only solver. In the code the only accesses to the original tile index field \`truth\` are the evaluation function \`accuracy\` and optional validation; \`energy\`, \`hungarian\`, and \`solve\` don't look at it.

## What the first checks tell us

An initial smoke test, with 20 lessons from each synthetic world and fresh unseen examples, found useful assembly for portraits, cityscapes and abstract patterns. For example, two held-out portraits gave **24/36 and 23/36** correct, versus **0/36 and 0/36** for the edge-only solver in those runs. The two city samples gave **23/36 and 21/36**, versus **1/36 and 1/36**. These are small exploratory sanity checks, **not general image-quality benchmarks**, and many correct-looking flat-colored regions can contain ambiguously permuted tiles.

Garden images were inconsistent: two examples yielded **22/36 and 2/36** correct with training. This is a meaningful failure, not an outlier to erase. A single mean prototype per fixed location cannot reliably model shifting flowers or multimodal arrangements; simulated annealing also has local minima.

To reproduce basic checks on Node 22 or newer:

\`\`\`bash
node tests/test_core.mjs
node tests/test_learning.mjs
\`\`\`

Browser GUI was also exercised through a mocked DOM for 20 lessons, assembly, stepping, forgetting, and save/restore; this is **not a substitute for human visual inspection across browsers and devices**.

## What a genuine Sieve Image Model would need next

This first version is **structured jigsaw assembly**. For the larger ambition—learning a visual world and *generating* the unseen fragments—we still need:

1. **Movable, reusable parts**, discovered without assigning every semantic role to a fixed grid cell.
2. **Nonlinear and conditional relationships** (e.g., eye–eye–mouth patterns that have no informative pairwise marginals), not just averages and pairwise differences.
3. **Coarse-to-fine synthesis** that predicts missing pixels or new tiles from learned structure, rather than selecting from tiles already provided.
4. **Controlled held-out tests**: untrained seam matching, per-slot mean, random features, equal-parameter MLP/CNN, fixed-vs-learned relations, shuffled teaching labels, and unseen camera movements.
5. **Resource budgeting and competition** so local units have to decide which part configurations deserve to survive.

The natural next model is a hierarchy of latent visual atoms that proposes configurations, relaxes to a consistent arrangement, and learns from the mismatch against real observations. The useful aspiration is **to see the internal geometry form and see how that geometry changes perception**—without mistaking a nice visualization for a demonstrated learning mechanism.

## Repository

- \`core.js\` — pure deterministic image generator, descriptor, online learning, assignment energy and search.
- \`index.html\` — interactive, self-contained GUI; does not send images to a server.
- \`tests/test_core.mjs\` — invariants, pixel roundtrip, learning and animation checks.
- \`tests/test_learning.mjs\` — held-out ablations and nonleakage-related checks.

MIT license. This work is a continuation of the [Sihti image lineage](https://github.com/anttiluode/SihtiImageModel5) and the higher-order relationship experiments in [ClaudeRepo1](https://github.com/anttiluode/ClaudeRepo1). Those related projects are not runtime dependencies.


---

## V2 — The world learns its missing pieces

**[Open the generative completion lab](world-model.html)** (direct URL once GitHub Pages is enabled: https://anttiluode.github.io/ModelImage/world-model.html).

The new page complements the original jigsaw assembler. **It generates genuinely missing pixels** from visual atoms learned in complete scenes; the tile to be predicted is absent from the inference input.

### What is actually learned

- A position-independent codebook of complete 16×16 RGB patch prototypes. A tile becomes a 4×4×RGB feature vector for matching but stores average **full-resolution patch pixels** for regeneration. When similar parts recur at different image positions, they refer to the same atom.
- All one-neighbor and two-neighbor conditional co-occurrence tables across eight possible adjacent offsets, updated directly from complete observations. None of the candidate patterns is predesignated as a "face" or "eye–eye–mouth"; the learner has to select useful relations by observed predictive lift and repeated support.
- Confidence and provenance for a reconstructed hidden tile: which atoms were observed, their positions, which candidate relation won and how many examples supported it.

Training is **unsupervised codebook adaptation and local associative counting**; it is **not** end-to-end backpropagation, biological Hebbian synapses, or automatic discovery of the image's entire causal world. The synthetic generator *does* deliberately put a nonlinear eye/eye/mouth dependency into its own images. The learner gets raw RGB pixels only, never the rule or a label.

### The decisive experiment

The data generator creates scenes from a small alphabet of nine visual motifs: background, two eye appearances, two mouth appearances, a cheek and three distractors. It randomly translates the four-tile face group around a 6×6 scene. Mouth type is determined either by equality or inequality of the two eyes, depending on the world. All single-piece and pairwise statistics are uninformative about which world the image came from; the three-part relation separates the two.

The learners see complete images with distractors, discover a codebook, and assess all neighboring one- and two-context associations. On a **new** image, the target mouth patch is completely masked; the original hidden pixels are not available to prediction. The hidden patch is reconstructed from a learned prototype, not moved from the existing input.

**Exploratory reproducible benchmark (360 training scenes per world, 100 independent held-out examples):**

| Gate | Two-context model | One-context control |
|---|---:|---:|
| World A, concealed mouth | **100 / 100** | 50 / 100 |
| World B, concealed mouth | **100 / 100** | 50 / 100 |
| Copy-world positive control | **100 / 100** | **100 / 100** |
| World A, contiguous half-image occlusion, mean hidden-pixel MSE | 0.0634 | 0.0728 |

A particularly useful check: presenting the **same** visible A-world scene to the learners trained in A and B produced different hidden-patch images in all 100 evaluated examples. The A-trained model agreed with the A-world target; the B model generally predicted the opposite history's missing piece. Identical present observations can produce different reconstructions after different experiences.

**A critical caveat:** at 80 training scenes, one run had only 35/80 correct for the two-context method, versus 39/80 for the one-context method. The useful relation emerges with sufficient repeated experience; this is not magic instantaneous discovery or universally monotonic progress. The page includes a held-out 80-scene gate that is run on the exact models trained in your browser.

**Shuffled-experience control:** training on the same patch motifs after randomizing their positions preserved the nine-atom dictionary, but raised held-out hidden-mouth pixel MSE from approximately **0.000012 to 0.0655**. The relational geometry, rather than just the motif vocabulary, is responsible for the result.

**Nonleakage check:** swapping the target region with a blank patch before calling the inference engine produces the same predictions as giving it an intact image with those target coordinates explicitly masked. We also keep the ground truth only for evaluation and preserve all unmasked original pixels in the completed image. This is covered by the new Node test.

### Run and inspect

Start a local HTTP server: \`python -m http.server 8000\` and visit \`http://localhost:8000/world-model.html\`. Press **Teach both ×360**; inspect the learned nine or so visual atoms and the strongest selected source fragments; press **New unseen scene** to reveal the conflicting histories, or choose **Half the image** to explore more difficult and frequently incomplete reconstructions. **Erase experience** removes the learned dictionary and relation tables.

To reproduce the Node tests: \`node tests/test_world.mjs\`. CI runs these alongside the original V1 tests.

### Why this is not yet a general-purpose image model

All nine source motifs recur with near-identical shape, translation occurs on a 16-pixel grid, and trained atoms are chosen from a small synthetic family. This setup makes it possible to test a higher-order relation without supplying that relation to the learner. It **does not demonstrate natural-image generalization**, full-scale object discovery, perspective invariance, a learned scale hierarchy, or generating truly unseen high-frequency texture. Capacity (24 learned atoms) and the candidate neighborhood geometry are engineered; combinatorial search beyond this small window remains unsolved.

A next-stage real-image study should teach RGB patches at several scales, learn a resource-bounded set of relations with attention to context and viewpoint, use external held-out photographs, and compare to convolutional and patch-dictionary baselines at a matched memory budget. This makes the leap from an inspectable synthetic world to the proposed readable **sieve image model** measurable rather than rhetorical.


### Running locally: JavaScript module MIME types

ModelImage uses `core.js` and `world-core.js` because Python's standard `http.server` on some machines serves `.mjs` with the strict-browser-blocked content type `text/plain`. The same `.js` modules run in Node tests through the repository's `package.json` (`"type": "module"`), so the browser and CI exercise the **same source**.

From an updated clone:

```bash
git pull
py -m http.server 8000
```

Then browse http://localhost:8000/world-model.html or http://localhost:8000/ and reload with **Ctrl+F5**. If port 8000 is already running, the existing server can usually serve the updated files; restart it only if necessary. The extension-originated `contentscript.js` listener warnings are unrelated to ModelImage. If a MIME failure persists, check the network response for `world-core.js`: it should respond with JavaScript content and a JavaScript MIME type, not `text/plain` or an HTML 404 page.

---

## V3 — Hierarchical, competitive, deformable sieve generation

**[Open the V3 lab](v3.html)** · **[Read its engine](v3-core.js)** · **[Core tests](tests/test_v3.mjs)** · **[Real-photo transfer gate](tests/test_v3_photos.mjs)**.

V1 rearranged given image tiles. V2 generated missing synthetic pixels by learning a discrete visual alphabet and three-part relations. V3 tests whether we can scale those two lessons toward real images *while preserving complete observability*. It has four independently inspectable components.

### 1. Three progressive frequency/residue populations

Each 96×96 image is divided into 12×12 position-independent 8×8 patches. The patch transform is additive:

- **Coarse:** 2×2 RGB low-resolution representation (12 floats)
- **Middle:** 4×4 RGB minus bilinear expansion of coarse (48 signed floats)
- **Fine:** 8×8 RGB minus bilinear expansion of the 4×4 signal (192 signed floats)

The explicit transform and its inverse are exact to floating-point precision before learned quantization. \`tests/test_v3.mjs\` verifies maximal error below 2×10⁻⁷. The learned dictionaries, however, **are lossy**. Don't confuse an invertible *sieve* with an invertible *trained model*.

Each level owns a separate visual dictionary with capacity 48/64/80 atoms by default. At each observation the winner minimizes feature MSE. This is a **competitive Hebbian-like** update with bounded moving prototypes: \`w <- w + (rate/sqrt(1+hits/8)) * (x-w)\`, or a new atom when novelty exceeds its threshold and capacity allows. It is a centroid-learning mechanism, **not a demonstrated biological Hebbian synapse** or end-to-end task-gradient learner.

### 2. Learned spatial groups, not fixed coordinates

The coarse level observes 12 possible support displacements (eight local directions plus four distance-2 cardinal directions) around each center. A sparse online ledger accumulates observed **single-support** and **two-support** conditional target atom counts. Only eight randomly rotating two-support candidates per target site are recorded per experience, limiting growth. During reconstruction, rival hypotheses are scored by predictive lift against the target atom's global frequency, discounted by amount of experience. Candidate geometry is shared across the image; a pattern can move from upper left to lower right.

The \`Local group flexibility\` slider allows a learned support offset to be matched with a **one-cell shift**, penalized by 0.22 score units per displaced cell. This is a real but restricted form of deformability—not optical flow, learned scale invariance, or unconstrained free-form object shape.

A dedicated synthetic parity-with-shift gate checks the mechanism: a two-input relation learned at offsets −1 and +1 is later presented at −2 and +1. Rigid relations infer **2/4** correct combinations, while offset-flexible learned groups infer **4/4**. The rule is *not* supplied as a hand-coded product feature; it is acquired as a supported conditional relation.

### 3. Generating hidden pixels, and checking trust

All hidden 8×8 patches are omitted from \`recognize\` and its codebook lookup. The coarse atom for each missing patch is proposed from learned groups. The middle atom is chosen from its learned coarse→middle condition; the fine atom from its learned coarse+middle→fine condition. The inverse sieve renders newly generated pixels. **Original visible pixels are copied through unchanged**. Both intact-target-versus-masked-target controls and a visible-pixel identity check run in CI.

We also implemented **Model 5-style trust gating**. On each inference, the model briefly hides several *still-visible* patches near the hole. It compares its own guesses for those probes against plain nearby-color interpolation. When learned completion predicts visible evidence poorly, the system does not trust its memory and instead uses the simpler interpolation. This is an empirical self-check of local predictive utility, **not proof that all future mistakes will be detected**.

The GUI deliberately shows **raw model output, coarse-only output, one-support baseline, simple interpolation, and the trust-checked final**, side by side. The confidence evidence and selected source-fragment links are inspectable. The trust-checked output is not a separate trained model.

### 4. Real-photo transfer: a negative gate we keep

Reproduce with:

\`\`\`bash
python -m pip install scikit-image pillow numpy
python benchmarks/photo_dataset.py
node tests/test_v3.mjs
node tests/test_v3_photos.mjs
\`\`\`

The benchmark loads **eight training photographs** from the scikit-image data gallery (astronaut, camera, coins, brick, grass, moon, immunohistochemistry, retina) and creates 96 deterministic, varied crops. It uses **four entirely different test photographs** (coffee, Chelsea cat, rocket, Hubble deep field), 12 crops and 24 masked-region tests. There are **no common original photographs** across train and test. We do not commit or redistribute their image files; the test script loads them from the installed public gallery.

The first completed gate measured hidden-pixel mean squared error (smaller is better):

| Method | MSE |
|---|---:|
| Learned two-support groups, all three levels | 0.03756 |
| Learned one-support associations, all levels | 0.03222 |
| Learned two-support groups, coarse only | 0.03391 |
| Simple nearby-color interpolation, **no learned model** | **0.00916** |
| Untrained prototype model | 0.09547 |
| Learned model **with self-checked trust** | **0.00916** |

**Result:** the new representation does **not** yet generalize better than plain interpolation on unseen real photographs. The two-support model was actually worse than the one-support model. In all 24 tests, the trust check correctly withheld the learned proposal and used the simple baseline (mean trust 0). This is encouraging for *rejecting unsupported memory* but is **not** improved image synthesis. On other scenes the same gate could still be wrong.

The real-world wall is now measured: the model can learn a patch alphabet and repeated spatial associations yet still lack the capacity to build useful object-conditioned predictions from unfamiliar natural-image structure.

### GUI

Run a local server with \`py -m http.server 8000\` (Windows) or \`python -m http.server 8000\`. Open **http://localhost:8000/v3.html**. Both JavaScript files use \`.js\` imports and work with the stock Python server on Windows.

- **Teach 300:** demonstrate a single learnable synthetic world. Choose A or B as the training history; deliberately test the other world to see the learned model's limitation.
- **Choose training photos:** load several photographs. The browser makes three deterministic crops from each; their bytes stay in the browser.
- **Choose held-out test photos:** select separate, genuinely unseen files. Evaluate held-out masked-patch reconstruction against untrained interpolation.
- **Explore:** switch mask size, one/two supporting fragments, one-cell deformability, and fine residue strength. Click a missing patch in the sensor view to see which visible parts supported it.
- **Erase experience:** discard every learned atom and relationship, leaving only the built-in inference procedure.

The GUI does not upload photos, does not secretly train on held-out test images, and does not invent accuracy statistics. Its visual source links report actual chosen support positions.

### What must happen next

The evidence suggests several specific next experiments, not a claim of a finished image foundation model: learn conditional residues from *jointly changing views*, not just per-codeword frequency; compress candidate relations by predictive utility under a fixed parameter budget; evaluate real-object viewpoint and scale changes; and compare with matched lightweight convolutional/diffusion baselines. A system that learns reusable higher-level parts from natural images without hand-labels remains the open problem.
