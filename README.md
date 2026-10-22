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
node tests/test_core.js
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
- \`tests/test_core.js\` — invariants, pixel roundtrip, learning and animation checks.
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
