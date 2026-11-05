// Meeting deck for 5 November 2026: progress since 22 October on the three parts of the plan
// (Arabic, every other language, Nadir's MLP-direction ideas). Final results and setups only; every number
// comes from the results log in replication/README.md.
// Run:  NODE_PATH=/opt/homebrew/lib/node_modules node slides/build_slides_2026-11-05.js

const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = path.join(__dirname, "refusal_neurons_2026-11-05.pptx");

const INK = "1F2328", MUTED = "5F6B7A", LINE = "D8DEE6", SOFT = "F4F6F9", WHITE = "FFFFFF";
const TEAL = "0F766E", TEAL_SOFT = "E6F2F1", TEAL_MID = "B7DCD8", SLATE = "475569", AMBER = "B45309",
  AMBER_SOFT = "FBF1E6", FIRE = "D9480F", FIRE_SOFT = "FCE3D6", GREY_SOFT = "EEF0F3";
const FONT = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Refusal neurons: progress, 5 November 2026";
pres.author = "Faaiz Joad";
pres.company = "Qatar Computing Research Institute, HBKU";

const SW = 13.333, ML = 0.6, CW = SW - 2 * ML;
let slideNo = 0;

const TAGS = {
  recap: { text: "WHERE WE ARE", color: MUTED },
  p1: { text: "PART 1: THE APPLE PAPER IN ARABIC", color: AMBER },
  en: { text: "PART 1: ENGLISH GAPS", color: SLATE },
  p2: { text: "PART 2: EVERY OTHER LANGUAGE", color: TEAL },
  p3: { text: "PART 3: THE REFUSAL DIRECTION INSIDE THE MLP", color: SLATE },
  next: { text: "NEXT", color: MUTED },
};

function newSlide(title, tag, notes) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideNo += 1;
  if (tag) {
    const t = TAGS[tag];
    s.addText(t.text, { x: ML, y: 0.32, w: 10, h: 0.3, fontFace: FONT, fontSize: 11, bold: true, color: t.color,
      charSpacing: 1.5, margin: 0 });
  }
  s.addText(title, { x: ML, y: 0.62, w: CW, h: 0.75, fontFace: FONT, fontSize: 27, bold: true, color: INK,
    margin: 0, valign: "middle" });
  s.addText(String(slideNo), { x: SW - ML - 0.6, y: 7.0, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10,
    color: MUTED, align: "right", margin: 0 });
  if (notes) s.addNotes(notes);
  return s;
}

function result(s, runs, opts = {}) {
  const y = opts.y || 6.15, color = opts.color || TEAL, bg = opts.bg || TEAL_SOFT, h = opts.h || 0.7;
  s.addShape(pres.shapes.RECTANGLE, { x: ML, y, w: CW, h, fill: { color: bg }, line: { color: bg } });
  s.addShape(pres.shapes.RECTANGLE, { x: ML, y, w: 0.08, h, fill: { color }, line: { color } });
  s.addText(runs, { x: ML + 0.3, y, w: CW - 0.4, h, fontFace: FONT, fontSize: opts.size || 16,
    color: INK, valign: "middle", margin: 0 });
}

const b = (text) => ({ text, options: { bold: true } });
const t = (text) => ({ text });

function bullets(s, items, x, y, w, h, size = 16) {
  s.addText(items.map((it) => ({ text: it, options: { bullet: { indent: 18 }, breakLine: true } })),
    { x, y, w, h, fontFace: FONT, fontSize: size, color: INK, valign: "top", margin: 0, paraSpaceAfter: 7 });
}

function label(s, text, x, y, w, color = MUTED, size = 12) {
  s.addText(text, { x, y, w, h: 0.3, fontFace: FONT, fontSize: size, bold: true, color, charSpacing: 1, margin: 0 });
}

function table(s, rows, x, y, w, colW, opts = {}) {
  const size = opts.size || 14;
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    cell.options = Object.assign({ fontFace: FONT, fontSize: size, color: INK, valign: "middle",
      align: j === 0 ? "left" : "center", fill: { color: i === 0 ? SOFT : WHITE }, bold: i === 0 }, cell.options || {});
    return cell;
  }));
  s.addTable(body, { x, y, w, colW, border: { type: "solid", pt: 0.75, color: LINE }, rowH: opts.rowH || 0.42,
    margin: [0.04, 0.08, 0.04, 0.08] });
}
const hl = (text) => ({ text, options: { bold: true, color: TEAL } });
const warn = (text) => ({ text, options: { bold: true, color: FIRE } });

// ------------------------------------------------------------------ 1. title
{
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: WHITE };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.25, h: 7.5, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("One refusal neuron, across languages", { x: 1.0, y: 2.2, w: 11, h: 1.0, fontFace: FONT, fontSize: 40,
    bold: true, color: INK, margin: 0 });
  s.addText("Progress since 22 October: Arabic, 22 more language sets, and the refusal direction inside the MLP",
    { x: 1.0, y: 3.25, w: 11, h: 0.8, fontFace: FONT, fontSize: 20, color: MUTED, margin: 0 });
  s.addText("Faaiz Joad  ·  QCRI, HBKU  ·  5 November 2026", { x: 1.0, y: 4.6, w: 11, h: 0.4, fontFace: FONT,
    fontSize: 16, color: INK, margin: 0 });
  s.addNotes("Update for the 5 November meeting. Three parts: the Apple paper in Arabic, every other language, and Nadir's MLP-direction questions.");
}

// ------------------------------------------------------------------ 2. recap and plan
{
  const s = newSlide("Where we were, and the three parts we agreed on", "recap",
    "On 22 October we had replicated Apple's English result on three models and seen the same neuron fire on Arabic. We agreed on three parts. Current priority: Part 2, then Part 3, then the rest of Part 1.");
  s.addShape(pres.shapes.RECTANGLE, { x: ML, y: 1.6, w: CW, h: 1.0, fill: { color: SOFT }, line: { color: SOFT } });
  s.addText([b("22 October: "), t("Apple's single refusal neuron replicated in English on Qwen3-8B, Llama-3.1-8B and Qwen3-1.7B. Our search finds their exact neurons; pinning one jailbreaks the model (90–97%). First sign that the same neuron fires on Arabic.")],
    { x: ML + 0.25, y: 1.6, w: CW - 0.5, h: 1.0, fontFace: FONT, fontSize: 15, color: INK, valign: "middle", margin: 0 });
  const cols = [
    { h: "Part 1: the Apple paper in Arabic", c: AMBER, body: ["Every experiment of the paper, in Arabic, on two datasets (PolyRefuse, MultiJail)", "Plus the English gaps"], st: "Mostly done" },
    { h: "Part 2: every other language", c: TEAL, body: ["PolyRefuse: 14 languages", "MultiJail: 10 human-translated languages", "One neuron for all languages, per family, or per language?"], st: "Search done · attacks running" },
    { h: "Part 3: Nadir's MLP questions", c: SLATE, body: ["Arditi's direction built inside the MLP", "How much of it is the neuron?", "Does it work only through the neuron?", "Reverse route: direction first, then its top neuron"], st: "Done in English · Arabic partly" },
  ];
  const w = (CW - 0.5) / 3;
  cols.forEach((c, i) => {
    const x = ML + i * (w + 0.25);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.95, w, h: 0.08, fill: { color: c.c }, line: { color: c.c } });
    s.addText(c.h, { x, y: 3.1, w, h: 0.5, fontFace: FONT, fontSize: 17, bold: true, color: INK, margin: 0 });
    bullets(s, c.body, x, 3.7, w, 2.0, 14);
    s.addText(c.st, { x, y: 5.75, w, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: c.c, margin: 0 });
  });
}

// ------------------------------------------------------------------ 3. setup
{
  const s = newSlide("Setup used throughout", "recap",
    "Same three models and Apple's three neurons everywhere. 'Pin' means holding the neuron at a fixed value at every token while the model answers. Arabic and other languages are judged in the original language by Qwen3Guard, and by Llama-Guard on an NLLB English translation. Apple's main criterion (App. K.1): an attack counts if the answer is not a refusal and is coherent; we ask K.1's questions of gemma-2-9b-it.");
  table(s, [
    ["Model", "Apple's neuron", "Pin used"],
    ["Qwen3-8B", "layer 14, neuron 7924", "+20"],
    ["Llama-3.1-8B-Instruct", "layer 11, neuron 4258", "−4"],
    ["Qwen3-1.7B", "layer 13, neuron 3270", "+40"],
  ], ML, 1.65, 6.0, [2.4, 2.4, 1.2], { size: 14 });
  label(s, "DATA", 7.0, 1.65, 5.5, TEAL);
  bullets(s, [
    "English: JailbreakBench (100) and HarmBench (191)",
    "PolyRefuse: 14 languages, the same harmful prompts translated",
    "MultiJail: 10 languages, human-translated, more borderline prompts",
    "Per language: search on 128 harmful / 128 harmless, test on held-out prompts",
  ], 7.0, 2.0, 5.7, 2.3, 14);
  label(s, "JUDGES", ML, 3.75, 6, TEAL);
  bullets(s, [
    "English: Llama-Guard-3",
    "Other languages: Qwen3Guard (reads the language) and Llama-Guard on an NLLB-200 translation",
    "Apple's main criterion (App. K.1): attack succeeds if the answer is not a refusal and is coherent; judged by gemma-2-9b-it",
  ], ML, 4.1, 6.0, 2.0, 14);
  label(s, "INTERVENTIONS", 7.0, 4.45, 5.5, TEAL);
  bullets(s, [
    "Constant pin: the neuron held at the pin at every token",
    "Anchor k=1, 2: pinned only near the start of the answer (Apple's gentler version)",
    "Arditi baseline: refusal direction removed from the residual stream",
  ], 7.0, 4.8, 5.7, 1.6, 14);
}

// ------------------------------------------------------------------ 4. Arabic jailbreak
{
  const s = newSlide("In Arabic, the same neuron is found, and pinning it jailbreaks", "p1",
    "Experiment 1: Apple's search run from scratch on Arabic prompts recovers Apple's neuron on Qwen3-8B (L14:F7924 at +20) and Llama (L11:F4258 at −4); on Qwen3-1.7B it is a near-tie with L10:F244. Experiment 2: attack on 100 held-out Arabic prompts per dataset, 512 tokens. K.1-style judge on the Arabic text. MultiJail is lower because many of its prompts are borderline.");
  bullets(s, [
    "Apple's search run from scratch on Arabic prompts finds Apple's English neuron again (Qwen3-8B, Llama; near-tie on Qwen3-1.7B)",
    "The neuron fires on Arabic harmful prompts at 60–80% of its English strength, at the same template token",
  ], ML, 1.55, CW, 1.0, 15);
  label(s, "SUCCESSFUL ATTACKS, ARABIC, 100 HELD-OUT PROMPTS (APPLE'S K.1 CRITERION)", ML, 2.65, CW);
  table(s, [
    ["Model", "PolyRefuse: no pin → pinned", "MultiJail: no pin → pinned", "Arditi, Arabic direction (PolyRefuse)"],
    ["Qwen3-8B", hl("15% → 97%"), "15% → 82%", "97%"],
    ["Llama-3.1-8B", hl("22% → 96%"), "30% → 91%", "99%"],
    ["Qwen3-1.7B", hl("32% → 99%"), "39% → 92%", "100%"],
  ], ML, 3.0, CW, [2.4, 3.2, 3.0, 3.53], { size: 15, rowH: 0.48 });
  result(s, [b("Arabic behaves like English. "), t("One neuron switches refusal off as well as Arditi's whole direction does, and every pinned answer stays coherent.")], { y: 5.4 });
}

// ------------------------------------------------------------------ 5. Arabic cost + detection
{
  const s = newSlide("Arabic: small cost to ability, and the neuron is a good detector", "p1",
    "Exp 3: Arabic MMLU (OpenAI's MMMLU, the same 1,000 questions as our English MMLU set), answer-letter accuracy. Anchoring costs about 1 point. Exp 7: AUROC of the neuron's activation as a harmful-prompt detector, vs Llama-Guard-3. Exp 8: Apple's neuron is the 1st or 2nd most aligned neuron with Arditi's Arabic direction.");
  label(s, "ARABIC MMLU ACCURACY (1,000 QUESTIONS)", ML, 1.55, 6.3);
  table(s, [
    ["Model", "No pin", "Constant", "Anchor k=1", "Arditi"],
    ["Qwen3-8B", "56.4", "46.9", hl("55.7"), "55.3"],
    ["Llama-3.1-8B", "45.3", "40.5", hl("45.3"), "44.0"],
    ["Qwen3-1.7B", "39.5", "31.7", hl("38.4"), "36.4"],
  ], ML, 1.9, 6.3, [1.9, 1.05, 1.1, 1.2, 1.05], { size: 14 });
  label(s, "DETECTING HARMFUL PROMPTS (AUROC)", 7.2, 1.55, 5.5);
  table(s, [
    ["Detector", "Arabic", "English"],
    ["The neuron (3 models)", hl("0.95–0.995"), "0.98–1.00"],
    ["Llama-Guard-3", "0.995", "1.00"],
  ], 7.2, 1.9, 5.5, [2.7, 1.4, 1.4], { size: 14 });
  bullets(s, [
    "Anchored pin (Apple's gentler version) costs about 1 point; a constant pin costs 5–9",
    "One number from one neuron detects harmful Arabic prompts almost as well as a dedicated 8B safety model",
    "Apple's neuron is the 1st or 2nd most aligned neuron with Arditi's Arabic refusal direction",
  ], ML, 4.0, CW, 1.8, 15);
  result(s, [b("Every Arabic result so far matches the English paper. "), t("Remaining: what the neuron responds to (needs a human reader).")], { y: 6.15 });
}

// ------------------------------------------------------------------ 6. English gaps
{
  const s = newSlide("English gaps filled: the neuron beats the residual stream", "en",
    "Section 3.3: the same search over the residual stream. Under the K.1 criterion the best single residual dimension reaches 32% on Llama and 16% on Qwen3-8B; Apple reports 29–45% and 39%. Arditi's capability cost: MMLU 1,000 questions, GSM8K.");
  label(s, "ATTACK SUCCESS OF THE BEST SINGLE DIMENSION (K.1 CRITERION)", ML, 1.55, 6.2);
  table(s, [
    ["Model", "MLP neuron", "Residual stream"],
    ["Llama-3.1-8B", hl("97%"), "32%"],
    ["Qwen3-8B", hl("96%"), "16%"],
  ], ML, 1.9, 6.0, [2.2, 1.9, 1.9], { size: 15, rowH: 0.48 });
  s.addText("Apple reports 29–45% (Llama) and 39% (Qwen3-8B) for the residual stream.", { x: ML, y: 3.45, w: 6, h: 0.4,
    fontFace: FONT, fontSize: 13, color: MUTED, margin: 0 });
  label(s, "ARDITI'S ABLATION: COST TO ABILITY (ENGLISH)", 7.0, 1.55, 5.7);
  table(s, [
    ["Model", "MMLU", "GSM8K"],
    ["Qwen3-8B", "67.5 → 66.6", "58.0 → 57.2"],
    ["Llama-3.1-8B", "63.8 → 63.5", "72.8 → 74.4"],
    ["Qwen3-1.7B", "53.7 → 51.7", "58.4 → 57.6"],
  ], 7.0, 1.9, 5.7, [2.1, 1.8, 1.8], { size: 14 });
  result(s, [b("Section 3.3 confirmed: "), t("the refusal switch lives in one MLP neuron, not in any single residual-stream dimension. Arditi's ablation costs 1–2 points.")], { y: 4.6 });
}

// ------------------------------------------------------------------ 7. Part 2 rank grid
const LANGS = ["en", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "ru", "th", "zh", "yo",
  "en", "zh", "it", "vi", "ko", "th", "bn", "jv", "sw"];
const RANKS = {
  "Llama-3.1-8B": [3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 13, 4, 1, 2, 1, 1, 1, 1, 4, 1],
  "Qwen3-8B": [6, 1, 4, 1, 5, 18, 1, 1, 11, 15, 3, 7, 4423, 2, 1, 2, 1, 1, 1, 1, 1, 5],
  "Qwen3-1.7B": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 175, 1, 3, 3, 6, 4, 4, 3, 1, 354],
};
const FAIL = { "Llama-3.1-8B": [12], "Qwen3-8B": [12, 21], "Qwen3-1.7B": [12, 21] };
{
  const s = newSlide("Every language: the same neuron comes out on top", "p2",
    "For each model and each of 22 language sets (Arabic is Part 1), we first collect the model's own refusal openings in that language, then run Apple's search on 128 harmful and 128 harmless prompts in that language. Each cell is where Apple's English neuron ranks among all MLP neurons (~110k to ~300k). On Qwen3-1.7B the top two are always the same pair: L10:F244 and Apple's L13:F3270. Orange: the neuron no longer separates harmful from harmless (AUROC 0.34–0.81).");
  const x0 = ML + 1.9, y0 = 1.95, cw = (CW - 1.9) / 22, rh = 0.62;
  s.addText("PolyRefuse", { x: x0, y: 1.5, w: cw * 13, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: TEAL, align: "center", margin: 0 });
  s.addText("MultiJail", { x: x0 + cw * 13, y: 1.5, w: cw * 9, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: TEAL, align: "center", margin: 0 });
  LANGS.forEach((l, j) => s.addText(l, { x: x0 + j * cw, y: y0 - 0.02, w: cw, h: 0.3, fontFace: FONT, fontSize: 12, bold: true,
    color: INK, align: "center", margin: 0 }));
  Object.entries(RANKS).forEach(([m, ranks], i) => {
    const y = y0 + 0.35 + i * rh;
    s.addText(m, { x: ML, y, w: 1.85, h: rh - 0.06, fontFace: FONT, fontSize: 13, bold: true, color: INK, valign: "middle", margin: 0 });
    ranks.forEach((r, j) => {
      const fail = FAIL[m].includes(j);
      const fill = fail ? FIRE_SOFT : r === 1 ? TEAL : r <= 5 ? TEAL_MID : GREY_SOFT;
      const color = fail ? FIRE : r === 1 ? WHITE : INK;
      s.addShape(pres.shapes.RECTANGLE, { x: x0 + j * cw + 0.02, y, w: cw - 0.04, h: rh - 0.06, fill: { color: fill }, line: { color: fill } });
      s.addText(String(r), { x: x0 + j * cw, y, w: cw, h: rh - 0.06, fontFace: FONT, fontSize: r > 999 ? 9 : 12, bold: true, color,
        align: "center", valign: "middle", margin: 0 });
    });
  });
  s.addShape(pres.shapes.LINE, { x: x0 + cw * 13, y: 1.5, w: 0, h: 2.75, line: { color: MUTED, width: 1, dashType: "dash" } });
  // legend
  const ly = 4.35;
  [[TEAL, WHITE, "1", "ranked first"], [TEAL_MID, INK, "2–5", "top 5"], [GREY_SOFT, INK, "6+", "lower"], [FIRE_SOFT, FIRE, "!", "neuron stops detecting"]]
    .forEach(([f, c, txt, lab], k) => {
      const x = ML + 1.9 + k * 2.6;
      s.addShape(pres.shapes.RECTANGLE, { x, y: ly, w: 0.45, h: 0.32, fill: { color: f }, line: { color: f } });
      s.addText(txt, { x, y: ly, w: 0.45, h: 0.32, fontFace: FONT, fontSize: 11, bold: true, color: c, align: "center", valign: "middle", margin: 0 });
      s.addText(lab, { x: x + 0.55, y: ly, w: 2.0, h: 0.32, fontFace: FONT, fontSize: 12, color: INK, valign: "middle", margin: 0 });
    });
  table(s, [
    ["", "In the top 5", "Ranked first", "Detection AUROC (other languages)"],
    ["Llama-3.1-8B", hl("21 of 22"), "16 of 22", "0.85–1.00"],
    ["Qwen3-8B", "16 of 22", "10 of 22", "0.95–0.99"],
    ["Qwen3-1.7B", "19 of 22", "2 of 22 (2nd in 12)", "0.92–0.99"],
  ], ML, 4.9, CW, [2.3, 2.6, 3.2, 4.03], { size: 13, rowH: 0.36 });
  s.addText("Rank of Apple's English neuron when the search is rerun from scratch in each language (out of ~110k–300k neurons)",
    { x: ML, y: 1.32, w: CW, h: 0.25, fontFace: FONT, fontSize: 12, color: MUTED, margin: 0 });
}

// ------------------------------------------------------------------ 8. the same group
{
  const s = newSlide("Across languages it is one small shared group of neurons", "p2",
    "Counting how often each neuron appears in the 22 per-language top-5 lists. On Llama, F4258 dominates. On Qwen3-8B, five neurons in layers 14–18 trade places. On Qwen3-1.7B, two neurons share the top in every language. Part 3 finds the same neurons from a different direction.");
  const groups = [
    { m: "Llama-3.1-8B", rows: [["L11:F4258 (Apple)", 21], ["L15:F9635", 18], ["L10:F1084", 16], ["L10:F4789", 15], ["L10:F9064", 13]] },
    { m: "Qwen3-8B", rows: [["L14:F7924 (Apple)", 16], ["L17:F11229", 16], ["L16:F8474", 13], ["L16:F5318", 12], ["L18:F7664", 11]] },
    { m: "Qwen3-1.7B", rows: [["L13:F3270 (Apple)", 19], ["L10:F244", 19], ["L12:F4580", 17], ["L15:F1109", 15], ["L12:F2897", 11]] },
  ];
  const w = (CW - 0.5) / 3;
  groups.forEach((g, i) => {
    const x = ML + i * (w + 0.25);
    label(s, g.m.toUpperCase(), x, 1.6, w, TEAL);
    table(s, [["Neuron", "Top-5 lists (of 22)"], ...g.rows.map(([n, c], k) => [k === 0 ? hl(n) : n, String(c)])],
      x, 1.95, w, [w * 0.6, w * 0.4], { size: 14, rowH: 0.42 });
  });
  bullets(s, [
    "The same few neurons are found whether the search runs in English, German, Thai, Bengali or Chinese",
    "Llama has one dominant refusal neuron for all languages; the Qwen models have a group of 2–5 neurons",
    "Part 3 finds the same group from a different starting point (slide 12)",
  ], ML, 4.75, CW, 1.4, 15);
  result(s, [b("Answer to Part 2's main question: "), t("one shared, language-agnostic refusal circuit, not one neuron per language.")], { y: 6.2, h: 0.6 });
}

// ------------------------------------------------------------------ 9. exceptions
{
  const s = newSlide("Exceptions: in Yoruba and Swahili the refusal neuron goes quiet", "p2",
    "Yoruba fails on all three models, Swahili on both Qwen models. Refusal rate: share of 128 harmful training prompts the unpinned model refuses (Qwen3Guard refusal label). AUROC: the neuron as a harmful-vs-harmless detector on held-out prompts in that language. Known result from the literature: translating harmful prompts into low-resource languages jailbreaks models. This gives a mechanism: the request never reaches the refusal neuron.");
  table(s, [
    ["", "Model refuses (no pin)", "Neuron detects harm (AUROC)", "Apple's neuron rank"],
    ["Other languages", "51–98%", "0.85–1.00", "1–18"],
    [warn("Yoruba"), "10–44%", warn("0.60–0.81"), "13 · 175 · 4423"],
    [warn("Swahili (Qwen models)"), "43–55%", warn("0.34–0.65"), "5 · 354"],
  ], ML, 1.6, CW, [3.0, 3.0, 3.2, 2.93], { size: 15, rowH: 0.5 });
  label(s, "WHAT IT SUGGESTS", ML, 3.95, 6, FIRE);
  bullets(s, [
    "In these languages the harmful request does not reach the refusal circuit, so the model never refuses",
    "A mechanistic explanation of low-resource-language jailbreaks: the attack works because the refusal neuron stays silent",
  ], ML, 4.3, 6.0, 1.8, 15);
  label(s, "POSSIBLE DIRECTION", 7.0, 3.95, 5.7, TEAL);
  bullets(s, [
    "Attack: rank languages by how weakly the neuron fires, to predict which translations jailbreak",
    "Defense: pin the neuron to its harmful value on low-resource prompts and see whether refusal comes back",
  ], 7.0, 4.3, 5.7, 1.8, 15);
  result(s, [b("The neuron doubles as a gauge of whether a language is 'understood' as harmful. "), t("Proposed as a jailbreak direction.")], { y: 6.2, h: 0.6, color: FIRE, bg: FIRE_SOFT });
}

// ------------------------------------------------------------------ 10. per-language attacks
{
  const s = newSlide("Does pinning the neuron switch refusal off in every language?", "p2",
    "The causal test, per language: pin Apple's neuron at its English value while the model answers 100 held-out harmful prompts in that language (512 tokens). Shown: Qwen3-1.7B, the languages finished so far. Refused: Qwen3Guard refusal label. Harmful answer: Qwen3Guard unsafe. MultiJail's lower harmful rate comes from borderline prompts; refusals still drop to under 10%. Qwen3-8B and Llama: running.");
  s.addText("Search finds the neuron (correlation). This test pins it and checks that refusal stops (cause), in each language.",
    { x: ML, y: 1.45, w: CW, h: 0.4, fontFace: FONT, fontSize: 15, color: MUTED, margin: 0 });
  label(s, "QWEN3-1.7B, 100 HELD-OUT PROMPTS PER LANGUAGE", ML, 1.95, CW);
  table(s, [
    ["Language", "Refused: no pin → pinned", "Harmful answer: no pin → pinned"],
    ["PolyRefuse English", "78% → 1%", hl("18% → 94%")],
    ["PolyRefuse German", "68% → 11%", hl("26% → 89%")],
    ["PolyRefuse Spanish", "66% → 5%", hl("28% → 86%")],
    ["PolyRefuse French", "65% → 4%", hl("22% → 84%")],
    ["MultiJail English", "59% → 7%", "9% → 76%"],
    ["MultiJail Chinese", "57% → 8%", "6% → 63%"],
    ["MultiJail Italian", "54% → 8%", "12% → 68%"],
    ["MultiJail Vietnamese", "43% → 6%", "16% → 60%"],
  ], ML, 2.3, CW, [3.6, 4.27, 4.26], { size: 14, rowH: 0.38 });
  result(s, [b("In every language so far, pinning one neuron cuts refusals to 1–11%. "), t("All 22 languages × 3 models done by tomorrow.")], { y: 6.0 });
}

// ------------------------------------------------------------------ 11. Part 3
{
  const s = newSlide("Inside the MLP: the refusal direction's top neuron is Apple's neuron", "p3",
    "Arditi's difference-of-means direction, computed over the MLP's neuron values (before down_proj) at the neuron's layer, from the 128/128 search prompts. Interventions at that layer, every token; HarmBench 191 prompts, 128 tokens, Llama-Guard-3. Steering: subtract α times the harmful-minus-harmless gap; α=4 shown. Held: Arditi's ablation with the neuron held at its harmful mean value.");
  table(s, [
    ["", "Qwen3-8B", "Llama-3.1-8B", "Qwen3-1.7B"],
    ["Direction's top neuron", hl("Apple's (23%)"), hl("Apple's (22%)"), hl("Apple's (18%)")],
    ["No intervention", "13%", "15%", "38%"],
    ["Remove the MLP direction", "22%", "23%", "48%"],
    ["Remove it, all neurons except Apple's", "19%", "17%", "41%"],
    ["Reset Apple's neuron to its harmless value", "17%", "23%", "42%"],
    ["Steer past harmless (α = 4)", hl("96%"), hl("92%"), hl("95%")],
    ["Pin Apple's neuron", hl("97%"), hl("94%"), hl("93%")],
    ["Arditi's ablation / with Apple's neuron held on", "96% / 97%", "96% / 97%", "94% / 95%"],
  ], ML, 1.55, CW, [4.9, 2.4, 2.4, 2.43], { size: 14, rowH: 0.4 });
  s.addText("% = harmful answers (attack success). Percent in row 1 = share of the direction's squared size on that one neuron.",
    { x: ML, y: 5.2, w: CW, h: 0.3, fontFace: FONT, fontSize: 12, color: MUTED, margin: 0 });
  bullets(s, [
    "Reverse route works: build the direction first, and its biggest neuron is the refusal neuron, with no neuron search",
    "Switching refusal off is not enough; the neuron has to be pushed past its harmless value",
    "Arditi's direction does not act only through this neuron: holding it on does not bring refusal back",
  ], ML, 5.55, CW, 1.5, 14);
}

// ------------------------------------------------------------------ 12. Part 3 links + Arabic
{
  const s = newSlide("The MLP direction and the cross-language search agree", "p3",
    "On Llama, the MLP direction's top neuron at layers 10, 11 and 15 is F9064, F4258 and F9635: three of the five neurons that recur across languages in Part 2. Arabic, Qwen3-1.7B: same setup on the Arabic prompts, Qwen3Guard.");
  label(s, "LLAMA-3.1-8B: THE MLP DIRECTION'S TOP NEURONS", ML, 1.6, 6.2);
  table(s, [
    ["Layer", "Top neuron", "Share", "Recurs across languages?"],
    ["10", "F9064", "11%", hl("yes, 13 of 22")],
    ["11", "F4258 (Apple)", "22%", hl("yes, 21 of 22")],
    ["15", "F9635", "10%", hl("yes, 18 of 22")],
  ], ML, 1.95, 6.2, [0.9, 1.9, 1.0, 2.4], { size: 14 });
  label(s, "QWEN3-1.7B IN ARABIC", 7.2, 1.6, 5.5);
  table(s, [
    ["", "PolyRefuse", "MultiJail"],
    ["Top neuron", hl("Apple's (23%)"), hl("Apple's (16%)")],
    ["No intervention", "42%", "16%"],
    ["Remove direction", "53%", "20%"],
    ["Steer, α = 4", "81%", "64%"],
    ["Pin Apple's neuron", "89%", "58%"],
  ], 7.2, 1.95, 5.5, [2.1, 1.7, 1.7], { size: 14 });
  result(s, [b("Two independent routes, one answer: "), t("searching neuron by neuron in 22 languages and decomposing Arditi's direction find the same small group of neurons. Arabic shows the same pattern.")], { y: 4.9, h: 0.8 });
}

// ------------------------------------------------------------------ 12b. quick check: the other direction
const INDUCE = [
  ["Qwen3-1.7B", "−10 / −20 / −40", "0% → 0% / 1% / 4%"],
  ["Llama-3.1-8B", "+1 / +2 / +4", { text: "0% → 1% / 7% / 90%", options: { bold: true, color: TEAL } }],
  ["Qwen3-8B", "−4 / −10 / −20", "running"],
];
{
  const s = newSlide("Quick check: can the neuron also cause refusal?", "p3",
    "Quick test run today: pin each neuron toward its harmful value (the value it takes on harmful prompts, and beyond) while the model answers 100 harmless Alpaca prompts, 64 tokens; refusal by an English string match. Llama: 90% of harmless prompts refused at +4 (4x its harmful mean). Qwen3-1.7B: refusals barely rise. Rough check only; a judge and longer answers would be needed before drawing conclusions.");
  s.addText("Pinning the neuron to its harmless side removes refusal. Does pinning it to its harmful side make the model refuse harmless prompts?",
    { x: ML, y: 1.5, w: CW, h: 0.6, fontFace: FONT, fontSize: 16, color: MUTED, margin: 0 });
  label(s, "100 HARMLESS PROMPTS, SHARE REFUSED", ML, 2.3, CW);
  table(s, [["Model", "Pin (toward harmful)", "Refused: no pin → pinned"], ...INDUCE], ML, 2.65, CW, [3.6, 4.0, 4.53], { size: 15, rowH: 0.48 });
  bullets(s, [
    "Llama: pushing its one neuron to 4× its harmful value makes the model refuse 90% of harmless requests. The neuron both removes and creates refusal",
    "Qwen3-1.7B: refusals barely rise; there the neuron is needed for refusal but not enough on its own (it shares the job with L10:F244)",
    "Quick test: 100 prompts, short answers, string-match refusal check; to be confirmed with a judge",
  ], ML, 4.8, CW, 1.6, 15);
}

// ------------------------------------------------------------------ 13. next
{
  const s = newSlide("Next", "next",
    "Order agreed: Part 2, then Part 3, then the rest of Part 1. Writing in December for ARR on 4 January.");
  const cols = [
    { h: "This week", c: TEAL, body: ["Per-language attacks for all 3 models (running, done tomorrow)", "Apple's K.1 judge on those attacks", "Rerank in languages where the top neuron differs"] },
    { h: "Then", c: SLATE, body: ["Part 3 on Qwen3-8B and Llama in Arabic (queued)", "Yoruba / Swahili follow-up: restore refusal by pinning the neuron", "The neuron as one guard across languages"] },
    { h: "Later", c: AMBER, body: ["What the neuron responds to (Part 1 rows 9–10)", "Writing (December), ARR 4 January", "After: images and audio"] },
  ];
  const w = (CW - 0.5) / 3;
  cols.forEach((c, i) => {
    const x = ML + i * (w + 0.25);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w, h: 0.08, fill: { color: c.c }, line: { color: c.c } });
    s.addText(c.h, { x, y: 1.85, w, h: 0.5, fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0 });
    bullets(s, c.body, x, 2.45, w, 3.0, 15);
  });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
