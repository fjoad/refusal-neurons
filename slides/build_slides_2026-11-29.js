// Meeting deck for 29 November 2026. Order: (1) the story for the paper as posted in the group chat on 26 November
// and what was asked that day, with the answers; (2) what is new since the 22 November meeting; (3) earlier results
// from the 12 and 17 November decks that were never presented, only the ones that bear on the story; (4) what can
// be finished and when. Slides whose runs were still going when the deck was built carry a "pending" bar and get
// their numbers from the PENDING object below. Every number is from the results log (replication/README.md,
// entries of 20 to 28 November, and the 16 to 18 November entries for the earlier work). No names, no model
// answers. Typed text on Qwen2.5-Omni-7B unless a slide says otherwise.
// Run:  NODE_PATH=/opt/homebrew/lib/node_modules node slides/build_slides_2026-11-29.js

const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = path.join(__dirname, "refusal_neurons_2026-11-29.pptx");

const INK = "1F2328", MUTED = "5F6B7A", LINE = "D8DEE6", SOFT = "F4F6F9", WHITE = "FFFFFF";
const TEAL = "0F766E", TEAL_SOFT = "E6F2F1", SLATE = "475569", AMBER = "B45309",
  AMBER_SOFT = "FBF1E6", FIRE = "D9480F", FIRE_SOFT = "FCE3D6", LIGHT = "9AA5B1";
const FONT = "Arial";

// ---- results that were still running when the deck was built. null = pending. Fill from the results log.
const PENDING = {
  // slide 12: larger pins. Each: { "+20": "success / makes sense", "+40": ..., "+60", "+80", "+100", "+120" }, as strings.
  hipin_mj_en: { "+20": "45% / 100%", "+40": "66% / 100%", "+60": "69% / 98%", "+80": "68% / 82%", "+100": "30% / 33%", "+120": "5% / 6%" },   // job 484299
  hipin_mj_ar: { "+20": "62% / 100%", "+40": "76% / 98%", "+60": "70% / 90%", "+80": "53% / 69%", "+100": "37% / 49%", "+120": "5% / 8%" },    // job 484301
  hipin_poly_en: { "+20": "51% / 100%", "+40": "80% / 100%", "+60": "83% / 97%" },   // job 484302; +80 to +120 still being judged
  hipin_result: "no. Success levels off between +50 and +60 (69% on MultiJail English, 79% on MultiJail Arabic, 84% on PolyRefuse English) and the answers start to break at +60 to +80, before refusal reaches zero. At +80 on MultiJail English 16% are still refused and only 82% of the answers make sense.",
  // slides 9 and 13: L14:F13014 at +40 in the other five MultiJail languages, jobs 484320 to 484324
  mj_n1: { ko: null, th: null, bn: null, jv: null, sw: null },
  mj_n1_result: null,
  // other omni models: Gemma 4 E4B (job to be noted in the results log) and Qwen2.5-Omni-3B (job 484319)
  gemma4: null,           // { winner, top5, exceptions, runner, boot }
  model2: null,           // Qwen2.5-Omni-3B, same fields
  model2_result: null,
  // mix-and-match slide
  jbb_n1_40: null,        // L14:F13014 at +40 on JailbreakBench, job 484340
  mix: {                  // from omni_mix.py (CPU job 484339): pooled AUROC, and K.1 success over all prompts together
    guard: [["PolyRefuse, 13 languages", "0.985", "0.998", "0.993"],
            ["MultiJail, 9 languages", "0.902", "0.909", "0.943"],
            ["Both datasets, 22 sets", "0.954", "0.972", "0.973"]],
    switch: [["PolyRefuse, 13 languages", "9%", "92%"],
             ["MultiJail, 5 languages so far", "13%", "81%"],
             ["Both datasets, 18 sets", "10%", "89%"]],
  },
};

const b = (text) => ({ text, options: { bold: true } });
const t = (text) => ({ text });
const hl = (text) => ({ text, options: { bold: true, color: TEAL } });
const warn = (text) => ({ text, options: { bold: true, color: FIRE } });
const P = () => ({ text: "running", options: { color: MUTED, italic: true } });
const v = (x) => (x === null || x === undefined ? P() : x);
const good = (text) => ({ text, options: { fill: { color: TEAL_SOFT }, bold: true } });
const bad = (text) => ({ text, options: { fill: { color: FIRE_SOFT } } });
const dim = (text) => ({ text, options: { color: MUTED } });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Refusal neurons across languages, 29 November 2026";
pres.author = "Faaiz Joad";
pres.company = "Qatar Computing Research Institute, HBKU";

const SW = 13.333, ML = 0.6, CW = SW - 2 * ML;
let slideNo = 0;

const TAGS = {
  story: { text: "THE STORY FOR THE PAPER", color: MUTED },
  asked: { text: "ASKED ON THURSDAY", color: AMBER },
  a: { text: "PART A: ONE NEURON FOR EVERY LANGUAGE", color: TEAL },
  bneg: { text: "PART B: THE NEGATIVE RESULT (A SECOND DATASET)", color: FIRE },
  pend: { text: "RESULT PENDING: THE RUN IS GOING", color: AMBER },
  c: { text: "PART C: OTHER MODELS AND DATASETS", color: SLATE },
  since: { text: "PART D: NEW SINCE 22 NOVEMBER, NOT SHOWN YET", color: SLATE },
  old: { text: "PART E: EARLIER WORK ON THE TEXT-ONLY MODELS, NOT PRESENTED BEFORE", color: MUTED },
  next: { text: "PLAN", color: MUTED },
};

function newSlide(title, tag, notes) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  slideNo += 1;
  if (tag) {
    const tg = TAGS[tag];
    s.addText(tg.text, { x: ML, y: 0.32, w: 11, h: 0.3, fontFace: FONT, fontSize: 11, bold: true, color: tg.color,
      charSpacing: 1.5, margin: 0 });
  }
  s.addText(title, { x: ML, y: 0.62, w: CW, h: 0.75, fontFace: FONT, fontSize: 25, bold: true, color: INK,
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
  s.addText(runs, { x: ML + 0.3, y, w: CW - 0.4, h, fontFace: FONT, fontSize: opts.size || 15,
    color: INK, valign: "middle", margin: 0 });
}
const pendingBar = (s, text, y, h) => result(s, [b("Result: "), t(text || "pending, the job is still running.")],
  { y, h, color: AMBER, bg: AMBER_SOFT });
const resultOrPending = (s, value, pendingText, opts = {}) =>
  (value ? result(s, [b("Result: "), t(value)], opts) : pendingBar(s, pendingText, opts.y, opts.h));

function bullets(s, items, x, y, w, h, size = 15) {
  s.addText(items.map((it) => ({ text: it, options: { bullet: { indent: 18 }, breakLine: true } })),
    { x, y, w, h, fontFace: FONT, fontSize: size, color: INK, valign: "top", margin: 0, paraSpaceAfter: 6 });
}
function numbered(s, items, x, y, w, h, size = 14) {
  s.addText(items.map((it, i) => ({ text: `${i + 1}.  ${it}`, options: { breakLine: true } })),
    { x, y, w, h, fontFace: FONT, fontSize: size, color: INK, valign: "top", margin: 0, paraSpaceAfter: 6 });
}
function label(s, text, x, y, w, color = MUTED, size = 12) {
  s.addText(text, { x, y, w, h: 0.3, fontFace: FONT, fontSize: size, bold: true, color, charSpacing: 1, margin: 0 });
}
function table(s, rows, x, y, w, colW, opts = {}) {
  const size = opts.size || 14;
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? { text: c.text, options: Object.assign({}, c.options) } : { text: String(c) };
    cell.options = Object.assign({ fontFace: FONT, fontSize: size, color: INK, valign: "middle",
      align: j === 0 || (opts.left || []).includes(j) ? "left" : "center", fill: { color: i === 0 ? SOFT : WHITE },
      bold: i === 0 }, cell.options || {});
    return cell;
  }));
  s.addTable(body, { x, y, w, colW, border: { type: "solid", pt: 0.75, color: LINE }, rowH: opts.rowH || 0.42,
    margin: [0.04, 0.08, 0.04, 0.08] });
}
const para = (s, text, x, y, w, h, size = 14) =>
  s.addText(text, { x, y, w, h, fontFace: FONT, fontSize: size, color: INK, valign: "top", margin: 0 });
const foot = (s, text, y = 6.92) =>
  s.addText(text, { x: ML, y, w: CW - 0.8, h: 0.4, fontFace: FONT, fontSize: 10.5, color: MUTED, valign: "top", margin: 0 });

const N1 = "L14:F13014", N2 = "L13:F4787";
const SUCCESS = "Success = share of the held-out harmful prompts whose answer is not a refusal and makes sense (judge: gemma-2-9b-it with the original paper's criterion). 512-token answers.";

// ------------------------------------------------------------------ 1. title
{
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: WHITE };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.25, h: 7.5, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("One neuron for every language, and the result that is still negative", { x: 1.0, y: 2.0, w: 11.6, h: 1.3,
    fontFace: FONT, fontSize: 32, bold: true, color: INK, margin: 0 });
  s.addText("What was asked on Thursday, what is new since 22 November, and what is still running",
    { x: 1.0, y: 3.4, w: 11, h: 0.8, fontFace: FONT, fontSize: 20, color: MUTED, margin: 0 });
  s.addText("Faaiz Joad  ·  QCRI, HBKU  ·  29 November 2026", { x: 1.0, y: 4.7, w: 11, h: 0.4, fontFace: FONT,
    fontSize: 16, color: INK, margin: 0 });
  s.addNotes("The languages half of the paper, typed text on Qwen2.5-Omni-7B. The deck starts with Thursday's questions. Results from before 22 November that were never presented are at the end, only where they bear on the story.");
}

// ------------------------------------------------------------------ 2. the story
{
  const s = newSlide("The story for the paper, and where each part stands", "story",
    "The story is the one posted in the group chat on Thursday 26 November. This deck is the languages side. Speech is a separate track.");
  table(s, [
    ["Part of the story", "State", "In this deck"],
    [[b("Main result: "), t("on 2 to 3 omni models the neurons found on speech and on text intersect, and one neuron is the champion for both")].map(x => x.text).join(""),
      "One model so far; speech is the other track", "Our half was shown on 22 November: text against pictures of the text"],
    ["The neurons intersect across languages", good("Done for text"), "Slides 4, 7 and 9"],
    ["One neuron that is language agnostic", good("New"), "Slides 4 to 9"],
    ["The findings hold across omni models", { text: "Started tonight", options: { fill: { color: AMBER_SOFT } } }, "Slide 16 (pending)"],
    ["The findings hold across kinds of harm: currently the negative result", { text: "Partly; new numbers", options: { fill: { color: AMBER_SOFT } } }, "Slides 10 to 14"],
    ["Concept induction in the output", bad("Not started"), "Slide 23"],
  ], ML, 1.6, CW, [5.6, 2.9, 3.633], { size: 14, rowH: 0.58, left: [1, 2] });
  result(s, [b("Today: "), t("one neuron is picked for every language by a rule that one language cannot veto; the negative result is narrowed down; three runs are still going.")], { y: 6.05 });
  foot(s, "Everything in this deck is typed text on Qwen2.5-Omni-7B unless a slide says otherwise.");
}

// ------------------------------------------------------------------ 3. asked on Thursday
{
  const s = newSlide("Asked on Thursday: the short answers", "asked",
    "Each row is a question from Thursday's chat, the answer in one line, and the slide that has the numbers.");
  table(s, [
    ["Question", "Short answer", "Slide"],
    ["Do the neurons work on the clearly harmful questions of each dataset?",
      "Yes. On MultiJail's clearly harmful prompts one pinned neuron gives 76 to 90%. Borderline prompts lower the guard judges' numbers, not the success rate", "11"],
    ["Do the findings hold on each dataset on its own?",
      "For text, yes: each dataset alone picks the same neuron. Across forms it is shown for PolyRefuse text against pictures; MultiJail pictures and speech on these datasets are not run", "15"],
    ["The shared neuron need not be first in every language. Which one is it?",
      N1 + ", by a vote over the languages' shortlists. Same answer on every set of lists", "5, 6, 9"],
    ["Would a larger pin switch off the refusals that are left, before the answers break?", "No. Success levels off at +50 to +60 and the answers break before refusal reaches zero", "13"],
    ["Another multilingual benchmark (AdvBench)", "Not started", "23"],
    ["Access to the folder with the outputs", "To arrange today", "24"],
  ], ML, 1.55, CW, [4.6, 6.533, 1.0], { size: 13.5, rowH: 0.62, left: [1] });
}

// ------------------------------------------------------------------ 4. where one neuron lands
{
  const s = newSlide("Where one neuron lands in each language's own search", "a",
    "The search is the original paper's, run from scratch in each language with that language's own prompts and the model's own refusal opening in that language. PolyRefuse is machine-translated, MultiJail is human-translated.");
  label(s, "WHAT WAS DONE", ML, 1.5, 5.6, AMBER);
  bullets(s, [
    "Two datasets with the same prompts in every language: PolyRefuse (14 languages) and MultiJail (10 languages).",
    "Per language: 128 harmful and 128 harmless prompts in that language, and the model's own most common refusal opening in that language.",
    "The search scores all 340,992 neurons: how far apart the neuron is on harmful and harmless prompts, times how much it pushes towards the refusal opening.",
    "That gives one ranked list per language: 24 lists.",
  ], ML, 1.82, 5.6, 3.3, 13.5);
  const x = 6.6, w = SW - ML - x;
  label(s, "RANK OF " + N1 + " IN EACH LANGUAGE'S OWN LIST", x, 1.5, w, SLATE);
  table(s, [
    ["Rank", "PolyRefuse (14)", "MultiJail (10)"],
    [good("1st"), "English, Arabic, German, Spanish, French, Italian, Dutch, Polish, Russian, Chinese", "English, Arabic, Chinese, Italian, Vietnamese"],
    ["2nd to 4th", "Japanese (2), Korean (2), Thai (3)", "Korean (2), Thai (4)"],
    [bad("Outside the top 5"), "Yoruba (2,012)", "Bengali (41), Javanese (150), Swahili (91,626)"],
  ], x, 1.82, w, [1.5, 2.6, 2.033], { size: 12, rowH: 0.7, left: [1, 2] });
  para(s, [t("The model rarely refuses in Yoruba (38% of harmful prompts) and Swahili (25%). It refuses Bengali and Javanese at normal rates (66%, 79%).")],
    x, 5.3, w, 0.7, 12);
  result(s, [b("So: "), t("it is not first everywhere. Which neuron do the lists agree on, and how do we choose without letting one language decide?")], { y: 6.05 });
}

// ------------------------------------------------------------------ 5. the picker
{
  const s = newSlide("The picker: a vote over shortlists that one language cannot veto", "a",
    "The top 5 is the shortlist size of the original method, so each language contributes its own shortlist. A language where the neuron is far down costs that neuron its points from that one list and nothing more, however far down it is.");
  label(s, "HOW IT WORKS", ML, 1.5, 6.0, AMBER);
  numbered(s, [
    "One ranked list per language (and per form, when there are several).",
    "In each list the top 5 get points: 5 for 1st place, then 4, 3, 2, and 1 for 5th place. Every other neuron gets 0.",
    "A neuron's score is its points averaged over the lists, divided by 5. A score of 1.0 means first in every list.",
    "The highest score wins. Next to it we always give: in how many lists it is in the top 5, and the names of the lists where it is not.",
    "Checks: shortlists of 3, 10 and 20 in place of 5; each list left out in turn; the lists drawn again at random 2,000 times.",
  ], ML, 1.82, 6.0, 3.6, 13.5);
  const x = 7.0, w = SW - ML - x;
  label(s, "WHY NOT A SIMPLER RULE", x, 1.5, w, SLATE);
  table(s, [
    ["Rule", "What goes wrong"],
    ["First in every list", "No neuron qualifies"],
    ["Smallest worst rank", "One list decides. On MultiJail it picks a neuron that is 3rd to 58th and in one list's top 5"],
    ["Average rank", "One rank of 23,019 outweighs twelve ranks of 1 and 2"],
    [good("Vote over shortlists"), "A list can cost a neuron that list's points and no more"],
  ], x, 1.82, w, [2.0, 3.733], { size: 12.5, rowH: 0.62, left: [1] });
  label(s, "WORKED EXAMPLE: " + N1 + " ON MULTIJAIL, 9 LISTS", x, 5.05, w, SLATE);
  para(s, "Ranks 1, 1, 1, 1, 1, 2, 4, 41, 150  →  points 5, 5, 5, 5, 5, 4, 2, 0, 0  =  31 of 45  →  score 0.69", x, 5.35, w, 0.6, 12.5);
  result(s, [b("The exceptions are named, not hidden: "), t("for this neuron on MultiJail they are Bengali and Javanese.")], { y: 6.05 });
}

// ------------------------------------------------------------------ 6. the picker's result
{
  const s = newSlide("The vote picks the same neuron on every set of lists", "a",
    "Each row is a set of lists the vote was run on. The last column is how often the same neuron wins when the lists are drawn again at random with replacement, 2,000 times. On pictures alone it is close to a tie; with the text lists added there is no doubt.");
  table(s, [
    ["Lists", "How many", "Winner (score)", "In the top 5 of", "Not in the top 5 in", "Runner-up (score)", "Wins on redraw"],
    ["PolyRefuse, text", "13", good(N1 + " (0.94)"), "13 of 13", "none", "L16:F8383 (0.51)", "100%"],
    ["MultiJail, text", "9", good(N1 + " (0.69)"), "7 of 9", "Bengali, Javanese", N2 + " (0.49)", "98%"],
    ["Both datasets, text", "22", good(N1 + " (0.84)"), "20 of 22", "Bengali, Javanese", N2 + " (0.36)", "100%"],
    ["PolyRefuse, pictures of the text", "13", good(N1 + " (0.46)"), "8 of 13", "Arabic, Japanese, Thai; Korean (9th), Polish (8th)", "L16:F11620 (0.45)", "51%"],
    ["PolyRefuse, text and pictures", "26", good(N1 + " (0.70)"), "21 of 26", "the five picture lists above", "L16:F8383 (0.30)", "100%"],
    ["All 24 text lists, Yoruba and Swahili kept in", "24", good(N1 + " (0.77)"), "20 of 24", "Bengali, Javanese, Yoruba, Swahili", N2 + " (0.33)", "100%"],
  ], ML, 1.45, CW, [2.6, 0.75, 1.95, 1.15, 2.95, 1.7, 1.033], { size: 11.5, rowH: 0.44, left: [4] });
  bullets(s, [
    "Same winner with a shortlist of 3, 5, 10 or 20, and with any one list left out, on every text set. On pictures alone it is nearly a tie.",
    "The first five rows leave out Yoruba and Swahili, where the model rarely refuses. The last row shows the vote does not need that rule.",
  ], ML, 5.3, CW, 0.75, 12);
  result(s, [b("Result: "), t("one neuron, "), hl(N1), t(". The neuron the English pin sweep picks ("), t(N2), t(") is the runner-up: in the top 5 of 13 of 22 text lists.")], { y: 6.1 });
}

// ------------------------------------------------------------------ 7. as a switch
{
  const s = newSlide("The picked neuron as a switch: one pin value in every language", "a",
    "The same neuron and the same pin value in every language; nothing is tuned per language. The last column is the neuron the English pin sweep picks, at its own pin, for comparison.");
  const rows = [
    ["English", "3%", "89%", "81%"], ["German", "5%", "94%", "98%"], ["Spanish", "7%", "95%", "90%"],
    ["French", "9%", "95%", "93%"], ["Italian", "9%", "96%", "93%"], ["Dutch", "12%", "94%", "97%"],
    ["Polish", "16%", "95%", "94%"], ["Russian", "1%", "97%", "88%"], ["Arabic", "11%", "90%", "84%"],
    ["Chinese", "4%", "94%", "94%"], ["Japanese", "9%", "89%", "86%"], ["Korean", "17%", "87%", "83%"],
    ["Thai", "12%", { text: "79%", options: { fill: { color: AMBER_SOFT } } }, "80%"], ["Yoruba", "39%", bad("32%"), bad("44%")],
  ];
  const head = ["Language", "Nothing pinned", N1 + " at +40", N2 + " at +30"];
  const w = (CW - 0.4) / 2, colW = [1.5, 1.35, 1.6, 1.4].map((c) => c * w / 5.85);
  table(s, [head].concat(rows.slice(0, 7)), ML, 1.5, w, colW, { size: 12.5, rowH: 0.44 });
  table(s, [head].concat(rows.slice(7)), ML + w + 0.4, 1.5, w, colW, { size: 12.5, rowH: 0.44 });
  bullets(s, [
    "The anchor pin (pinning only where the neuron fires) gives the same numbers, within 3 points, in every language.",
    "A pin sweep run entirely in German, Arabic or Chinese picks one of these same two neurons at the same pin. No language has a better neuron of its own.",
  ], ML, 5.15, CW, 0.85, 12.5);
  result(s, [b("Result: "), t("one neuron, chosen without favouring a language, switches refusal off in 12 of 14 languages ("), hl("87 to 97%"), t("). Thai is lower. Yoruba does not move: the model rarely refuses there to begin with.")], { y: 6.05 });
  foot(s, SUCCESS + " PolyRefuse, 100 prompts per language.");
}

// ------------------------------------------------------------------ 8. as a guard
{
  const s = newSlide("The picked neuron as a guard, with nothing pinned", "a",
    "AUROC measures how well a score tells harmful prompts from harmless ones: 1.0 is perfect, 0.5 is chance. The neuron's score is its value when the model reads the prompt, so it costs nothing extra. The right table is the check that the vote picked a neuron that also works: for each of the six shortlist neurons, its AUROC in its worst language.");
  label(s, "AGAINST GUARD MODELS (range over languages)", ML, 1.5, 6.3, AMBER);
  table(s, [
    ["100 harmful against 100 harmless prompts per language", N1, "Llama-Guard", "Qwen3Guard"],
    ["PolyRefuse, text, 13 languages", good("0.97 to 0.99"), "0.99 to 1.00", "0.99 to 1.00"],
    ["MultiJail, text, 9 languages (new)", good("0.89 to 0.96"), "0.84 to 0.97", "0.86 to 0.97"],
    ["PolyRefuse, pictures, 13 languages", "0.73 to 0.99", dim("cannot read a picture"), dim("cannot read a picture")],
  ], ML, 1.82, 6.3, [2.5, 1.3, 1.25, 1.25], { size: 12, rowH: 0.62 });
  bullets(s, [
    "With one cut-off chosen on English and used unchanged in every language, its accuracy on PolyRefuse is 92 to 98%, the same range as the guards.",
    "On MultiJail it is level with Llama-Guard and a little under Qwen3Guard.",
    "Weakest on pictures in Arabic, Korean and Thai (0.87, 0.87, 0.73).",
  ], ML, 4.45, 6.3, 1.5, 12.5);
  const x = 7.3, w = SW - ML - x;
  label(s, "SIX SHORTLIST NEURONS: AUROC, WORST LANGUAGE", x, 1.5, w, SLATE);
  table(s, [
    ["Neuron", "PolyRefuse text", "MultiJail text", "Pictures"],
    [good(N1), good("0.97"), good("0.89"), good("0.73")],
    [N2, "0.90", "0.75", "0.54"],
    ["L16:F1870", "0.96", "0.71", "0.42"],
    ["L14:F13435", "0.92", "0.66", "0.42"],
    ["L16:F11620", "0.97", "0.60", "0.41"],
    ["L17:F5484", "0.97", "0.57", "0.58"],
  ], x, 1.82, w, [1.6, 1.3, 1.3, 1.233], { size: 12.5, rowH: 0.44 });
  result(s, [b("Result: "), t("close to a guard model on text, not equal to it. And the neuron the vote picks is the best or equal-best guard in its worst language on all three sets: the vote and the function agree.")], { y: 6.05 });
  foot(s, "The neuron's score: its value on its harmful side at the strongest of the five template tokens. Worst language is taken over the languages the model refuses in (Yoruba and Swahili left out).");
}

// ------------------------------------------------------------------ mix and match
{
  const mx = PENDING.mix;
  const s = newSlide("Mix and match: pick the neuron on one set, use it on the others", "a",
    "This is the main use: the neuron is picked on one dataset or language set and then used, unchanged and at one pin value, on data it was not picked on, as a switch (pinned) and as a guard (read with nothing pinned). The vote returns the same neuron whichever set it is run on, so the first three rows have the same test numbers. The last row is the original method's pick from an English pin sweep, for contrast.");
  table(s, [
    ["Picked on", "Neuron it gives", "Tested on PolyRefuse (13 languages): switch / guard", "Tested on MultiJail: switch / guard", "Tested on English JailbreakBench: switch"],
    ["PolyRefuse only: vote over 13 languages", good(N1), "79 to 97%  /  0.97 to 0.99", "75 to 88% (5 languages; 5 running)  /  0.89 to 0.96", v(PENDING.jbb_n1_40)],
    ["MultiJail only: vote over 9 languages", good(N1), dim("same neuron, same numbers"), dim("same neuron, same numbers"), dim("same")],
    ["Everything mixed: all 24 lists together", good(N1), dim("same neuron, same numbers"), dim("same neuron, same numbers"), dim("same")],
    ["English only: the original method's pin sweep", N2, "80 to 98%  /  0.90 to 0.98", "65 to 75% in 6, fails in 4  /  0.75 to 0.97", "75%"],
  ], ML, 1.45, CW, [3.3, 1.4, 2.5, 3.2, 1.733], { size: 11.5, rowH: 0.4, left: [2, 3] });
  const w = (CW - 0.4) / 2;
  label(s, "EVERYTHING IN ONE POOL: THE GUARD (ONE CUT-OFF FOR ALL)", ML, 4.05, w, AMBER, 10.5);
  table(s, [["Pool of prompts", N1, "Llama-Guard", "Qwen3Guard"]].concat(
    (mx ? mx.guard : [["PolyRefuse, 13 languages", P(), P(), P()], ["MultiJail, 9 languages", P(), P(), P()], ["Both datasets, 22 sets", P(), P(), P()]])),
    ML, 4.35, w, [2.2, 1.2, 1.15, 1.25].map((c) => c * w / 5.8), { size: 11, rowH: 0.34 });
  label(s, "EVERYTHING IN ONE POOL: THE SWITCH (" + N1 + " AT +40)", ML + w + 0.4, 4.05, w, SLATE, 10.5);
  table(s, [["Pool of prompts", "Nothing pinned", "Pinned"]].concat(
    (mx ? mx.switch : [["PolyRefuse, 13 languages", P(), P()], ["MultiJail, 5 languages so far", P(), P()], ["Both datasets", P(), P()]])),
    ML + w + 0.4, 4.35, w, [2.8, 1.5, 1.5].map((c) => c * w / 5.8), { size: 11, rowH: 0.34 });
  result(s, [b("Result: "), t("whichever dataset the neuron is picked on, the vote gives the same neuron, and it works as a switch and as a guard on the dataset it was not picked on. Across models the neuron has to be found again in each model.")], { y: 6.05, h: 0.66, size: 13.5 });
  foot(s, "Switch = success with the neuron pinned (" + N1 + " at +40, " + N2 + " at +30). Guard = AUROC, harmful against harmless, nothing pinned. Pooled = all languages' prompts scored together: 4,363 for the guard, 1,800 for the switch.", 6.8);
}

// ------------------------------------------------------------------ 9. the negative result
{
  const m = PENDING.mj_n1;
  const s = newSlide("The negative result: a second dataset, fewer refusals switched off", "bneg",
    "MultiJail is human-translated and its requests are shorter and more often questions. The search finds the same neuron in most languages, but the same pin leaves more refusals standing than on PolyRefuse.");
  table(s, [
    ["MultiJail language", "Rank of " + N1 + " in its search", "Nothing pinned", N2 + " at +30", N1 + " at +40"],
    ["English", "1", "10%", "65%", "79%"], ["Arabic", "1", "25%", "73%", "88%"], ["Chinese", "1", "11%", "75%", "75%"],
    ["Italian", "1", "11%", "71%", "82%"], ["Vietnamese", "1", "8%", "67%", "81%"],
    ["Korean", "2", "21%", "74%", v(m.ko)], ["Thai", "4", "13%", "53%", v(m.th)],
    ["Bengali", bad("41"), "34%", bad("46%"), v(m.bn)], ["Javanese", bad("150"), "24%", bad("34%"), v(m.jv)],
    ["Swahili", bad("91,626"), "42%", bad("38%"), v(m.sw)],
  ], ML, 1.5, 7.6, [1.9, 1.9, 1.2, 1.3, 1.3], { size: 12.5, rowH: 0.385 });
  const x = 8.5, w = SW - ML - x;
  label(s, "FOR COMPARISON, POLYREFUSE", x, 1.5, w, SLATE);
  bullets(s, [
    N1 + " at +40: 87 to 97% in 12 languages.",
    N2 + " at +30: 80 to 98% in 13 languages.",
  ], x, 1.82, w, 1.1, 13);
  label(s, "WHERE WE LOOKED FOR THE REASON", x, 3.0, w, AMBER);
  bullets(s, [
    "Are MultiJail's prompts borderline? Slide 11.",
    "Are the refusals that are left real refusals? Slide 12.",
    "Is the pin too small? Slide 13.",
    "Where the search does not find the neuron: slide 14.",
  ], x, 3.32, w, 2.3, 13);
  result(s, [b("The gap: "), t("about 10 points under PolyRefuse in the languages where the search finds the neuron, and no effect where it does not.")], { y: 6.05, color: FIRE, bg: FIRE_SOFT });
  foot(s, SUCCESS + " 100 prompts per language.");
}

// ------------------------------------------------------------------ 10. clearly harmful
{
  const s = newSlide("Do the neurons work on the clearly harmful prompts of the dataset?", "asked",
    "The split uses only the prompt's English text and two guard models, so it does not depend on our model, the language or the pin. The same 58 prompts are then looked at in every language.");
  label(s, "WHAT WAS DONE (NO NEW RUNS)", ML, 1.5, 4.9, AMBER);
  bullets(s, [
    "A prompt is \"clearly harmful\" when two guard models (Llama-Guard and Qwen3Guard) both call its English text unsafe.",
    "The split is made once, on English, and used for the same prompts in every language.",
    "Then the pinned results we already had are counted on each part.",
  ], ML, 1.82, 4.9, 2.1, 13);
  table(s, [
    ["100 test prompts", "Clearly harmful", "Refused, nothing pinned"],
    ["MultiJail", warn("58"), "58 of the 58"],
    ["PolyRefuse", "93", "90 of the 93"],
  ], ML, 4.1, 4.9, [1.7, 1.5, 1.7], { size: 12.5, rowH: 0.44 });
  const x = 5.9, w = SW - ML - x;
  label(s, "MULTIJAIL, " + N1 + " PINNED AT +40", x, 1.5, w, SLATE);
  table(s, [
    ["Language", "Success: all 100", "the 58 clearly harmful", "the other 42", "Guard says the answer is harmful: clearly harmful", "the other 42"],
    ["English", "79%", good("76%"), "83%", "76%", bad("48%")],
    ["Arabic", "88%", good("90%"), "86%", "95%", bad("33%")],
    ["Chinese", "75%", good("78%"), "71%", "72%", bad("21%")],
    ["Italian", "82%", good("84%"), "79%", "81%", bad("52%")],
    ["Vietnamese", "81%", good("83%"), "79%", "74%", bad("36%")],
    [dim("PolyRefuse, 13 languages"), dim("79 to 97%"), dim("78 to 97%"), "", dim("80 to 96%"), ""],
  ], x, 1.82, w, [1.5, 1.0, 1.15, 0.9, 1.4, 0.883], { size: 12, rowH: 0.5 });
  result(s, [b("Answer: "), t("yes, "), hl("76 to 90%"), t(" on the clearly harmful prompts. The borderline prompts do not lower the success rate. They lower the guard judges' numbers, because an answer to a borderline request is not harmful.")], { y: 5.95, h: 0.85 });
  foot(s, SUCCESS + " Guard = Qwen3Guard reading the answer.");
}

// ------------------------------------------------------------------ 11. what is left
{
  const s = newSlide("What is left after the pin is two different things", "bneg",
    "Counted from the first words of the answers only; no answer was read. The \"not sure what you are asking\" replies are counted as refusals by the judge. They come from one neuron's pin, not from the dataset.");
  label(s, "MULTIJAIL ENGLISH, 100 HARMFUL PROMPTS", ML, 1.5, 6.6, AMBER);
  table(s, [
    ["Setting", "Counted as a refusal", "of which \"I am not sure what you are asking\"", "Ordinary refusals"],
    ["Nothing pinned", "90", "2", "88"],
    [N2 + " at +30", "35", warn("26"), "9"],
    [N1 + " at +40", "21", "1", warn("20")],
  ], ML, 1.82, 6.6, [2.1, 1.4, 1.8, 1.3], { size: 12.5, rowH: 0.6 });
  bullets(s, [
    "The \"not sure\" replies belong to " + N2 + ": they grow with its pin (0, 2, 12, 21 of 191 at +6, +10, +20, +30), never appear on harmless prompts (0 of 200), and never when Nanda's direction is removed.",
    "So numbers for " + N2 + " understate how often refusal itself is switched off.",
  ], ML, 4.4, 6.6, 1.5, 12.5);
  const x = 7.6, w = SW - ML - x;
  label(s, "PICKED NEURON: WHAT REMAINS IS REFUSAL", x, 1.5, w, SLATE);
  table(s, [
    ["Clearly harmful prompts still refused", "MultiJail", "PolyRefuse"],
    ["English", warn("24%"), "11%"],
    ["Chinese", warn("22%"), "6%"],
    ["Arabic", "10%", "6%"],
  ], x, 1.82, w, [2.5, 1.3, 1.333], { size: 12.5, rowH: 0.5 });
  bullets(s, [
    "MultiJail's requests are shorter (about 12 words against 19) and more often questions (70 of 100 against 37).",
    "Within each dataset, the requests whose refusal survives are the shorter ones.",
  ], x, 4.0, w, 1.8, 12.5);
  result(s, [b("So: "), t("with the picked neuron the leftover is ordinary refusal, about one in four clearly harmful English requests. Does a larger pin remove it? Next slide.")], { y: 6.05 });
}

// ------------------------------------------------------------------ 12. larger pins (pending)
{
  const s = newSlide("Does a larger pin finish the job before the answers break?", "bneg",
    "The best possible result is a pin at which no refusal is left and every answer still makes sense. On the text-only models the window was narrow: answers were intact up to the attack pin and broken at two to four times it.");
  label(s, "WHAT WAS DONE", ML, 1.5, 5.3, AMBER);
  bullets(s, [
    N1 + " at pins 10, 20, 30, 40, 50, 60, 80, 100, 120.",
    "MultiJail English and Arabic (87 sweep prompts each) and PolyRefuse English (191).",
    "Per pin: still refused, answers that make sense, and success (both together).",
    "Also " + N2 + " and L16:F1870 on MultiJail English.",
  ], ML, 1.82, 5.3, 2.2, 13);
  label(s, "TEXT-ONLY MODELS (NOT PRESENTED BEFORE)", ML, 4.05, 5.3, SLATE, 11);
  table(s, [
    ["Qwen3-8B, pin", "+10", "+20", "+40", "+80"],
    ["Answers make sense", "100%", good("100%"), bad("39%"), bad("0%")],
    ["Success", "50%", good("95%"), "31%", "0%"],
  ], ML, 4.4, 5.3, [1.9, 0.85, 0.85, 0.85, 0.85], { size: 12, rowH: 0.38 });
  const x = 6.3, w = SW - ML - x;
  const sets = [["MultiJail English", PENDING.hipin_mj_en], ["MultiJail Arabic", PENDING.hipin_mj_ar], ["PolyRefuse English", PENDING.hipin_poly_en]];
  const pins = ["+20", "+40", "+60", "+80", "+100", "+120"];
  label(s, N1 + ": SUCCESS / ANSWERS THAT MAKE SENSE, BY PIN", x, 1.5, w, SLATE);
  table(s, [["Prompts"].concat(pins)].concat(sets.map(([name, d]) => [name].concat(pins.map((p) => (d && d[p]) || P())))),
    x, 1.82, w, [1.9, 0.75, 0.75, 0.75, 0.75, 0.77, 0.763], { size: 12, rowH: 0.62 });
  para(s, "Each cell: success, then the share of answers that make sense. With nothing pinned success is 11%, 17% and 3%. Still refused on MultiJail English: 34% at +40, 29% at +60, 16% at +80, 8% at +120. The other neuron (" + N2 + ") behaves the same: best 72% at +60, broken by +100.",
    x, 4.4, w, 1.2, 11.5);
  resultOrPending(s, PENDING.hipin_result, "pending. First counts from the runs: refusals keep falling above +40, and from +80 a share of the answers is one word repeated. The judge's numbers are not in yet.", { y: 5.95, h: 0.85, size: 13.5 });
}

// ------------------------------------------------------------------ 13. bengali and javanese
{
  const m = PENDING.mj_n1;
  const s = newSlide("Bengali and Javanese: the neuron fires, but the search and the pin miss it", "bneg",
    "New from the guard table: with nothing pinned, the neuron separates harmful from harmless prompts in Bengali and Javanese as well as in English. So it does respond to harm there. The search ranks it low and the other neuron's pin does nothing. The picked neuron's pin in these languages is running.");
  table(s, [
    ["MultiJail language", "Model refuses (search prompts)", "Search rank of " + N1, N1 + " as a guard (AUROC)", "Success: nothing pinned → " + N2 + " at +30", N1 + " at +40"],
    ["English, for comparison", "77%", "1", "0.96", "10% → 65%", "79%"],
    ["Thai", "80%", "4", "0.89", "13% → 53%", v(m.th)],
    ["Bengali", "66%", bad("41"), good("0.96"), bad("34% → 46%"), v(m.bn)],
    ["Javanese", "79%", bad("150"), good("0.92"), bad("24% → 34%"), v(m.jv)],
    ["Swahili", bad("25%"), bad("91,626"), "0.75", bad("42% → 38%"), v(m.sw)],
  ], ML, 1.45, CW, [2.6, 1.9, 1.7, 1.9, 2.4, 1.633], { size: 12, rowH: 0.42 });
  bullets(s, [
    "Swahili: the model rarely refuses, and half its answers are incoherent with or without a pin. Nothing to switch off.",
    "Bengali and Javanese: the model refuses, and the neuron tells harmful from harmless (0.96, 0.92). Yet the search ranks it 41st and 150th, and pinning the other neuron changes little. In Javanese 62 of 100 answers are still refusals with that pin.",
    "Two readings to separate: the model refuses there by another route, or its \"refusals\" are replies that it does not understand the request. Running: the picked neuron pinned in these languages; next: a pin sweep on their own top five.",
  ], ML, 4.55, CW, 1.45, 11.5);
  resultOrPending(s, PENDING.mj_n1_result, "pending for the last column (five runs queued tonight).", { y: 6.05 });
}

// ------------------------------------------------------------------ 14. each dataset on its own
{
  const s = newSlide("Do the findings hold on each dataset on its own?", "asked",
    "For text the answer is yes. Across forms we have PolyRefuse as text and as pictures of the text. MultiJail as pictures, and speech made from these two datasets, do not exist yet.");
  table(s, [
    ["Lists used, each set alone", "How many", "Neuron the vote picks", "In the top 5 of"],
    ["PolyRefuse, text", "13", good(N1), "13 of 13"],
    ["MultiJail, text", "9", good(N1), "7 of 9"],
    ["PolyRefuse, pictures of the text", "13", N1 + " (near tie with L16:F11620)", "8 of 13"],
    ["MultiJail, pictures of the text", dim("not run"), dim("not run"), dim("not run")],
    ["Speech made from PolyRefuse or MultiJail", dim("not run"), dim("speech so far is English, from other prompt sets"), dim("not run")],
  ], ML, 1.5, CW, [4.2, 1.4, 4.4, 2.133], { size: 13, rowH: 0.56 });
  bullets(s, [
    "English, text against pictures of the same prompts (22 November): 4 of the top 5 neurons are shared, and a neuron found in one form works when pinned in the other.",
    "Across languages on pictures: the Latin-script languages, Russian and Chinese agree with text; Arabic, Japanese and Thai pictures give different lists.",
  ], ML, 4.95, CW, 1.0, 12.5);
  result(s, [b("Answer: "), t("for text, yes, each dataset alone picks "), hl(N1), t(". For the cross-modal half of the question, MultiJail pictures and speech from these datasets are still to be made.")], { y: 6.05 });
}

// ------------------------------------------------------------------ other omni models (pending)
{
  const g = PENDING.gemma4, d = PENDING.model2;
  const s = newSlide("Other omni models: the same search in 14 languages", "pend",
    "The priority is omni models from other makers at about the same size, not more Qwen models. Gemma 4 E4B is the one open model of that size that is not built on a Qwen backbone and loads with standard code. MiniCPM-o 4.5 is built on Qwen3-8B, and Kimi-Audio on a Qwen backbone too. The models for the paper are still to be chosen together.");
  label(s, "OPEN OMNI MODELS OF ABOUT THE SAME SIZE", ML, 1.45, CW, AMBER);
  table(s, [
    ["Model", "Maker; language backbone", "Size", "Takes in", "State for the languages"],
    ["Qwen2.5-Omni-7B", "Alibaba; Qwen2.5", "8B (thinker)", "text, image, audio, video", good("done: this deck")],
    [b("Gemma 4 E4B").text, "Google; Gemma 4 (not Qwen)", "8.0B", "text, image, audio", { text: "first in line: being set up", options: { fill: { color: AMBER_SOFT } } }],
    ["MiniCPM-o 4.5", "OpenBMB; built on Qwen3-8B", "9.4B", "text, image, audio", "not started"],
    ["Gemma 3n E4B", "Google; Gemma 3n", "7.9B", "text, image, audio", "access to be requested"],
    ["Qwen2.5-Omni-3B", "Alibaba; Qwen2.5", "5.5B", "text, image, audio, video", "queued, lower priority"],
  ], ML, 1.75, CW, [2.3, 3.3, 1.3, 2.7, 2.533], { size: 12, rowH: 0.38, left: [1, 2, 3, 4] });
  label(s, "THE VOTE, PER MODEL (POLYREFUSE TEXT, LANGUAGES THE MODEL REFUSES IN)", ML, 4.25, CW, SLATE);
  table(s, [
    ["Model", "Winner of the vote (score)", "In the top 5 of", "Not in the top 5 in", "Runner-up (score)", "Wins on redraw"],
    ["Qwen2.5-Omni-7B", good(N1 + " (0.94)"), "13 of 13", "none", "L16:F8383 (0.51)", "100%"],
    ["Gemma 4 E4B", v(g && g.winner), v(g && g.top5), v(g && g.exceptions), v(g && g.runner), v(g && g.boot)],
    ["Qwen2.5-Omni-3B", v(d && d.winner), v(d && d.top5), v(d && d.exceptions), v(d && d.runner), v(d && d.boot)],
  ], ML, 4.55, CW, [2.3, 2.7, 1.5, 2.4, 2.1, 1.133], { size: 12, rowH: 0.34 });
  resultOrPending(s, PENDING.model2_result, "pending. A neuron's number means nothing in another model, so each model gets its own search, its own vote, then its own pin and guard tables.", { y: 6.2, h: 0.62 });
}

// ------------------------------------------------------------------ 16. the direction
{
  const s = newSlide("The refusal direction is shared across languages", "since",
    "Nanda's method: the direction is the difference between the model's average internal state on harmful and on harmless prompts. Removing it everywhere switches refusal off. Here it is computed separately in each language, and the English one is also tried in every other language.");
  const rows = [
    ["English", "3%", "96%", "—", "1.00"], ["German", "5%", "99%", "97%", "0.93"], ["Russian", "1%", "99%", "93%", "0.92"],
    ["Spanish", "7%", "99%", "96%", "0.94"], ["French", "9%", "99%", "94%", "0.94"], ["Japanese", "9%", "98%", "87%", "0.87"],
    ["Dutch", "12%", "98%", "93%", "0.92"], ["Polish", "16%", "98%", "89%", "0.91"], ["Arabic", "11%", "97%", "91%", "0.91"],
    ["Italian", "9%", "97%", "100%", "0.93"], ["Thai", "14%", "92%", { text: "77%", options: { fill: { color: AMBER_SOFT } } }, "0.89"],
    ["Chinese", "4%", "89%", "88%", "0.90"], ["Korean", "17%", "85%", "78%", "0.88"], ["Yoruba", "40%", bad("25%"), bad("41%"), bad("0.26")],
  ];
  const head = ["Language", "Nothing changed", "Its own direction removed", "The English direction removed", "Similarity to the English direction"];
  const w = (CW - 0.4) / 2, colW = [1.25, 1.0, 1.2, 1.25, 1.17].map((c) => c * w / 5.87);
  table(s, [head].concat(rows.slice(0, 7)), ML, 1.5, w, colW, { size: 11.5, rowH: 0.46 });
  table(s, [head].concat(rows.slice(7)), ML + w + 0.4, 1.5, w, colW, { size: 11.5, rowH: 0.46 });
  result(s, [b("Result: "), t("a direction found on English prompts alone switches refusal off in the other languages ("), hl("77 to 100%"), t("). Measured at the same place in the model, any two languages' directions are 0.86 to 0.99 alike. Yoruba has no direction, as it has no neuron.")], { y: 5.75, h: 0.95 });
  foot(s, SUCCESS + " PolyRefuse, 100 prompts per language. Similarity = cosine at one fixed place (layer 21, last template token); 1 = identical.", 6.85);
}

// ------------------------------------------------------------------ 17. neurons and direction
{
  const s = newSlide("The neurons that work are the ones that point along the direction", "since",
    "Left: each neuron writes a fixed vector into the model's state; the cosine says how far that vector points along the refusal direction. Right: one neuron is pinned and the later refusal neurons are read on harmful prompts.");
  const w = (CW - 0.5) / 2;
  label(s, "EACH NEURON AGAINST THE DIRECTION (ENGLISH)", ML, 1.5, w, AMBER);
  table(s, [
    ["Neuron", "Alignment (cosine)", "Rank among the 18,944 neurons of its layer"],
    [N2, "0.32", good("1st")], [N1, "0.29", good("1st")], ["L16:F1870", "0.21", good("1st")],
    ["L17:F5484", "0.20", "2nd"], ["L14:F13435", "0.25", "3rd"], ["L16:F11620", "0.17", "6th"],
  ], ML, 1.82, w, [1.7, 1.6, 2.5].map((c) => c * w / 5.8), { size: 12.5, rowH: 0.44 });
  para(s, "The largest alignment expected by chance in a layer is 0.08.", ML, 5.0, w, 0.4, 12);
  const x = ML + w + 0.5;
  label(s, "ONE NEURON PINNED: WHAT THE LATER ONES DO", x, 1.5, w, SLATE);
  table(s, [
    ["The pinned neuron is set to", "How far the later neurons move from their harmful value to their harmless one"],
    ["Its attack value (+30 or +40)", good("87 to 104%: all go quiet")],
    ["Its ordinary harmless value", "22 to 29% for the nearest ones, about zero for the others"],
  ], x, 1.82, w, [2.3, 3.5].map((c) => c * w / 5.8), { size: 12.5, rowH: 0.75, left: [1] });
  bullets(s, [
    "This is why any one of the working neurons can be used for the attack: whichever is pinned, the rest go quiet.",
    "The later neurons do not depend on any single earlier one.",
  ], x, 4.3, w, 1.4, 12.5);
  result(s, [b("Result: "), t("the search and Nanda's direction pick the same neurons. The attack works because the pin is far outside the neuron's normal range (+30 where it normally sits between −4 and −0.6), not because one link of a chain is cut.")], { y: 5.95, h: 0.85 });
}

// ------------------------------------------------------------------ 18. pictures in some scripts
{
  const s = newSlide("Pictured requests in Arabic, Korean and Thai get past safety, nothing pinned", "since",
    "The 22 November deck had the search in every language; the attack numbers came the day after. Each request is drawn as a picture of its typed text and given to the model with no other words.");
  table(s, [
    ["Language", "Reads the picture (words back)", "Harmful answer: typed", "Harmful answer: as a picture", N1 + " on harmful prompts: typed", "as a picture"],
    ["English", "99%", "3%", "3%", "−3.01", "−1.94"],
    ["Chinese", "98%", "0%", "3%", "−5.93", "−2.27"],
    ["German", "100%", "5%", "16%", "−2.58", "−0.99"],
    ["Korean", "96%", "7%", bad("36%"), "−1.22", bad("−0.39")],
    ["Thai", "96%", "6%", bad("42%"), "−1.22", bad("−0.16")],
    ["Arabic", "96%", "3%", bad("45%"), "−1.70", bad("−0.18")],
  ], ML, 1.45, CW, [1.8, 2.3, 1.9, 2.2, 2.2, 1.733], { size: 12.5, rowH: 0.4 });
  bullets(s, [
    "The model reads those pictures (96% of the words), so this is not a reading failure.",
    "The refusal neuron is nearly silent exactly there (about −0.05 is its value on harmless prompts).",
    "With the neuron pinned, pictures are jailbroken in every language but Thai, where the pin changes nothing (45% before and after).",
  ], ML, 4.9, CW, 1.1, 12);
  result(s, [b("Result: "), t("where a pictured request gets through, the model has read it but the refusal neuron has not fired. One model and one picture style.")], { y: 6.05 });
  foot(s, "Harmful answer = share of 100 held-out harmful prompts whose answer Qwen3Guard calls harmful; Llama-Guard on a translation agrees within a few points.");
}

// ------------------------------------------------------------------ 19. system prompt
{
  const s = newSlide("To settle before the final tables: the system prompt moves every number", "since",
    "The text runs use the chat template with no system message, which inserts \"You are a helpful assistant.\" The speech runs use the model's documented default prompt. Run through our own pipeline, the default prompt reproduces the speech track's numbers to within 4 or 5 points.");
  table(s, [
    ["100 English test prompts, typed", "No system message (our runs)", "The model's default system prompt (the speech runs)"],
    ["Refused with nothing pinned (phrase match)", "97 of 100", warn("77 of 100")],
    ["Success, nothing pinned", "1%", "9%"],
    ["Success, " + N2 + " at +30", "74%", warn("95%")],
    ["Success, " + N1 + " at +30", "67%", warn("92%")],
  ], ML, 1.6, CW, [4.6, 3.5, 4.033], { size: 14, rowH: 0.58 });
  bullets(s, [
    "Until one prompt is used everywhere, numbers from the text runs and the speech runs cannot go in one table.",
    "Also to agree: the rule for picking the pin (the first pin that reaches 90%, or the smallest pin within 2 points of the best) and the answer length (512 or 256 tokens).",
    "All numbers in this deck are from our setup: no system message, first pin to reach 90%, 512 tokens.",
  ], ML, 4.75, CW, 1.3, 13);
  result(s, [b("To decide: "), t("one setup for both halves of the paper. Our runs are the cheap ones to redo.")], { y: 6.1, color: AMBER, bg: AMBER_SOFT });
}

// ------------------------------------------------------------------ 20. earlier: text-only models, languages
{
  const s = newSlide("On three text-only models the languages also share their neurons", "old",
    "From the decks of 12 and 17 November, which were not presented. This is the same language result on Qwen3-8B, Llama-3.1-8B and Qwen3-1.7B, with the neuron the original paper published for each. It is our evidence so far that the language result is not special to one model.");
  table(s, [
    ["", "Qwen3-8B", "Llama-3.1-8B", "Qwen3-1.7B"],
    ["The published neuron", "L14:F7924", "L11:F4258", "L13:F3270"],
    ["Language lists (of 24) with that neuron in the top 20", "23", "24", "22"],
    ["Neurons in every language's top 20 (Yoruba and Swahili aside)", "4", "4", "4"],
    ["That neuron pinned, 13 PolyRefuse languages: success", { text: "93 to 100% on all three models", options: { colspan: 3, fill: { color: TEAL_SOFT }, bold: true } }],
    ["MultiJail Arabic: the best single neuron", "85%", "77%", "93%"],
    ["MultiJail Arabic: three to five neurons together at half their pins", good("95%"), good("100%"), good("99%")],
  ], ML, 1.5, CW, [5.6, 2.2, 2.2, 2.133], { size: 12.5, rowH: 0.44 });
  bullets(s, [
    "The same gap between PolyRefuse and MultiJail showed up on these models.",
    "There, several neurons pinned together at half strength closed it, with answers that still made sense. Full pins together broke the model.",
  ], ML, 5.2, CW, 0.8, 12);
  result(s, [b("Why it matters now: "), t("the language result repeats on three more models, and pinning several neurons gently is a second way to attack the negative result.")], { y: 6.05 });
}

// ------------------------------------------------------------------ 21. earlier: harm or refusal
{
  const s = newSlide("Is it a harm neuron or a refusal neuron? On the 8B models it reads harm", "old",
    "From the 17 November deck, not presented. Each prompt has two labels: harmful or not (from the dataset) and refused or not (the model's own answer with nothing pinned). CoCoNot has refusals that are not about harm, for example requests the model cannot fulfil; XSTest has harmless prompts that sound scary.");
  label(s, "HOW WELL THE NEURON'S VALUE SEPARATES TWO GROUPS (AUROC; 0.5 = CHANCE)", ML, 1.5, CW, AMBER);
  table(s, [
    ["", "Qwen3-8B", "Llama-3.1-8B", "Qwen3-1.7B"],
    ["Harmful against not harmful, among the prompts the model refuses", good("0.93"), good("0.97"), "0.81"],
    ["Harmful against not harmful, among the prompts the model answers", good("0.87"), good("0.87"), "0.78"],
    ["Refused against answered, among the prompts that are not harmful", bad("0.45"), bad("0.47"), "0.62"],
  ], ML, 1.85, CW, [6.4, 1.9, 1.9, 1.933], { size: 13.5, rowH: 0.6 });
  bullets(s, [
    "Hold refusal fixed, and the neuron still tells harmful from not harmful. Hold harm fixed at \"not harmful\", and it cannot tell the prompts the model refuses from the ones it answers.",
    "The model's refusals that are not about harm come with the neuron at its harmless level.",
    "Yet forcing the neuron still overrides the decision: on the Qwen models the attack pin also removes refusals that have nothing to do with harm (\"humanizing\" requests: 58% refused → 5%).",
  ], ML, 4.4, CW, 1.6, 12.5);
  result(s, [b("Why it matters now: "), t("for \"kinds of harm\" the neuron should be read as a harm detector that the refusal decision listens to. Not yet repeated on the omni model.")], { y: 6.05 });
  foot(s, "Prompts: CoCoNot (five categories, 55 to 60 each) and XSTest (200 unsafe, 250 safe). Refused = Qwen3Guard's refusal label on a 96-token answer.");
}

// ------------------------------------------------------------------ 22. what can be finished
{
  const s = newSlide("What can be finished, and by when", "next",
    "In the story's priority order. The first three rows were started on Saturday evening and have a slide waiting for their numbers.");
  table(s, [
    ["Experiment", "Part of the story", "State", "Ready"],
    ["Gemma 4 E4B (a non-Qwen omni model): the search in 14 languages and the vote (slide 16)", "Across omni models", { text: "Setting up", options: { fill: { color: AMBER_SOFT } } }, "Sunday, if it runs on our cards"],
    ["Larger pins on MultiJail and PolyRefuse (slide 13)", "Kinds of harm", good("Done"), "In this deck"],
    ["The picked neuron pinned in the other five MultiJail languages (slides 10, 14)", "Kinds of harm", { text: "Running", options: { fill: { color: AMBER_SOFT } } }, "Sunday"],
    ["Each new model: the picked neuron pinned in every language; the guard table", "Across omni models", "After its search", "About 2 days per model"],
    ["Qwen2.5-Omni-3B: the same search (slide 16)", "Across omni models", { text: "Queued, lower priority", options: { fill: { color: AMBER_SOFT } } }, "Sunday or Monday"],
    ["Bengali and Javanese: pin sweeps on their own top five", "Kinds of harm", { text: "Queued", options: { fill: { color: AMBER_SOFT } } }, "Monday"],
    ["Results split by harm category; AdvBench as a third dataset", "Kinds of harm", "Not started", "This week"],
    ["The languages as speech", "Language and modality agnostic", "Needs the speech pipeline", "To agree"],
    ["Concept induction across languages", "Last in the story", "Not started", "After the above"],
  ], ML, 1.5, CW, [6.0, 2.6, 2.0, 1.533], { size: 12, rowH: 0.44, left: [1, 2, 3] });
  result(s, [b("The paper draft "), t("on Overleaf follows this story and already holds every result above, with what is missing marked as pending. Submission: 4 January.")], { y: 6.1 });
}

// ------------------------------------------------------------------ 23. to decide
{
  const s = newSlide("To decide today", "next", "Each of these blocks a set of final runs.");
  const cols = [
    { h: "For the final runs", c: AMBER, body: [
      "Which two or three omni models go in the paper. Proposed: one non-Qwen model of the same size first (Gemma 4 E4B).",
      "One setup for both halves: the system prompt, the rule for picking the pin, the answer length (slide 20).",
    ] },
    { h: "For the claims", c: TEAL, body: [
      "The rule for naming the shared neuron: the vote over shortlists, with the exceptions named (slides 5, 6, 9).",
      "What counts as the vision result: pictures of typed text, real images, or none.",
      "Whether the three text-only models stay in the paper (slides 21, 22).",
    ] },
    { h: "For working together", c: SLATE, body: [
      "Where everyone can read the outputs and run their own analysis.",
      "Who makes the languages as speech.",
    ] },
  ];
  const w = (CW - 0.5) / 3;
  cols.forEach((c, i) => {
    const x = ML + i * (w + 0.25);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w, h: 0.08, fill: { color: c.c }, line: { color: c.c } });
    s.addText(c.h, { x, y: 1.85, w, h: 0.5, fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0 });
    bullets(s, c.body, x, 2.45, w, 4.3, 14);
  });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
