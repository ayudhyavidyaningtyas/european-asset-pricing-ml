# Chapter 5 Format Benchmark Against Past Dissertations

**Purpose.** This note benchmarks `docs/CHAPTER5_DRAFT.md` against the three
past MSc dissertation examples supplied by the user. It focuses on chapter
format, evidence ordering, table/figure presentation and what should change in
the next Chapter 5 pass.

**Source PDFs.**

- `/Users/ayudhya/Downloads/2022-23 Dissertation Example 2 (1).pdf`
- `/Users/ayudhya/Downloads/2022-23 Dissertation Example 4 (1).pdf`
- `/Users/ayudhya/Downloads/2022-23 Dissertation Example 3 (1).pdf`

---

## 1. Which Example Is The Closest Format Benchmark?

**Primary benchmark: Example 4.**

Example 4 is the closest format benchmark for this dissertation's Chapter 5
because it is a table-heavy empirical finance chapter. Its analysis chapter is
organised as a sequence of diagnostic and regression sections:

- 4.1 Descriptive Statistics
- 4.2 Multicollinearity Test and Correlation Analysis
- 4.3 Heteroskedasticity Test
- 4.4 Regression Analysis
- 4.5 Heterogeneity Analysis
- 4.6 Discussion of Findings

The useful pattern is:

1. Introduce the empirical test in one short paragraph.
2. Present one labelled table.
3. Interpret the table immediately in prose.
4. Move to the next test.
5. End with a discussion/summary section that synthesises findings rather than
   repeating every number.

This is the best match for Chapter 5 because our results chapter is also a
sequence of empirical tests: model comparison, spanning, capacity gradients,
data-depth interaction, mechanism tests, implementability and US benchmark.

**Secondary benchmark: Example 2.**

Example 2 is useful for its compact "Results and Discussions" style. Its
Chapter 5 begins with a setup paragraph, then shows a numbered table with the
title above, notes below and interpretation immediately after. This is a strong
model for individual subsections in our Chapter 5.

**Tertiary benchmark: Example 3.**

Example 3 separates "Result" and "Discussion" into different chapters and uses
third-level subsections for individual metrics such as accuracy, ROC AUC,
precision/recall and SHAP values. That format is useful for model-performance
reporting, but less suitable for our dissertation because each empirical test
requires its own caveats and economic interpretation. For our Chapter 5, keep
results and interpretation integrated rather than splitting all discussion into
Chapter 6.

---

## 2. Front-Matter And Dissertation-Level Format

All three examples use a conventional UCL-style dissertation package:

- Cover/submission page.
- Title page.
- Abstract.
- Acknowledgements.
- Table of contents.
- Often list of tables and list of figures.
- Numbered main chapters.
- References and appendices.

Format implication for this project:

- `docs/CHAPTER5_DRAFT.md` should remain a chapter draft, not a full
  dissertation file.
- The final assembled dissertation should include a list of tables and figures,
  because Chapter 5 is table-heavy.
- The provenance notes in `docs/CHAPTER5_DRAFT.md` are useful during drafting
  but should move to an appendix, footnote system or separate internal note
  before submission.

---

## 3. Chapter Title And Section Numbering

The examples use simple chapter titles:

- Example 2: `5 Results and Discussions`
- Example 4: `Chapter 4: Analysis and Discussion`
- Example 3: `5. Result`

Recommended final title for this dissertation:

> **Chapter 5: Empirical Results and Discussion**

The current title in `docs/CHAPTER5_DRAFT.md` is:

> Chapter 5 - Empirical Results: Characteristic Breadth, Model Depth And
> Implementability

That title is informative for drafting, but it is long compared with the past
dissertations. In the final dissertation, use the shorter chapter title above
and let the first paragraph state the chapter's conceptual frame.

Recommended section numbering:

- 5.1 Empirical Design and Common Sample
- 5.2 Model Comparison
- 5.3 Incremental Value of Complexity
- 5.4 Capacity Barrier
- 5.5 Analyst Data Depth and Model Depth
- 5.6 Forecast-Error Mechanism
- 5.7 Implementability
- 5.8 Matched US Benchmark
- 5.9 Discussion of Findings

The current draft is already close. The only format adjustment is to keep
headings short and avoid over-descriptive titles where possible.

---

## 4. Table Format

The past dissertations consistently put table titles above tables. Example 4
uses the clearest empirical-table format:

- Table number and title centred above the table.
- Thick horizontal rule above and below the table body.
- Column labels in bold.
- No vertical gridlines.
- Short prose before and after the table.

Example 2 also uses a helpful note below the table to define abbreviations and
units. This is directly relevant for Chapter 5 because many tables use IC,
Sharpe, HAC, Holm, DRE, HistGBM and basis-point terminology.

Recommended table rules for Chapter 5:

- Keep table titles above the table.
- Add table notes below when abbreviations, units or inference choices matter.
- Round main-text numbers more aggressively than the working notes:
  - ICs: 3 or 4 decimals.
  - Annual returns/alphas: percentages with 1 or 2 decimals, or decimals with 3
    places, but be consistent.
  - p-values: 3 decimals unless very small, then use scientific notation.
  - Confidence intervals: 2 or 3 decimals, depending on scale.
- Do not include columns that say `not shown`, `not needed` or placeholders.
- Wide tables should move to the appendix, or be split into two panels.
- Use the main text for headline rows only; put full robustness grids in an
  appendix.

Current Chapter 5 draft status:

- The draft follows the examples by introducing, showing and interpreting each
  table.
- The draft has too many exact six-decimal numbers for final dissertation prose.
- The draft includes 13 tables, which is acceptable for a results chapter only
  if several robustness-heavy tables are moved to the appendix.

Recommended main-text table set:

1. Common-sample rank ICs.
2. Selected paired IC tests.
3. Complexity spanning ladder.
4. Capacity gradients: breadth versus momentum.
5. Data-depth effects and interaction.
6. Forecast-error mechanism summary.
7. Constrained implementation summary.
8. Matched Europe-US benchmark.

Move or compress:

- Detailed market-cap bucket examples.
- Depth-gradient table versus ridge.
- Benchmark-relative constrained-performance table.
- US-minus-Europe Sharpe-difference table.

These can remain in the internal Markdown draft but should be considered
appendix material in the final document.

---

## 5. Figure Format

The examples use figures sparingly. Example 3 places a model-performance figure
immediately after the relevant subsection heading; Example 4 lists figures in
front matter and uses figure references in analysis.

Recommended Chapter 5 figures:

- Capacity-barrier figure:
  `figures/manuscript/capacity_barrier_flexibility_premium.png`
- Capacity-gradient premia:
  `figures/manuscript/capacity_gradient_premia.png`
- US-Europe rank IC/Sharpe figure if available from the manuscript figure
  builder.

Figure rules:

- Place each figure soon after the paragraph that first interprets it.
- Caption below the figure.
- Do not use figures as decoration; every figure should carry a specific
  empirical point.
- Keep detailed source paths out of captions. Use captions such as "Source:
  author's calculations from Refinitiv, Compustat and LSEG analyst estimates."

---

## 6. Paragraph And Page Style

Observed style across the examples:

- A4 page size.
- Standard Word thesis layout.
- Body text usually 11 or 12 point.
- Wide margins, page number in footer.
- Chapter headings in larger bold type.
- Tables centered on the page.
- Most empirical sections follow a short setup -> table/figure -> explanation
  rhythm.

Recommended style for final Chapter 5:

- Use 12-point body text and 1.5 line spacing unless the programme template
  requires otherwise.
- Use justified or left-aligned body text consistently across the dissertation.
- Avoid very long paragraphs; target 120-180 words for most empirical
  paragraphs.
- Avoid code-style provenance paths in the main chapter.
- Use table notes and appendix references instead of bracketed result-file
  citations in the final submitted chapter.

Current draft status:

- The prose style is stronger and more academically controlled than the
  examples.
- The current Markdown is more detailed than the example chapters. That is
  useful for drafting, but final formatting should be cleaner and less
  provenance-heavy.

---

## 7. Recommended Chapter 5 Format After Benchmarking

The final Chapter 5 should use Example 4's table-heavy empirical structure,
Example 2's table-note discipline and Example 3's clear model-performance
subsectioning only where needed.

Recommended final rhythm for each subsection:

1. **Purpose paragraph.** Explain what the test identifies.
2. **Design sentence.** State sample, model cell and inference method.
3. **Table or figure.** One main exhibit per subsection where possible.
4. **Interpretation paragraph.** State the result in economic language.
5. **Caveat paragraph.** State what the result does not prove.

Applied to the current Chapter 5:

- 5.1 should stay mostly prose.
- 5.2 should keep Tables 5.1 and 5.2.
- 5.3 should keep the spanning ladder table.
- 5.4 should keep one capacity-gradient table and one figure; move extra bucket
  and depth-gradient detail to appendix.
- 5.5 should merge data-depth effects and interaction into one compact table.
- 5.6 should keep one mechanism summary table; put joint/specificity details in
  appendix or prose.
- 5.7 should keep one implementation table and discuss benchmark-relative
  results in prose.
- 5.8 should keep one Europe-US comparison table and discuss Sharpe-difference
  inference in prose.
- 5.9 should be a synthesis, not a numbered repetition of every table.

---

## 8. Specific Edits To Make To `docs/CHAPTER5_DRAFT.md`

High-priority format edits:

1. Rename the displayed chapter title to `Chapter 5: Empirical Results and
   Discussion`.
2. Move the italic opening draft note, provenance notes and revision notes out
   of the final chapter body before submission.
3. Reduce table count from 13 to about 8 main-text tables.
4. Add formal table notes under tables that define abbreviations and inference.
5. Round all main-text numerical values to final-display precision.
6. Add references to the capacity and US benchmark figures in the relevant
   subsections.
7. Keep the updated US caveat: the US analyst-estimates files exist, but the
   current evidence is EUR-currency interim pipeline validation, not a final
   USD-symmetric US Test B.

Medium-priority format edits:

1. Convert Markdown tables to final Word/LaTeX table style once the dissertation
   container is chosen.
2. Decide whether Chapter 5 should include a short "Discussion of Findings"
   subsection or whether synthesis should remain in 5.9.
3. Move code paths from prose into hidden comments, footnotes or an appendix.
4. Standardise all model names: `ElasticNet` versus `elastic net`,
   `HistGBM`, `DRE`, `MLP`.

Do not copy from the examples:

- Do not adopt Example 2's very short results chapter; this dissertation has
  more empirical layers.
- Do not adopt Example 3's split between Results and Discussion for this
  chapter; immediate interpretation is clearer for this project.
- Do not follow any example's tendency to overstate p-values mechanically. This
  dissertation should retain the stricter Holm-adjusted language.

---

## 9. Bottom Line

The current Chapter 5 draft is directionally aligned with the past dissertation
format: it is numbered, table-led and interpretive. The main difference is that
it is still an internal research draft, with too many exact numbers, source
paths and auxiliary tables for a final submitted chapter.

The best final format is:

> Example 4's empirical-table chapter structure, Example 2's table-note style,
> and the current Chapter 5 draft's stronger econometric caveats.
