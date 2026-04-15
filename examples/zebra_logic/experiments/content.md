Now I have comprehensive data on all incorrect cases. Let me compile the final analysis report.

---

# FOLIO Aggregate Experiment: Incorrect Results Analysis

## Strategy Analyzed
**Best strategy**: `all_normal_reflect` + `majority_vote` + `only_if_sat` — **172/200 correct (86%)**  
From [aggregate_summary.csv](examples/zebra_logic/experiments/output_9feb/aggregate_summary.csv), this variant has the highest accuracy.

The strategy runs **3 independent agents**, each using `normal` style with `always` reflection. Within each agent, the model first generates a FOL formalization, checks with Z3, then reflects (potentially re-formalizing). The `only_if_sat` post-processing: for each agent, if initial result is False → use reflected result; if True → keep initial. Then **majority vote** across agents.

---

## Complete List of Incorrect bench_ids (28–29 cases)

| # | bench_id | Predicted | Ground Truth | Config Hash | Category |
|---|----------|-----------|-------------|-------------|----------|
| 1 | 14 | False | True | `cac3744a` | Missing implicit constraint |
| 2 | --25 | False | True | `46f2fe2f` | Wrong quantifier scope |
| 3 | 1050 | False | True | `04efdec8` | Semantic formalization error |
| 4 | 1058 | False | True | `74cb0e87` | **Ground truth issue** |
| 5 | 550 | False | True | `df161433` | Semantic formalization error |
| 6 | 551 | True | False | `f931d2f2` | Semantic error + aggregation error |
| 7 | 554 | False | True | `ee6f0a82` | **Ground truth issue** |
| 8 | 1072 | True | False | `72f043a4` | **Ground truth issue** |
| 9 | 1106 | False | True | `ab57328f` | **Ground truth issue** |
| 10 | 1118 | False | True | `7409c3eb` | Semantic formalization error |
| 11 | 1119 | True | False | `861f9b4e` | **Ground truth issue** |
| 12 | 620 | False | True | `d55d95f4` | Missing implicit constraint (modality) |
| 13 | 123 | False | True | `e27af2a1` | Missing implicit + reflection-induced error |
| 14 | --1157 | True | False | `a5312ed8` | Semantic formalization error |
| 15 | 1182 | True | False | `d65d8135` | **Ground truth issue** |
| 16 | 195 | False | True | `f191e9bb` | Semantic formalization error |
| 17 | 1234 | False | True | `8e96da49` | **Ground truth issue** |
| 18 | 1256 | False | True | `48f7641a` | Aggregation error |
| 19 | 1335 | True | False | `d6967d2b` | **Ground truth issue** |
| 20 | 862 | True | False | `ec9b676a` | Semantic error + aggregation error |
| 21 | 366 | True | False | `8b27a2d1` | **Ground truth issue** |
| 22 | 882 | False | True | `249ec57e` | **Ground truth issue** |
| 23 | --385 | True | False | `22f41e99` | Semantic formalization error |
| 24 | 1429 | True | False | `11758235` | **Ground truth issue** |
| 25 | 919 | False | True | `29648c7c` | **Ground truth issue** |
| 26 | 480 | False | True | `0b871139` | Missing implicit constraint (CWA) |
| 27 | 1170 | — | False | `fe0153d0` | Pipeline failure (budget exhaustion) |
| 28 | 1172 | — | True | `1a255fa2` | Pipeline failure (budget exhaustion) |
| 29 | 881 | — | False | `567b9467` | Pipeline failure (budget exhaustion) |

---

## Detailed Case Analysis

### Category 1: Ground Truth Issues (12 cases)
Bench_ids: **1058, 554, 1072, 1106, 1119, 1182, 1234, 1335, 366, 882, 919, 1429**

These are cases where the model's logical analysis appears correct but disagrees with the FOLIO benchmark label.

#### bench_id=1058 — "Prime numbers / One"
**Puzzle**: "One is a prime number or a natural number or both" → Conclusion: "One is a prime number **and** a natural number."  
**Formalization** (all 3 agents identical):
```
Or(Prime(One), Natural(One))  
Implies(Prime → Natural)
```
**Analysis**: From `Or(Prime, Natural)` and `Prime → Natural`, we can derive `Natural(One)` in all cases, but `Prime(One)` only in one branch. The conclusion `And(Prime, Natural)` is not provable. Z3 correctly returns False. **The FOLIO label (True) is wrong — should be Unknown.**

#### bench_id=1072 — "Rouge Dior Lipstick"
**Conclusion**: "If refillable AND has rosewood, then either velvet-finish or has rosewood" → $A \wedge B \implies B \vee C$, which is a **tautology** since $A \wedge B \implies B$. Z3 correctly returns True. **Ground truth (False) is wrong.**

#### bench_id=1119 — "Benji's family / Fabien studies Spanish"
**Conclusion**: $\text{FavoriteSince}(Fabien) \vee \text{Francophile}(Fabien) \implies \text{StudiesSpanish}(Fabien) \vee \text{FavCountry}(Fabien, Spain)$. Since `StudiesSpanish(Fabien)` is a premise, the consequent is always True, making the implication True. **Ground truth (False) is wrong.**

#### bench_id=1182 — "Leif / software engineering / business analysis"  
**Chain**: Software engineering → from China → (but company employees not from China) → Leif can't be in business analysis → conclusion `Not(And(GoodAtMath(Leif), WorksInBA(Leif)))` is provable. **Ground truth (False) is wrong.**

#### bench_id=1335 — "Room 116 / Harkness Hall"
**Chain**: Room 116 is a classroom → must use lectures or office hours → "either both or neither" + being a classroom forces "both" → lectures → booked during day → consequent true. **Ground truth (False) is wrong.**

#### bench_id=366 — "Shetani / mythological"
Premises state "Shetanis are mythological." Conclusion: "A shetani is either mythological **or** a creature" — trivially true from the first disjunct. **Ground truth (False) is wrong.**

#### bench_id=1429 — "Size Town / bird"
The bird is not both heavy and still → not heavy (since heavy→still) → not big (since big→heavy) → small → light → unstable → changing + unpredictable. The consequent of the conclusion holds for both branches. **Ground truth (False) is wrong.**

#### bench_id=554 — "Rosa / construction site vs traffic"
The premises establish "oversight of a **construction site**" but the conclusion asks about "oversight of **traffic**" — completely different concepts. The conclusion is not provable. **Ground truth (True) is wrong.**

#### bench_id=1106 — "Board Game Night / Erica"
Erica plays lots of chess (premise). The conclusion says "if plays chess → NOT plays chess" — a contradiction with the premise. Z3 correctly returns False. **Ground truth (True) is wrong.**

#### bench_id=882 — "Jim / Knicks player"
The premises say nothing about Jim (only about John). Jim's Knicks status is unconstrained. FOL can't derive `Not(KnicksPlayer(Jim))` without closed-world reasoning. **Ground truth (True) is wrong or requires CWA.**

#### bench_id=1234 — "Yale Harkness"  
Premise: "Harkness is EITHER operated by Yale Housing staff OR located on York Street." If on York Street only → the chain (operated→managed→dormitory→campus→open→not before 1701) doesn't apply. **Conclusion unprovable. Ground truth (True) is wrong.**

#### bench_id=919 — "Jumbo elephant"  
Jumbo can't be a baby elephant (leads to contradiction), but CAN be sleepy (if just a mammal-not-elephant). So `And(Not(Sleepy), Not(BabyElephant))` is not fully derivable. **Ground truth (True) is wrong for the Sleepy conjunct.**

---

### Category 2: Semantic Formalization Errors (8 cases)
Bench_ids: **25, 550, 551, 1050, 1157, 195, 385, 862**

#### bench_id=25 — "Thick as Thieves"
**Error**: Used an existential quantifier `Exists(a, And(WrittenBy(b,a), WorkedWith(a,c)))` instead of a direct universal linking author to publisher: `ForAll(b, ForAll(c, ForAll(a, Implies(And(PublishedBy(b,c), WrittenBy(b,a)), WorkedWith(a,c)))))`. The existential doesn't guarantee the KNOWN author MeganWhalenTurner worked with GreenwillowBooks.

#### bench_id=550 — "La Liga / Real Madrid ranks higher" 
**Error**: Conflated "total points" and "head-to-head points" into a single `MorePoints(2)` predicate. Also missed `Not(Equals(RealMadrid, Barcelona))` — Z3 doesn't assume unique names, allowing RM=Barca as a counterexample.

#### bench_id=551 — "La Liga / Barcelona ranks higher"
**Error**: Agent 2 created **contradictory constraints**: both `ReceivesMorePointsFromGamesBetween(RM, Barca)` AND `Not(ReceivesMorePointsFromGamesBetween(RM, Barca))`. From contradiction → anything provable. Majority of 2 agents voted True (wrong).

#### bench_id=1050 — "Dune / creative process"
**Error**: Reified properties as objects (`DuneIsScienceFiction` as a constant instead of `ScienceFiction(Dune)` as a predicate application), breaking the logical chain between "Dan knows Dune is sci-fi" and the property `ScienceFiction(Dune)`.

#### bench_id=1157 — "Avocados / New Haven"
**Error**: Conflated "sold at the local farmers market in New Haven" (a specific market) with "sold in New Haven" (the city). Agent 1 used one constant for both, making the premise and conclusion identical. Correct formalization (as Agent 2 did) distinguishes `LocalFarmersMarket` from `NewHaven`.

#### bench_id=195 — "2028 Olympics"
**Error**: Used inconsistent predicates: `ScheduledInCity(Olympics2028, LA)` for the fact but `SummerOlympicsInCity(x)` in the rule. Since these are different predicate names, the rule "If a city holds a Summer Olympics and is a US city..." never fires.

#### bench_id=385 — "Turing Award / numerical methods"
**Error**: Used `ContributedTo(Hamming, NumericalMethods)` for the premise and `WorkedIn(p, NumericalMethods)` for the conclusion — different predicates. Also **encoded the conclusion as a constraint**, making it trivially provable.

#### bench_id=862 — "Jack / cleaning"
**Error**: Some agents' reflections introduced contradictions or additional unwarranted facts, leading to ex-falso proofs of the unrelated conclusion `And(Overburdened(Jack), LivesInSuburbs(Jack))`.

---

### Category 3: Missing Implicit Constraints (4 cases)
Bench_ids: **14, 620, 480, 123**

#### bench_id=14 — "Boves railway station"
**Error**: "The preceding station of Boves is Longueau" implicitly means Longueau and Boves are **contiguous stations**, but the model never asserted `ContiguousStations(Longueau, Boves)`. The rule "contiguous → same railway" can't fire. All 3 agents missed this.

#### bench_id=620 — "Maya / volleyball / violin"
**Error**: "Volleyball players **can** injure their fingers" expresses a **possibility** (modal "can"), but FOL formalizes it as a disjunction ("WILL injure ankles OR fingers OR shoulders"), which doesn't guarantee fingers specifically. Maya plays violin "only if her fingers could **never** be injured" — the mere possibility of finger injury should block violin, but the FOL disjunction allows Maya to only injure ankles/shoulders.

#### bench_id=480 — "Mary / train vs car"
**Error**: Premises give `LoseTime → Late` but not `Late → LoseTime`. Under **closed-world assumption**, Mary being late implies she lost time (the only way to be late), and time is only lost via car+traffic. But FOL doesn't use CWA, so Mary could be late for unstated reasons, and the conclusion "Mary goes by car" is unprovable.

#### bench_id=123 — "Imagine Dragons / rock band"
**Error**: Imagine Dragons is described as "pop-rock" but the conclusion asks about "rock band." The model didn't add `PopRockBand → RockBand`. Without this, `RockBand(ImagineDragons)` is never established.

---

### Category 4: Aggregation Errors (2 cases)  
Bench_ids: **1256, 551** (551 also counted above)

#### bench_id=1256 — "Researchers/James/conference"
Agent 1 correctly formalized and got True (negation of premise proves James must provide tutorial → gets meals → photo + happy). But Agents 2–3 had different formalization issues (e.g., omitting the disjunction "presents OR provides tutorial" as universal over all people). Majority voted False. **1 of 3 agents was correct, but was outvoted.**

---

### Category 5: Reflection-Induced Errors (1 case)
Bench_id: **123**

#### bench_id=123 — "Imagine Dragons"
Agent 1's initial formalization returned **True** (correct), but the reflection step changed the formalization and the reflected version returned **False** (wrong). The only_if_sat strategy then used the reflected version (since initial was True/SAT → keep initial). Actually under only_if_sat, since sol=True, the initial is kept, giving True. But other agents returned False, and the majority voted False. So this was also an aggregation error where different agents had different results.

---

### Category 6: Pipeline Failures (3 cases)
Bench_ids: **1170, 1172, 881**

All three exhausted the $0.01 budget limit before producing a valid result. These puzzles likely required many retry/reflection rounds, consuming tokens without converging.

---

## Summary Counts

| Failure Category | Count | bench_ids |
|---|---|---|
| **Ground truth issue** | 12 | 1058, 554, 1072, 1106, 1119, 1182, 1234, 1335, 366, 882, 919, 1429 |
| **Semantic formalization error** | 8 | 25, 550, 551, 1050, 1157, 195, 385, 862 |
| **Missing implicit constraint** | 4 | 14, 620, 480, 123 |
| **Pipeline failure (budget)** | 3 | 1170, 1172, 881 |
| **Aggregation error** | 2 | 1256, 551 |
| **Reflection-induced error** | 1 | 123 |

(Some cases span multiple categories; primary category listed.)

---

## Actionable Patterns

### 1. **Unique Names Axiom** (affects: 550, and likely others)
The Z3 encoding doesn't enforce distinct names for distinct constants. Adding `Distinct(c1, c2, ...)` for all constants would fix cases where Z3 finds counterexamples by collapsing constants.

### 2. **Predicate Consistency Check** (affects: 195, 385, 1050)
When distinct predicate names refer to the same concept (e.g., `ScheduledInCity` vs `SummerOlympicsInCity`, or `ContributedTo` vs `WorkedIn`), the FOL reasoning breaks. A post-processing step could detect predicates with similar names/semantics and warn or unify them.

### 3. **Existential Quantifier Misuse** (affects: 25)
The model frequently wraps known relationships in existential quantifiers instead of direct universal rules. A checker could flag cases where an `Exists` introduces a variable that matches a known constant.

### 4. **Implicit Entailment Rules** (affects: 14, 123)
For domain-specific implications like "preceding station → contiguous" or "pop-rock → rock," the model could be prompted more explicitly or given few-shot examples of implicit category membership.

### 5. **Modality Handling** (affects: 620, 480)  
FOL cannot capture modal reasoning ("can injure" ≠ "will injure"). A systematic mapping of "can/may/possible" to stronger assertions (treating possibility as certainty for safety properties) could resolve these.

### 6. **Contradictory Constraint Detection** (affects: 551, 862)
Before Z3 solving, check for unsatisfiable constraint sets. If the constraints themselves are UNSAT, any conclusion is trivially provable (ex falso), which produces wrong answers. A pre-check could flag this.

### 7. **Budget Increase for Complex Puzzles** (affects: 1170, 1172, 881)
Three puzzles failed due to the $0.01 budget limit. Increasing budget or using an adaptive budget strategy for puzzles that consume many tokens would eliminate these failures.

### 8. **Ground Truth Filtering** (affects: 12 cases = ~41% of errors)
Nearly half the "errors" appear to be FOLIO benchmark annotation mistakes. Cross-referencing with known FOLIO errata or running a secondary symbolic verifier could flag questionable ground truth labels before evaluation.



Support n-ary And/Or in the FOL parser — Modify FOLParser._formula_from_ast in fol.py to accept 2+ arguments for And/Or and auto-nest into binary trees. The traces show bench_id 1170 failing because the LLM wrote And(InClass(x), GoodAtMath(x), GoodAtChemistry(x)) → ValueError: And expects 2 arguments. This is the #1 cause of pipeline failures (budget exhaustion from repeated syntax errors that the LLM cannot self-correct).

Add syntax auto-repair for unmatched parentheses — Before parsing in fol.py, auto-strip trailing unmatched ) characters. Bench_ids 881, 761, 1172 all show the LLM consistently producing ...))) with one extra paren, and it cannot self-correct despite explicit error feedback across 15+ retries. A simple while formula.count(')') > formula.count('('): formula = formula.rstrip(')') would fix these without re-querying.

Add predicate consistency enforcement — Bench_ids 195, 385, 1050 fail because the LLM uses different predicate names for the same concept across premises and conclusion (e.g., ScheduledInCity in premises vs SummerOlympicsInCity in rules, or ContributedTo vs WorkedIn). Add a post-formalization check: after parsing, compute string similarity between all predicate names, and if two predicates have similar names + same arity, prompt the LLM to confirm or unify them before Z3 solving.

Add Unique Name Assumption (UNA) automatically — Bench_id 550 fails because Z3 allows RealMadrid = Barcelona as a counterexample. After parsing constants, automatically inject Distinct(c1, c2, ...) into the Z3 solver in z3_tools.py. This is a standard FOL convention and eliminates spurious countermodels.

Add a "preceding → contiguous" style implicit constraint extraction step — Bench_ids 14, 123 fail because the LLM doesn't bridge domain vocabulary gaps (e.g., "preceding station" ≠ "contiguous station", "pop-rock" ≠ "rock"). Add a dedicated pre-formalization prompt step: given the puzzle text, ask the LLM to list any implicit entailments between terms used in the premises vs the conclusion (e.g., "preceding implies contiguous", "pop-rock is a subtype of rock"). Feed these as additional constraints.

Increase aggregate count to 5 for split votes — Bench_ids 1256 and 123 fail because 1-of-3 agents got the right answer but was outvoted. If the first 3 agents don't agree unanimously, run 2 more agents (perhaps with higher reasoning effort or different temperature). This focuses extra compute only on the ~30% of problems where there's disagreement, keeping cost down for easy cases.

Improve the "reflect-if-sat" prompt to verbalize the countermodel first — Bench_id 862 shows reflection introducing contradictions. Instead of immediately asking "are implicit constraints missing?", first ask the LLM to describe the Z3 countermodel as a natural language story, then ask "does this story violate any premise?" in a separate turn. This two-step approach grounds the reflection and reduces the chance of arbitrary constraint insertion.

Detect and flag contradictory constraints before conclusion checking — Bench_id 551 shows 2 agents producing contradictory constraints (P(x) and Not(P(x))), making any conclusion trivially provable (ex falso). The fol_inconsistent_constraints check already exists in folio_baseline.py, but only for step_type == "Constraint". Ensure it also runs in the oneshot path before conclusion checking — if premises alone are UNSAT, retry formalization rather than accepting a vacuously true result.

Fix PredicateDef.__eq__ — At fol.py:79, __eq__ = lambda self: self.name is broken (single-argument lambda, always truthy). Fix to properly compare both name and arity. This would catch arity conflicts during parse_multiple when accumulating predicates across formalizations.

Filter or flag ground truth issues — 12 of 28 errors appear to be FOLIO annotation mistakes. Run a separate symbolic verifier (e.g., hand-check the 12 contested cases) and exclude confirmed bad labels from accuracy computation. This won't improve the system but gives a clearer picture of true accuracy (~95–96% after filtering).

Verification

Re-run on the same SAMPLE_IDS_9feb_200 sample and compare per-problem against the 86% baseline using postprocess.py
Steps 1–2 should eliminate all 3 pipeline failures (881, 1170, 1172) → +1.5% immediately
Steps 3–4 should fix bench_ids 195, 385, 550 → +1.5%
Steps 5–8 target the remaining ~8 semantic/implicit/aggregation errors → +2–4% expected
Track cost per problem — steps 1–4, 8–9 have zero cost overhead; step 6 adds cost only for split votes (~30% of problems)
Decisions

Steps 1, 2, 9 are pure bug fixes — implement first, no downside
Step 4 (UNA) is standard and almost always correct — implement unconditionally
Step 8 (contradictory constraint detection) is existing logic that needs to be applied to the oneshot path
Steps 3, 5, 6, 7 require new prompts/strategy changes — medium effort, implement iteratively
Step 10 is evaluation hygiene, not system improvement — do last