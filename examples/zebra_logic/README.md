# FOLIO Case Study

This case study evaluates some Delphyne strategies on FOLIO, a benchmark of natural-language first-order logic problems. The main result is that our best custom strategy outperforms the direct baselines: our `Aggregate` strategy reaches **87.8% mean accuracy** across three seeds, compared with **81.6%** for direct question answering baseline and **81.1%** for the baseline agent given access to a formalization tool that internally uses Z3. All of them were run with `gpt-5-nano-2025-08-07` on `low` reasoning effort with a budget of `$0.01` per problem. 

The experiment uses 134 problems from the validation set of FOLIO and repeats each configuration with three seeds. Reported scores below are mean correct answers with standard deviations.

## Results

### Custom strategies

| Strategy | Configuration | Mean correct +- std | Accuracy |
| --- | --- | ---: | ---: |
| Aggregate | reflection + favor-pos. | 117.7 +- 1.5 | 87.8% |
| Aggregate | favor pos. (no reflection) | 116.3 +- 1.5 | 86.8% |
| Aggregate | reflection + maj vote | 114.0 +- 2.0 | 85.1% |
| One-shot | with reflection | 110.0 +- 3.6 | 82.1% |
| Iterative | with reflection | 109.7 +- 2.5 | 81.8% |

### Baseline strategies

| Strategy | Mean correct +- std | Accuracy |
| --- | ---: | ---: |
| Only Ask | 109.3 +- 2.1 | 81.6% |
| Formalization Agent | 108.7 +- 4.0 | 81.1% |
| Z3 Agent | 105.7 +- 1.5 | 78.9% |


## Details

A problem from the FOLIO benchmark consists of around 5 to 10 premise sentences and a single conclusion sentence. The task is to find whether the conclusion follows from the premises.

### Baseline strategies

The baseline agents test straightforward ways of using an LLM to solve FOLIO problems. 

- **Only Ask**: Requests a direct yes/no answer from the model.

- **Formalization Agent**: A conversational agent equipped with a tool that can be called with a proposed first-order logic formalization of the problem (in a fixed formalization language). This is then translated and run in Z3 to check the satisfiablility of the premises together with the negation of the conclusion. The result (which also contains a found satisfying assignemnt or unsat-core by Z3) is fed back to the model. It can choose to continue refining its answer within the same interaction loop.

- **Z3 Agent**: A conversational agent equipped with a tool that can be called to run expressions in the Z3 Python API directly, restricted to a whitelisted subset of Z3 constructors and operations sufficient for formalizing FOLIO. The model can choose to continue refining its answer within the same interaction loop.

### Custom strategies

The custom agents put more structure around formalization. They can be thought of as more advanced versions of the baseline `Formalization Agent`.

- **One-shot**: In an interaction loop, the model is asked to formalize the whole problem. Then, Z3 checks the formalization, both for consistency of the premises and whether the conclusion follows from the premises. If `reflection` flag is enabled (which can be done on the policy-side), in a subsequent interaction loop the model is asked to refine the formalization taking into account the found countermodel by Z3 (in case the conclusion is not proved). The new formalization is checked not to be spurious in order to prevent the model from "cheating" e.g by artificially adding premises as conclusions.


- **Aggregate** strategy collects results from several `One-shot` strategies. How many `One-shot` strategies should be run and with which configuration can be specified on policy-side. Depending on another policy-side `aggregation_type` flag, the collected results can either be combined by majority vote, by favoring any positive result (i.e. "answer 'yes' if at least one run finds that the conclusion follows from the premises"), or by issuing an LLM query to judge among the candidate formalizations. We used 3 `One-shot` runs in the experiment.


- **Iterative** strategy first asks the model to fix a shared predicate vocabulary, then splits the problem into parts and asks the model to formalize them one by one using that vocabulary. In case of failure, each part can be retried with a blacklist of previously failed attempts. After a certain number of failed attempts (syntax errors or inconsistent premises), a backtrack happens to the previous part. At each step, more than one formalization can be sampled and majority-voted using Z3 equivalence checks. This happens on the policy-side via the `majority_vote` stream transformer. After the full formalization is assembled, Z3 checks whether the conclusion follows and optionally a `reflection` happens with a provided countermodel if the conclusion is not proved, just like in the `One-shot` strategy.


## Qualitative examples

We sampled a few cases where one custom strategy succeeds and all the others fail.

The **Aggregate** strategy (with reflection and favor-positive rule) makes a real difference when one independent `One-shot` run is able to find the right formalization after reflection. There are indeed problems, where only one of the three runs (after reflection) found a proof that the conclusion follows from the premises, and the aggregation rule preserved that success. Typical fixes in the reflection phase of these problems introduced small semantic links that were missing from the initial formalization: For examplle, `Equals(LastSummerOlympics, Tokyo)` to connect the event name to its location, `ForAll([x, y], Implies(And(Writes(x, y), MusicPiece(y)), Composer(x)))` to connect writing music pieces with being a composer, and `ForAll(x, Implies(BornInMultipleBirthWithSiblings(x), BornInMultipleBirth(x)))` to connect two closely related predicates. These are probably those kind of repairs that the countermodel from Z3 makes visible.


The **Iterative** strategy tends to sometimes help when the problem benefits from committing to a shared vocabulary in a separate initial step. In one sampled problem, fixing predicates up front made the strategy use a `NotTidy` predicate consistently, rather than inventing a separate `Tidy` predicate with no connection to it. Moreover, majority voting can filter out bad local formalizations. For example, in the same problem two samples selected the useful rule `ForAll(x, Implies(Cleanly(x), Not(NotTidy(x))))`, while a third sample produced the vacuous rule `ForAll(x, Implies(Cleanly(x), Cleanly(x)))`.


## Takeaway

The direct tool-using baselines were only as good as directly asking the model for a yes/no answer. The best result came from adding some structure around the model: running multiple independent formalization attempts, using Z3 feedback to filter and refine them, and combine the outcomes with a simple rule that preserves successful proofs when they appear.

This case study suggests that Delphyne is most useful when it is used not just as a wrapper for tool-calling conversational agents, but as a way to modularly implement feedback loops with retries, reflection, and aggregation around LLM answers. Moreoever, it enables rigorously testing implemented approaches with reproducibility.


## Project Structure

- `folio_standard.py`: The baseline strategies `folio_only_ask`, `folio_formalization_agent` and `folio_z3_agent` that use `interact` to implement basic conversational agents.

- `folio_oneshot.py`: The `folio_oneshot` strategy is implemented using `interact`. It calls sub-strategies `reflect` and `check_constraints`. The latter is used throughout the whole project to issue Z3 calls using the `Compute` effect. The reflection and formalization style can be controlled through flags.

- `folio_aggregate.py`: It defines an `Aggregation` effect that is used to extract all solutions from an opaque space. The `folio_aggregate` strategy uses that effect. The `aggregation_type` that is used can be controlled through flags.

- `folio_iterative.py`: The `folio_iterative_blacklist` strategy formalizes a problem part by part. Each part is fed into a sub-strategy `formalize_single_blacklist` that is wrapped inside a `branch` and `iterate` effect. That means, every time it is encountered it is run with information from the previously failed attempts of that part. On policy-side, `majority_vote` stream transformer collects all solutions at each part of the problem and the equivalance of candidate formalizations is established via Z3 before the majority voting. 
