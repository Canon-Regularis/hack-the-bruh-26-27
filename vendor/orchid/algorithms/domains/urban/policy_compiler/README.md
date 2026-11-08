# Policy Compiler

Policy text is the hard side of a street design. A traffic order says a road
through the park is prohibited, that Fitzwilliam Street may not be closed, that
a crossing must be built at a named junction, that the footway on Lower Baggot
Street must be between 3.0 and 4.0 metres. This module is the MOSAIC paper's
§5.5 made **deterministic**: reading is out of scope by design. An upstream
reader (a person, or a model whose output is treated as untrusted) proposes
**typed clauses** — a verbatim quote, a page, a named place and one ruling per
search operation — and everything from there is lookup, binding and
assignment:

1. the quote is verified against the source **character for character**
   (`source[start:end] == quote`), with offset recovery and a fuzzy fallback
   that is reported but never trusted;
2. the place is bound to network identifiers by rules that **refuse rather than
   guess** — and never invent an identifier;
3. the rulings are compiled into **fixings and bounds** on the search vector
   `x = (z, y, w, k)` that `urban/urban_optimisation` explores.

Anything that cannot be verified or bound unambiguously goes to an
**abstention queue** for a human. Conflicting clauses are withheld, not
arbitrated.

```python
from policy_compiler_template import (build_example_district, compile_policy,
    project_design, policy_check, relaxation_analysis)

ex = build_example_district()          # TRO + amendment, gazetteer, design space
res = compile_policy(ex.clauses, ex.gazetteer, ex.sources, at=ex.at,
                     instruments=ex.instruments, baseline_y=ex.baseline_y,
                     baseline_junctions=ex.baseline_junctions)
print(res.fixings.to_dict()["fixedZero"])     # ['c1', 'c2']  no road through the park
print(res.fixings.to_dict()["fixedOne"])      # ['e1', 'e2']  Fitzwilliam St protected
print([(i.clause_id, i.reason) for i in res.queue.items()])
# [('A08', 'binding_ambiguous'), ('A11', 'span_unverified')]

d = project_design(ex.space.baseline_design(), res.fixings, ex.space).design
print(policy_check(d, res).compliance_ratio)  # 1.0
```

This is deliberately not `urban/public_voice`, which turns consultation
comments into a *soft*, located concern field, and not `urban/urban_optimisation`,
which searches. This module owns what policy **prohibits, protects and bounds**,
when it is in force, and how a design stands against it clause by clause.
`urban_optimisation` consumes the `CompiledFixings` dict as plain JSON without
importing this file.

---

## The load-bearing constraint: never bind across a street type

The highest-risk failure is a **wrong binding** — a clause about Fitzwilliam
Street compiled onto Fitzwilliam Gate — because it is invisible in the output.
Whole-name similarity cannot prevent it:

| query | candidate | Jaro-Winkler |
| --- | --- | --- |
| Fitzwilliam Street | Fitzwilliam St (correct) | 0.9556 |
| Fitzwilliam Street | **Fitzwilliam Gate (wrong)** | **0.9306** |
| Fitzwilliam Street | Fitzalan Square | 0.8244 |

Any threshold low enough to accept `St` also accepts `Gate`. So
`split_street_name` splits every name into `(specific, thoroughfare type,
qualifiers)`, normalises the type through `THOROUGHFARE_TYPES` (32 classes,
`St → street`, `Sq → square`, leading `St → Saint`), requires the type class
**and** the qualifiers (`Upper` / `Lower` / …) to match **exactly**, and
fuzzy-matches only the specific — refusing a specific that is another's
proper prefix plus two or more letters (`Church` / `Churchill`, `Kings` /
`Kingston` score ≥ 0.9 under Jaro-Winkler: a different name, not a typo).
With Fitzwilliam St removed from the
gazetteer, "Fitzwilliam Street" is **unresolved** — even with the specific
threshold at zero. This triple is a permanent regression test.

---

## Pick

| Goal | Use |
| --- | --- |
| Compile clauses into the search contract | `compile_policy` → `CompilationResult.fixings` |
| The JSON handed to `urban_optimisation` | `CompiledFixings.to_dict()` / `from_dict()` |
| Is this quote really in the document? | `verify_span` |
| Bind a street name safely | `resolve_street_name` / `Gazetteer.resolve` |
| "the junction of A and B" | `Gazetteer.resolve_junction` |
| Candidates for a reviewer (never invented) | `Gazetteer.enumerate_candidates` |
| Two Main Streets — which one? | `GeometricHint` + `disambiguate` |
| What a human must look at | `CompilationResult.queue` (`AbstentionQueue`) |
| "Mon-Fri 07:00-10:00 except buses" | `parse_predicate` → `Predicate` |
| What is in force on a date | `resolve_in_force` (amendment DAG, Allen's algebra) |
| "should prioritise walking" → weights | `classify_wording`, `directional_priors`, `weight_prior` |
| Contradicting clauses | `detect_conflicts` / `merge_fixings` |
| Clip a design into the feasible set | `project_design` |
| How much search space is left | `free_variables` |
| The candidate generator | `enumerate_designs` |
| Which constraints actually bite | `relaxation_analysis` |
| Clause-by-clause verdict on a design | `policy_check` |
| Fixing → quote → page chain | `build_provenance_graph` → `citation_chains` |
| Compiler vs gold | `binding_agreement`, `agreement_score`, `cohen_kappa` |
| JSON a browser can render | `build_policy_compiler_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. `policy_compiler_template.py` is a single zero-dependency
file, CPython 3.8+. When run from the repository root it delegates the string
metrics to `algorithms/core/nlp/similarity` and BM25 / phrase search / RRF to
`algorithms/core/nlp/search`; the local fallbacks are bit-identical.

```bash
python policy_compiler_template.py     # the demo, prints "demo OK"
pytest _stress_test.py -q              # 309 oracle-based tests
```

---

## What's inside

### 0 — Validation
One error type, `PolicyCompilerError(ValueError)`, for every bad input:
wrong types (a number is not a street name, a bare string is never iterated as
a list of ids, a bool is not an integer), NaN / inf, negatives, empty input,
unknown enums.

### 1 — String metrics
`levenshtein`, `damerau_levenshtein` (optimal string alignment — *not* a
metric: `d(CA, ABC) = 3 > d(CA, AC) + d(AC, ABC) = 2`, pinned by a test),
`levenshtein_similarity` / `levenshtein_normalised_distance`, `jaro`,
`jaro_winkler`, `jaccard_tokens`, `lcs_similarity`, `ratcliff_obershelp`,
`soundex`, `metaphone`, `string_similarity` by name and `compare_strings`
(which uses the core `.compare()` API). Validation happens before delegating
to the core `.similarity()` / `.distance()` / `.compare()` classes.
*Convention:* `jaro` halves the transposition count with integer division,
as Winkler's `strcmp95` and the core do; implementations that use `t / 2`
differ when the count is odd (`jaro("aBeebe ", "Beaa cc")` is 0.6310 here,
0.5893 there).

### 2 — Text normalisation
`normalise_text`: accent folding, case, apostrophes dropped
(`Stephen's → stephens`), punctuation to spaces, ordinals to numerals
(`Twenty-First → 21st`), compass abbreviations to words (`NE` and dotted
`N.E.` → `northeast`, never two separate directions),
an abbreviation table (`Jct → junction`, `HGV → heavy goods vehicle`).
Street-type abbreviations are left to the resolver, because `St` is Street at
the end of a name and Saint at the start.

### 3 — Span verification
`verify_span` returns one of `exact` (claimed offsets hold), `recovered`
(literal substring; nearest occurrence to the claimed offset), `whitespace`
(substring after collapsing whitespace runs; offsets bracket the original
text), `fuzzy` (best token-aligned window by normalised Levenshtein above
`fuzzy_threshold` — returned for a human, **not verified**) or `rejected`.
The fuzzy search shortlists `max_fuzzy_windows` anchor windows by token
overlap and scores every window inside each anchor widened by one token, so a
typo'd quote is not lost to a neighbour that picked up a stray matching word;
a quote longer than the whole source is scored against all of it.

### 4 — JSON-schema-lite
`validate_json` supports `type / enum / const / required / properties /
additionalProperties / items / min-maxItems / uniqueItems / minimum / maximum /
exclusiveMinimum / exclusiveMaximum / min-maxLength / pattern / anyOf / oneOf /
allOf / not`, reports sorted errors with JSON-pointer paths, treats `True` and
`1` as different values, follows draft-6 in reading an integral float (`2.0`)
as an `integer`, flags NaN / inf anywhere, and refuses unknown keywords (a typo
like `requried` would otherwise accept everything) and malformed keyword values
(a non-bool `uniqueItems` would be a silent no-op; empty or repeated `type` /
`required` lists, non-string `title`). The checker may be stricter than
jsonschema's metaschema, never laxer.

### 5 — Constraint schema
`OPERATIONS` = ADD_EDGE, REMOVE_EDGE, REPARTITION, ADD_CROSSING,
GREEN_CORRIDOR, JUNCTION_REDESIGN. Every `PolicyClause` carries **exactly one
ruling per operation** from `PERMIT / PROHIBIT / REQUIRE / ABSTAIN`; silence is
explicit (`make_rulings` fills ABSTAIN). Clauses validate against
`CLAUSE_SCHEMA` and round-trip through `to_dict` / `from_dict`.

### 6 — Street-name resolver
`split_street_name`, `thoroughfare_class`, `resolve_street_name` → `bound`,
`ambiguous` (several admissible entries within `margin`) or `unresolved`;
every candidate examined is reported with its reason (`type_mismatch`,
`qualifier_mismatch`, `below_threshold`, `accepted`).

### 7 — Gazetteer
`GazetteerEntry` holds existing edges, **candidate** edges, nodes, geometry,
aliases and a district. `Gazetteer` adds a manual **override table**
(`add_override` refuses an id that is not an entry), `enumerate_candidates`
(reciprocal-rank fusion of override, exact-name, resolver and BM25 lists —
existing ids only, an empty list rather than a guess), `resolve` and
`resolve_junction` (the node both streets share; an ambiguous street makes
the junction ambiguous, with the shared nodes of every admissible pair as
candidates -- a street that IS bound contributes only its bound entry, never
the runners-up its margin ruled out). A geometric hint also **vetoes** a
unique name match it rules out (another district, too far, wrong
orientation): that is a contradiction for the queue, not a binding. An
applicable manual override always ranks **first** in `enumerate_candidates`,
consistent with `resolve`, which binds it. An edge that one entry lists as
existing and another as a candidate is refused at construction (the entry
rule, across entries): protecting it would *build* the link.

### 8 — Geometric disambiguation
Planar `bearing_deg`, `angular_difference`, `axial_difference` (undirected
streets), point-to-segment / polyline distances, `chain_contiguity` (is a
named street one connected run, and in what order) and `disambiguate`
(district, distance cap, orientation tolerance, adjacency to named streets;
with several survivors the nearest wins only if it is at most **half** as far
as the runner-up — a relative rule, identical in metres or kilometres; two
candidates both AT the point are a dead heat and stay ambiguous, never
decided by id order). *Convention:* a candidate with no coordinates cannot be
measured, so the distance and orientation tests do not eliminate it -- it can
survive (and be selected) by elimination of a measured rival. Give every
entry geometry if that is not wanted. The same holds for orientation when an
entry's polyline is a closed ring (a square drawn as a loop): its chord has
no bearing, so a bearing hint keeps it rather than aborting the compile. Distances are computed in units of the
segment's own length, so they stay exact from 1e-200 to 1e300.

### 9 — Ambiguity and abstention
`ambiguity_score = s2/s1` of the two best candidates (0 unchallenged, 1 at a
dead heat, scale-free), `normalised_entropy`, and the `AbstentionQueue`
ordered by ambiguity then clause id. A reviewer can `resolve` only to one of
the item's own candidates, or `reject`.

### 10 — Predicates
`TimeWindow` (half-open, wraps midnight), `Scenario`, `Predicate` (time
windows, days, vehicle classes, exemptions, scenario tags) with an **exact**
`intersects` (`Predicate` and `GeometricHint` are frozen: compiled fixings
share the clause's predicate object, so editing it later must be impossible),
and `parse_predicate` for order prose. The parser is a **closed,
total-consumption grammar**: the phrase is tokenised and every token must be
consumed by an explicit rule, or the call raises (the clause abstains) —
*refuse, never guess*:

```
phrase    := [ "no" (activity | classes) ] group* [ "only" ] [ marker exemption (joiner exemption)* ]
group     := days | times | classes | school-days     (each kind at most once, any order;
                                                       items of ONE kind joined by and / or / & / ,)
days      := ["on"] day | day SEP day | "from" day SEP day | "between" day "and" day
             | ["on" | "at"] weekday(s) / weekend(s)              ("at" only before weekend(s))
times     := clock SEP clock | "from" clock SEP clock | "between" ["the hours of"] clock "and" clock
             | ["at"] "any time" | "anytime" | "all times"
clock     := hh[:.]mm | hhmm  ["hrs" | "hours"]  |  h[.mm] ("am"|"pm"|"a.m."|"p.m.")
             | ["12"] noon / midday / midnight  |  a bare hour borrowing the other end's am/pm
classes   := car(s) taxi(s) bus(es) coach(es) van(s) lgv(s) lorry/lorries hgv(s) motorcycle(s)
             (bi)cycle(s) cyclists pedestrian(s) | emergency [vehicles] | motor vehicles
             | motor[-]cycles | [light|heavy|large] goods vehicles ["over"|"exceeding"|"above" "3.5" t|tonne(s)]
SEP       := - | to | until | till | through | thru       (en / em dashes read as -)
marker    := except [for] | excepting | excluding | other than | save for | apart from
             | with the exception of | but not
activity  := entry | loading | unloading | waiting | stopping | parking | left / right / u-turn
```

Exemptions are one class, day or time item each (mixed kinds may be joined by
*and* / *or*, a comma only joins items of one kind): days named there are
removed ("08:00-18:00 except Sundays" holds Monday to Saturday), times are
subtracted ("at any time except 10:00-16:00" is 16:00-10:00, an exemption
covering every restricted minute raises), classes are exempt. A meridiem
serves a bare or dotted unpadded end ("7.30-9.30am", "10-2pm" = 10am-2pm;
`08:00` and four-digit `1000` are 24-hour and never borrow: "1000-11pm" is
10:00-23:00). **A range with no am / pm needs one
plainly 24-hour end** (zero-padded, four-digit or past 12): "8.00-6.30" (the
usual UK sign style, 12-hour) and "7-10" raise. Plain *goods vehicles* is
vans **and** lorries (road-traffic law); `over 3.5 t` — the legal LGV / HGV
line — makes it `hgv`; motor vehicles exclude cycles and pedestrians; *school
days* is the scenario tag `school_days`; a trailing full stop is ignored.

Everything else raises, including: a single or open-ended time or day
("after 18.30", "from Monday"), holidays, *term time* (school days have no
weekends), a facility ("bus lane"), *access*, *loading*, *permit holders*,
any other weight ("over 7.5 tonnes" would restrict every 3.5-7.5 t HGV), a
post-positive *buses exempt*, *not Sundays*, "X only" after a class (on a
sign "Buses only" admits ONLY buses — the opposite reading), "only" anywhere
but the end of the head, bare *vehicles*, a day range from a day to itself,
two groups of one kind ("Mon-Fri 8am-6pm, Sat 8am-1pm" is a product a single
predicate cannot hold) and an exemption item mixing kinds ("except buses
Mon-Fri 7-10am"). `24:00-00:00` is refused (it would wrap into an empty
window).

### 11 — In-force resolution
`Instrument` revokes, supersedes or amends (clause-level `clause_changes`).
`amendment_edges` / `find_amendment_cycle` / `topological_order` (Kahn, ties
on start date then id) and `resolve_in_force`: effective periods are cut at
the earliest commenced revocation or supersession; an instrument cut short on
or before its own start never commences, so its own revocations,
supersessions and amendments never take effect; amendments are permanent
once commenced (the replacement clause belongs to the amended instrument and
survives revocation of the amender). Allen's 13 relations with
`allen_relation`, `allen_inverse` and an **exact** `allen_composition` table
built by exhaustive enumeration over endpoints `0..5`.

### 12 — Directional wording
`classify_wording` (longest modal cue wins: `must not` beats `must`) gives a
modality and `p ∈ [-1, 1]`; a cue never spans a comma ("shall, not later than
14 days …, notify" is an obligation), and a negator ending an earlier comma
clause never flips it. An aside in brackets, `[...]` or a dash pair (`--`,
spaced en dash, em dash, opening lower-case) is a comma clause of its own
("shall (not later than 14 days ...) notify" is an obligation). A cue that
opens a later comma clause (or follows only auxiliaries there: *be / is / will
/ will be / shall not* ...) *continues* an earlier one -- "will not, however,
support", "shall not, in any case, be prohibited", "The owner who does not,
shall pay", "no new development, other than ..., will be supported" -- so a
negation in the clauses before it, or a lone "Never," clause, is unresolved
(`neutral`). Symmetrically, a modal that *ends* its comma clause continues
into the later ones ("shall, in any event, never commence", "may, however,
not wait"), and one ending its sentence at a colon or semicolon into the
list after it, up to the full stop ("shall: (a) not obstruct …; (b) …"): a
negation or a second deontic word there is refused (`neutral`),
never dropped. A colon ends a sentence only when it is not between two digits
("Article 3: No …", but never *08:00*). A full stop ends a sentence only
before a word that is not lower-case, never before a number, a modal or a
whole day name (*Thus* is not *Thu*) ("Schedule No. 3", "Art. 5", "6 p.m.
Monday", "NO H.G.V. SHALL") and never after *Mr.*, *e.g.*, *i.e.*, a Saint's
*St.* ("on St. Stephen's Green") or before "No. 5" after *S.I.*: "No vehicle
exceeding 7.5 tonnes m.g.w. shall …" keeps its *No*. After an initial
(*J.*, *H.G.V.*), *a.m.* / *p.m.*, a capitalised word of at most four letters
(*Co.*, *Sq.*, *Ord.*, the *St.* / *Rd.* / *Ave.* ending a street name) or
another abbreviation of the table (*opp.*, *para.*, *etc.*, *Dept.*,
*approx.*) a full stop before a capital may or may not end the sentence ("Zone
A. No vehicle …" or "… Fitzwilliam St. Vehicles must …" against "J. Smith"):
the wording is read both ways and refused (`neutral`) when the two readings
differ. A *no* just after a clock time ("Between 8 a.m. and 6 p.m.
no person shall …") opens its clause. The bare participle *prohibited* under
a soft modal earlier in its clause -- *should*, *may*, *ought to*, *could*,
*might*, *would*, *can*: "should be prohibited", "may, in places, be
prohibited", "could be prohibited" -- is `neutral` with the modal as its cue,
so a hard clause worded with *should* / *ought to* is queued (*ought to* /
*ought not to* are soft cues like *should* / *should not*, also around an
interjection: "ought, wherever possible, to be prohibited"); a cue with
its own verb ("It should be noted that parking *is prohibited*") is not
governed. The soft verbs are read in their inflected forms too ("will be
encouraged / supported / resisted", "is discouraged", "prioritized", "is
supporting", "seeks / sought to"). A negator the cue does not contain flips it
deontically -- "is not prohibited" is a permission, "will not support" a
discouragement, "No vehicle shall …" and "Save as provided in Article 5, no
person shall …" (a *No* opening the cue's comma clause, or opening the
sentence when the cue opens its own clause -- "No vehicle, other than a bus,
shall …"; unless an idiom: *no doubt*, *no later / less … than*) and
"must never" prohibitions, with "at no time", "in / under no
circumstances", "on no account" read as *never* ("No 5 Main Street …" is a
house number, not a negation); a negated *requirement* permits ("No permit
is required"); contractions are expanded (*mustn't*, *don't*,
*isn't*, *wasn't*, *hasn't*, *mightn't*, *oughtn't*, *ain't* …) whatever the
apostrophe (`'`, `’`, `‘`, `ʼ` or a backtick). Negation is handled **conservatively**: any other negating word in
the cue's comma clause (*nor*, *nothing*, *without*, *prohibited*, *unlawful*,
*fail*, *no longer* …, other than the comparatives *not less / more than*), a
sentence-opening *No* whose clause is not the cue's subject ("No objection
having been made, the Council shall …"), or a second
deontic word governed by the cue ("shall not be **required**", "shall be
**permitted**") is a configuration the rules do not model, so the wording is
`neutral` with `p = 0` but keeps its `cue` — and `compile_policy` still
queues a hard clause whose unresolved cue is soft (*should*, *encourage* …)
as `wording_ambiguous`. *May* after *in / of / from / by …* or before a
number is the month, not a cue. `directional_priors` averages per objective;
`weight_prior` computes `w = w_planner ⊙ (1 + η·p)` and refuses a negative
weight.

### 13 — Compilation
`compile_clause` maps rulings onto a bound target: ADD_EDGE acts on the
target's **candidate** edges (`PROHIBIT → z_e := 0`), REMOVE_EDGE on its
**existing** edges (`PROHIBIT → z_e := 1`, protected), REPARTITION bounds
`quantity:edge` variables (prohibiting it pins the baseline), GREEN_CORRIDOR
bounds `green:edge`, ADD_CROSSING fixes `w_v`, JUNCTION_REDESIGN restricts
`k_v` within `K = (priority, signalised, signalised_banned_turns, roundabout,
mini_roundabout)`. A non-trivial predicate makes the result conditional.
REPARTITION REQUIRE without `bounds` ("must be repartitioned" = `y ≠ baseline`,
no interval can say it) is queued as `unrepresentable`; prohibiting every type
in K would empty `k_v`'s domain and is queued as a `conflict`. Variables are
`quantity:edge` with a **colon-free quantity** (enforced), so the edge id is
everything after the first colon and may itself contain `:` (OSM-style
ids).
`compile_policy` runs in-force filter → span → wording → binding → compile →
merge for every clause in id order.

The contract:

```json
{"fixedZero": ["c1", "c2"], "fixedOne": ["e1", "e2"],
 "bounds": {"footway_width:e5": [3.0, 4.0]}, "crossingFixings": {"n2": 1},
 "junctionDomains": {"n2": ["mini_roundabout", "priority", "signalised",
                            "signalised_banned_turns"]},
 "conditional": [{"predicate": {"timeWindows": [[1320, 360]], "days": [],
                  "vehicleClasses": [], "exemptClasses": ["emergency"],
                  "scenarioTags": []},
                  "fixing": {"fixedZero": ["e3"], "fixedOne": [], "bounds": {},
                             "crossingFixings": {}, "junctionDomains": {}}}],
 "provenance": {"fixedZero:c1": ["A03"], "...": ["..."]}}
```

### 14 — Conflict detection
`detect_conflicts` finds `zero_vs_one`, `crossing_clash`, `empty_bounds` and
`empty_junction_domain` between clauses whose predicates intersect (a
never-satisfiable predicate meets nothing). Junction domains are also checked
**jointly**, unconditional and conditional together, by evaluating one
scenario per (day, vehicle class, window cell) — exact, because three pairwise
overlapping sets can be jointly empty only in some scenarios. A conflict names
exactly the clauses in some **minimal** contradictory family (a pair for
bounds and binaries; at most five for junction domains, since each member
must be the only one missing some type), so a clause compatible with every
culprit is not blamed. `merge_fixings` combines agreeing fixings (bounds and
domains intersect), **withholds the atoms of every conflicting clause**
everywhere (the clauses go to the queue) and keeps an undisputed clause's atom
on the same key — dropping it would silently lose a valid constraint.

### 15 — Projection and free-variable reduction
`DesignSpace`, `Design`, `applicable_fixings` (unconditional + predicates that
hold in a scenario), `violations`, `project_design` (idempotent, always
feasible, touches only fixed variables), `free_variables` (free binaries,
reduced domains, design counts before / after, and `unknown_keys` — fixings on
no variable of the space, judged per kind: a crossing fixing on a node that is
only a junction node is unknown) and `enumerate_designs`, the
candidate generator that iterates **only free variables** — a prohibited
design is unrepresentable, not filtered.

### 16 — Relaxation analysis
`dominates`, `non_dominated_indices`, exact `hypervolume` (slicing),
`pareto_set` (exhaustive) and `relaxation_analysis`: drop each constraint,
recompute the non-dominated set exactly, report added / removed objective
vectors (as sets), whether the design set is identical, and the hypervolume
gain.

### 17 — Policy check
`policy_check(design, result)` → per clause `satisfied`, `in_tension` (with the
violated keys), `abstained` or `not_applicable`, each with quote, page and
document; compliance ratio over the checkable clauses. A clause that compiled
to no fixing (a bare permission) is `not_applicable`: it can never be in
tension, so counting it satisfied would only inflate the ratio. A queued
clause is `abstained` -- unless it was queued for a partial issue (say, no
candidate edges for one of its rulings) and the fixings it still contributes
to the contract are broken: then it is `in_tension`. A clause withheld by a
conflict loses only the clashing key (as in `merge_fixings`): it is
`abstained` unless a fixing it still contributes on another key is broken,
and the clashing key is never blamed on it.

### 18 — Provenance and agreement
`build_provenance_graph`: document → page → span → clause → binding → fixing;
`citation_chains` walks back from any fixing; `verify` reports a fixing with no
chain or one citing an unverified span. `agreement_score` (precision over
answered, recall over gold, abstention rate), `binding_agreement`,
`cohen_kappa`.

### 19 — Payloads, 20 — Example district, 21 — Demo
`build_policy_compiler_payload` / `build_a2ui_payload` are strict JSON
(camelCase). `build_example_district()` is a Fitzwilliam-area fixture — a 2024
traffic order with its 2025 amendment, a gazetteer with the street-name trap,
two Main Streets, a park with candidate links, and a design space with an
objective — so the full pipeline runs with no data.

---

## Design notes

- **Refuse, don't guess.** Unverified quote, fuzzy quote, unresolved or
  ambiguous binding, missing baseline, no target elements, a hard clause with
  soft wording, a ruling no fixing can express (`unrepresentable`), a
  conflict: each is an abstention with a reason, never a silent drop and
  never a best guess.
- **Soft clauses do not compile.** `strength="soft"` clauses feed the weight
  prior only. A clause marked hard whose wording is a recommendation
  ("should …") is queued as `wording_ambiguous`.
- **PERMIT restricts nothing** — with one exception. It is recorded (for the
  permission-biased candidate generator downstream) but compiles to no fixing,
  except a REPARTITION permit that carries explicit `bounds`: that is a
  permission scoped to the range and compiles the bounds.
- **Wording is checked one way only.** A hard clause whose wording is a
  recommendation or discouragement ("should …") abstains as
  `wording_ambiguous` (conservative: the hard/soft call is unclear). A hard
  clause worded as a bare permission ("may …") or with the opposite polarity
  to its ruling is not cross-checked — the ruling is the reader's typed claim.
- **A clause's `instrument_id` lists it.** With `instruments`, a clause naming
  an instrument that does not list it in `clause_ids` is treated as that
  instrument's clause (so a clause of a revoked order is `not_in_force`); an
  unknown instrument, or one contradicting the instrument that does list the
  clause, raises -- so a replacement clause must name the **amending**
  instrument that lists it. A clause no instrument lists and that names none
  has no validity record and is compiled as in force (*convention*). Without
  `instruments` no in-force filter runs.
- **Amendments are permanent.** Revoking an amending instrument after it
  commenced does not restore the clause it replaced (the usual drafting
  convention: a repeal does not revive). Revoking it *before* it commences
  means it never acted at all.
- **Relaxation analysis is exhaustive** — exact, and meant for small or
  already-reduced spaces. "Binding" means the **set** of distinct
  non-dominated objective vectors (the Pareto front) changes — the usual
  sense of a binding constraint. Freeing a variable no objective sees only
  adds alternative optima that tie existing front vectors: not binding;
  `identical_designs` reports that design-level change separately.
- **Inherent limits.** The fuzzy span search needs at least one shared token
  to anchor on: a quote in which *every* token is misspelt is scored only
  against the first windows and is usually rejected (fails safe — the clause
  is queued). `parse_predicate` reads any `dddd-dddd` as a clock window, so a
  year range ("2024 to 2025") must not be passed to it as predicate text.
  Fuzzy specific matching cannot tell a typo from a different name one letter
  longer that is not a prefix extension (`Ash` / `Nash`, JW 0.917 binds at the
  default threshold 0.9 when Ash Road is absent — pinned; raise `threshold` or
  add an override). Cue-based wording is deliberately coarse: double
  negatives and second deontic words are refused to `neutral` rather than
  read ("Nothing in this Order shall prohibit …", "It shall be unlawful …",
  "shall not be required", "No vehicle shall be permitted …" all score 0), so
  some correct prohibitions lose their prior; it feeds the soft prior only,
  never a fixing. The cue table is finite: synonyms outside it ("will be
  welcomed", "is expected to", "opposed") read neutral. And the longest cue
  wins across the whole text, not only among overlapping cues, so a quote
  holding two sentences with different cues ("Vehicles may not enter; walking
  is encouraged") takes the longer cue (*convention*: quote one sentence per
  clause). A range start written `24:00` / `2400` is kept as the window
  `[1440, x]`, the same minutes as `[0, x]`. The comma and full-stop rules
  are conservative as well: a contrast ("Parking, not cycling, is
  prohibited"), a stray comma after a negated subject and an introductory
  phrase with no comma before a *no* ("In the park no vehicle shall …") are
  refused to `neutral`, and so is a sentence whose full stop may or may not
  end it ("No vehicle owned by J. Smith shall …", "No vehicle in Co. Dublin
  shall …"). The abbreviation knowledge is finite: an unlisted abbreviation
  of five letters or more before a capital ("the Residents Assoc. Committee")
  still ends the sentence and cuts the cue off from its *No* (pinned). ALL-CAPS
  text has no case to tell a sentence start from a name, so there the listed
  abbreviations join as before ("NO PARKING ON MAIN ST. VEHICLES SHALL BE
  REMOVED" reads -1) and a dash aside is not split; a dash pair opening
  upper-case (a typeset range "Mon – Fri 8am – 6pm") is no aside (pinned). A
  modal ending its comma clause makes any negation after it unresolved, so
  "The operator shall, without delay, notify …" is refused too (safe). A cue is a word whatever its grammatical role, so
  "Supporting documents must be submitted" reads the longer soft cue
  *support* and such a hard clause is queued as `wording_ambiguous` (safe).
  One plainly 24-hour end makes a range with no am / pm 24-hour throughout:
  "08:00-6.30" is the overnight window 08:00-06:30 (*convention*, as
  documented in section 10). The geometric tests compare exact floats: a
  candidate exactly at `max_distance`, or a hint point interpolated onto an
  oblique segment (a "dead heat" only in exact arithmetic), is decided by one
  ulp of rounding and may flip under rescaling; a point that is a *vertex* of
  both polylines stays a dead heat at every scale (pinned).
- **Validated at the entry points.** `GazetteerEntry`, `PolicyClause` and
  `Design` are ordinary (mutable) dataclasses: they are validated when built
  or when passed to an API function, and a caller editing an entry after the
  `Gazetteer` holds it (say, adding a candidate edge another entry lists as
  existing) bypasses those checks (*convention*). Mixed str / int keys in a
  dict argument never leak a bare `TypeError`: they raise `PolicyCompilerError`
  wherever the keys must be string ids; `Design.key()` / `as_dict()` on a
  hand-built design with such keys are not entry points.
- **Core sums are re-ordered.** The core BM25 sums over a `set` of query
  tokens, whose order follows `PYTHONHASHSEED`; `bm25_scores` asks the core
  for one token at a time and sums the sorted tokens with `math.fsum`.

---

## Testing

`_stress_test.py` — **309 tests**, zero skips, green from the module directory
and the repository root and under `PYTHONHASHSEED` 0 / 1 / 1010:

1. **Closed forms.** `levenshtein(kitten, sitting) = 3`;
   `soundex(Robert) = soundex(Rupert) = R163` and the census rules;
   `jaro_winkler(MARTHA, MARHTA) = 17.3/18`; `(DWAYNE, DUANE) = 0.84`; the trap
   values 0.9306 / 0.9556 / 0.8244; a hand-worked BM25 score; RRF
   `1/61 + 1/62`; kappa 0.4 on the textbook table; Allen's published
   compositions; a hand-built amendment chain on five dates; a three-edge
   design space whose front and hypervolume (52, gain 32) are worked on paper;
   the district's full compiled dict.
2. **Exact identities.** Literal span recovery on fuzzed substrings; JSON
   round trip of the contract; projection idempotent and feasible; **a compiled
   prohibition is unrepresentable** — exhaustive enumeration shows the
   generator's output is exactly the feasible subset; **relaxing a non-binding
   constraint leaves the non-dominated set identical** (vectors and designs).
3. **Metamorphic.** Levenshtein symmetry / identity / triangle (fuzzed) against
   an independent recursion; the resolver never crosses any of the 32 type
   classes; prefix shifts move spans exactly; scale invariance at 1e±6
   (ambiguity, geometry, hypervolume, weight prior, relaxation, bounds);
   determinism across hash seeds in subprocesses.
4. **Differential.** networkx (cycles, topological validity, components),
   numpy (non-dominated filter, staircase hypervolume), sklearn (kappa),
   jsonschema (Draft-7 on mutated clauses), difflib (Ratcliff-Obershelp), and
   the core `nlp/similarity` and `nlp/search` modules — bit-identical values and
   identical exception types with the core forced on and off.

```bash
pytest _stress_test.py -q -n 8
PYTHONHASHSEED=1010 pytest _stress_test.py -q
cd ../../../.. && python -m pytest algorithms/domains/urban/policy_compiler/ -q
```

**What the suite caught.** A JSON-schema gap: NaN / inf nested under a property
the schema left unconstrained passed validation; the validator now walks the
whole instance for non-finite numbers first. A seeded junk-argument fuzz of
every public name (and the main methods) found ~25 entry points where a
non-iterable, a dict or a bytes object escaped as a bare `TypeError` /
`AttributeError` / `IndexError`; all now raise `PolicyCompilerError` and are
pinned in the regression tier, alongside input-mutation (no aliasing) checks.
A 28-mutant hand-made mutation run left seven survivors (inclusive margin and
threshold, touching time windows, equidistant span recovery, Soundex `W`
transparency, baseline-preferring projection, duplicate front vectors); each
now has a value test that kills it, and all 28 are killed.

**What the bug hunt caught** (each pinned in tier 29 by a test that fails
against the pre-hunt template): an instrument revoked before it commenced
still revoked and amended others; `except Sunday(s)` / `except weekends` made
a restriction apply ONLY on those days; three junction domains with an empty
common part were missed when one was conditional (and an unsatisfiable
predicate "met" every unconditional fixing, withholding a valid protection);
two candidates both at distance 0 from a hint were "won" by id order; a
geometric hint ruling out the only name match was ignored; `margin` never
reached junction targets (and an ambiguous street made the junction
"unresolved" with no candidates); the fuzzy span search rejected a one-letter
typo whose window ranked just below a neighbour, and a quote longer than the
source escaped as a bare `AssertionError`; `violations` accepted NaN widths
and `z = 2`; a clause aliased the caller's rulings dict; `N.E.` normalised to
two directions; `24:00-00:00` built an empty window. Relaxation "binding" now
compares the front as a set (the builder's multiset oracle was the test bug).
Independent oracles added: brute-force scenario enumeration for conflicts and
a fixed-point in-force reference. A second, 28-mutant run on different
targets (Allen relations and composition, cycle colouring, cut ranking, void
rule, predicates, bearings, span offsets, bag overlap, ordinals, eta,
agreement denominators, merge, joint cells, kappa, compliance, entropy) left
three survivors; two now have killing tests, the third (`len(ends) >= 2` in
`chain_contiguity`) is equivalent: a connected graph of maximum degree 2 is a
path or a cycle and has exactly 2 or 0 ends.

**What the second bug hunt caught** (tier 30; every test fails against the
round-1 template): `validate_json` leaked a bare `TypeError` comparing a dict
with a non-string key, refused `2.0` as an `integer` (jsonschema accepts it —
now a random-schema differential over every keyword), and `check_schema`
accepted a non-bool `uniqueItems` that then did nothing; the junction of an
ambiguous street and a margin-bound one leaked nodes of the runner-up street
the margin had ruled out; instrument **iterators** were walked several times
(a `KeyError`, and an amendment cycle went undetected); REPARTITION REQUIRE
without bounds compiled to nothing and passed every policy check, and
prohibiting all five junction types silently emptied the design space; a
baseline lookup split `width:osm:1` at the last colon; `free_variables`
judged unknown keys against the union of all ids; negated wording kept the
cue's sign ("will not support" scored +0.5, "No vehicle shall" +1); a pairwise
clash suppressed the joint junction check, so other jointly-contradictory
clauses were withheld unnamed, and a conflict withheld an **undisputed**
clause's valid bound on the same key; an override did not rank first among
candidates; `compile_policy` accepted a NaN threshold (and leaked
`AttributeError` for a list baseline) when no clause reached that stage;
midpoints overflowed near `1.7e308` and segment distances overflowed /
underflowed at `1e160` / `1e-200`; a bare permission counted as "satisfied" in
the compliance ratio; and a caller could mutate a clause's `Predicate` after
compilation, rewriting the handed-over contract. A new oracle checks the
clause SETS in conflict (minimal families, by brute-force enumeration on
15-minute cells with tags) and that the merge enforces every undisputed atom.
A 26-mutant run on the round-1 and round-2 code and on untouched functions
left seven survivors (the leading-*No* window, an oblique-segment norm, the
exact factor-of-two boundary, a span-less citation chain, alias best score,
an ambiguous landmark fallback, a local-only Metaphone table entry); tier 31
kills all seven. *Convention noted:* `metaphone` reproduces the core's
simplified variant, which collapses repeated codes even across a vowel
(`BOB → B`, classic Metaphone gives `BB`).

**What the third bug hunt caught** (tier 32; each test fails against the
round-2 template): `parse_predicate` read no am/pm time at all ("Mon-Fri
7am-10am" became an ALL-DAY restriction), read "8.00 - 6.30pm" as the
overnight window 08:00 → 06:30, ignored "between X and Y", read a window after
`except` as the restriction itself, turned "light goods vehicles" into HGVs
(dropping an LGV exemption) and "motor cycles" into cycles; the standard TRO
form "Save as provided …, no person shall …" (and "Article 3. No …", "No
vehicle exceeding 7.5 tonnes shall …") scored +1 instead of −1; a clause's
`instrument_id` was never read, so a clause of a revoked order compiled as in
force; the street resolver bound "Church Road" to "Churchill Road" (and Kings
→ Kingston, Hill → Hillside) when the true street was absent; a predicate
that can never hold compiled a dead fixing that `policy_check` reported as in
tension. A hand-written nine-clause TRO is now a closed-form end-to-end oracle,
and brute-force fuzzes confirmed the conflict sets are exactly the minimal
infeasible families and that no clause leaves `compile_policy` without being
compiled, queued or skipped with a reason. A 25-mutant run on round-2 and
round-3 code and untouched functions left five killable survivors (tier 33
kills them) and one equivalent mutant (duplicate amendment edges in
`topological_order` are decremented as often as counted).

**What the fourth bug hunt caught** (tier 34; each test fails against the
round-3 template). All in `parse_predicate`, all *silent*: "with the
exception of buses", "save for", "apart from" put buses into the restricted
classes (inverted); typeset en-dashes ("Mon–Fri") and "Monday through /
until Friday", "between Monday and Friday" named the two end days only;
"07.00 hrs to 19.00 hrs", "0700 hrs", single dotted or four-digit times and
bare ranges ("Mon-Sat 8-6") were dropped, making the restriction all day;
"except buses Mon-Fri 7am-10am" exempted the weekdays and the hours,
"Mon-Fri 8am-6pm, Sat 8am-1pm" applied 8-6 on Saturdays and "except Sundays
10am-2pm" exempted 10-2 every day; "except for access / loading / permit
holders" was dropped, "bus lane" restricted buses only, a weight limit with
no class restricted every vehicle; plain "goods vehicles" let vans through;
"08:00-9pm" became 20:00-21:00. In `classify_wording`, "At no time shall …"
and "Under no circumstances may …" scored +1 / 0 instead of −1. Each misread
now parses correctly or raises (a safe failure: the caller cannot build the
clause). New oracles: a seeded fuzz of well-formed TRO phrases checked
minute by minute against the meaning they were built from, and a random
clause-set fuzz checking that the contract is independent of clause order,
round-trips through JSON byte for byte and projects feasibly in every probed
scenario. Re-verified clean: the in-force DAG with clause-level
`instrument_id` (revoked, void, amended and permanent-amendment cases), the
prefix-extension guard against `St` / `Street` / `Saint` forms, and the
round-3 am/pm, noon / midnight and `except`-subtraction rules. A 24-mutant run
on round-3 and round-4 code left four survivors (equal hours across the
meridiem twice, hour 0 under am/pm, three alternating day/time groups); tier
35 kills all four.


**What the fifth bug hunt changed** (tiers 36 and 37; each regression test
fails against the round-4 template). Rounds 3 and 4 were dominated by silent
misreads in `parse_predicate`: a permissive regex parser over open English
half-understood new phrasings every round. It is now the **closed grammar**
above — every token consumed by an explicit rule or the phrase is refused.
Misreads it removes: "7am-10am not Sundays" (restricted Sundays ONLY),
"8.00-6.30" (an overnight 08:00-06:30 window), "buses only" / "light goods
vehicles only" (inverse of a bus-only sign), "goods vehicles over 7.5 tonnes"
(restricted every HGV from 3.5 t), "term time" (read as school days), "No
vehicles" / "No through traffic" / "No entry for taxis" (scope dropped),
"Monday to Monday", and any unread word ("approx", "roughly", "daily",
"inclusive", "(Mon-Fri)") silently ignored. Two round-3 / round-4 pins were
test bugs and are rewritten: "X only" and the 7.5 t limit were *conventions*
that read a wrong restriction; and the round-4 fuzz generated "11:45 to 11.15"
as 24-hour, which is genuinely ambiguous (now expected to raise). New
oracles: a grammar generator whose phrases must parse to the meaning an
independent structural reference computes, minute by minute, and a
**mutation oracle** (the convergence check) that inserts or substitutes
English words — never-read ones (*after*, *permit*, *approx* …) and
sometimes-read ones (*until*, *only*, *loading*, *weekdays*, *excluding*,
*tonnes*) — and requires the parser to raise or return exactly the meaning
the reference gives the mutant. In `classify_wording`: "shall not be
prohibited" scored −1, "It is not the case that … must", "Nothing … shall
prohibit" and "It shall be unlawful" +1, "No works in May" −1 and "don't
encourage" +0.5; contractions were unread. A 24-mutant run on the new parser
and wording code left four killable survivors ("from Monday" read as Monday,
"between X - Y", a negator ending the previous sentence, *not less than* as
a negation; tier 37 kills them) and one equivalent mutant (a redundant
school-day check, removed). Re-verified clean outside the parser: hypervolume
against a brute-force grid (200 random sets) and at 1e±6, NaN / inf refusal
on every numeric entry point probed, dead-parameter and input-mutation
probes on `verify_span`, `enumerate_candidates` and `compile_policy`.

**What the sixth bug hunt caught** (tiers 38 and 39; each tier-38 test fails
against the round-5 template). The confirmation round did not converge: eight
template bugs. In the closed grammar, one: a four-digit 24-hour end borrowed
the other end's meridiem ("1000-11pm" became 22:00-23:00; the round-4
`08:00` fix covered zero-padded ends only). A seeded fuzz of mixed 24-hour /
am-pm ranges -- a shape the tier-36 generator never builds -- now pins it.
Five in `classify_wording`, all sign errors or unread cues:
- a cue matched across a comma, so "The operator shall, not later than 14
  days …, notify" read as *shall not* (−1);
- "No permit is required" read as a prohibition (a negated requirement
  permits);
- *longer* was an idiom even without *than*, so "must no longer enter" scored
  +1;
- a sentence-opening *No* negated a cue in a later clause with its own subject
  ("No objections having been received, the Council shall …" scored −1);
- inflected soft verbs ("will be encouraged / resisted", "is discouraged",
  "prioritized") were unread, so a hard clause worded that way skipped the
  `wording_ambiguous` check.

Outside them, two:
- a gazetteer square drawn as a closed ring plus a bearing hint made
  `disambiguate` raise, aborting `compile_policy` for every clause;
- `policy_check` called a clause queued for a partial issue `abstained` while
  the fixings it still contributed were broken (compliance 1.0 for an
  infeasible design).

Clean this round:
- the metamorphic fuzz of 28,904 grammar phrases (case, dashes, a.m. forms,
  spacing, full stop);
- 60,000 token-soup phrases (no leak, and every accepted predicate
  round-trips);
- the string metrics against independent references and the core (the Jaro
  rounding is a *convention*, above);
- hypervolume, non-dominated and kappa differentials;
- scale invariance of disambiguation, hypervolume, weight prior, ambiguity
  and entropy at 1e±6;
- NaN / inf on 60 numeric entry points;
- every optional parameter is live;
- no aliasing of inputs or results.

A 24-mutant run on the new and parser code left four survivors (the midnight
rejoin of exempt-time runs, which *No* is handled, a queued conditional
outside its predicate, a fixing on an edge outside the space); tier 39 kills
all four.

**What the seventh bug hunt caught** (tiers 40 and 41; each tier-40 test fails
against the round-6 template). The confirmation round did not converge: seven
template bugs, none in the predicate grammar. Five in `classify_wording`:
- commas scoped negation wrongly both ways: a negator ending the comma clause
  before the cue flipped it ("The owner, if he does not, shall pay the costs"
  scored −1), while the cue's own negation behind an interjection was lost
  ("will not, however, support" +0.5, "Never, under any circumstances, shall
  …" +1, "shall not, in any case, be prohibited" −1);
- an abbreviation's full stop ended the sentence and cut a cue off from its
  *No*: "No H.G.V. shall enter", "… 7.5 tonnes m.g.w. shall proceed", "… of
  the Dept. of Transport, shall enter", "No vehicle on St. Stephen's Green
  shall wait", "No vehicle between 8 a.m. and 6 p.m. shall wait" all scored
  +1;
- "should be prohibited" / "may be prohibited" took the longer cue
  *prohibited* and scored a hard −1, so a hard clause worded as a
  recommendation compiled a fixing instead of being queued;
- the *-ing* forms and the inflections of *seek to* ("seeks / sought to", "is
  supporting", "prioritizing") were unread, so a hard clause so worded
  skipped the `wording_ambiguous` check;
- *wasn't*, *weren't*, *hasn't*, *haven't*, *hadn't* were not expanded:
  "Parking wasn't prohibited" scored −1.

Outside it, two:
- `policy_check` called every clause a conflict names `abstained`, but a
  conflict withholds only the clashing key: the clause's other fixings stay
  in the contract, and a design breaking them got compliance 1.0 (a seeded
  fuzz now checks that every broken contract key is reported `in_tension` on
  a clause its provenance names);
- the gazetteer accepted an edge listed as existing by one entry and as a
  candidate by another: a node clause "shall not remove" then fixed z := 1 on
  the candidate (building the link), and one prohibiting both ADD_EDGE and
  REMOVE_EDGE aborted `compile_policy` for every clause.

Clean this round:
- an exhaustive generator of 2,102,128 mixed-format clock ranges (24-hour,
  four-digit, unpadded dotted, bare, am / pm in four spellings, noon /
  midday / midnight with and without "12", 12am / 12pm, 24:00, minutes,
  hyphen / en-dash / to / until / from … to / between … and) against an
  independent reference: 0 mismatches;
- 60,000 clock-heavy predicate token soups (no leak; every accepted
  predicate round-trips) and 200,000 wording token soups (no leak);
- string metrics against independent DPs and difflib, hypervolume against a
  grid count, the non-dominated filter, kappa against sklearn, and Allen
  inverse / composition over all intervals on 0..5;
- scale invariance at 1e±6 (hypervolume, non-dominated sets, disambiguation,
  weight prior, ambiguity, entropy, compiled bounds and projection);
- NaN / inf on 38 numeric entry points;
- every optional parameter is live;
- no aliasing of inputs or results.

The `08:00-6.30` 24-hour reading is a documented *convention*, not a bug. A
34-mutant run on the new and untouched code left three survivors (a
one-comma continuation, `minutes_per_week` with no day listed, and the
soft-modal rule applied to cues carrying their own verb). The last exposed
an over-broad rule, now narrowed to the bare participle. Tier 41 kills all
three.

**What the eighth bug hunt caught** (tiers 42 and 43; each tier-42 regression
test fails against the round-7 template). The confirmation round did not
converge: six template bugs, five of them in `classify_wording` scope, all
silent sign errors or skipped wording checks. A metamorphic generator --
subject x modal x negation x verb, with a cue-free modifier inserted between
subject and modal and the bare sentence's reading as the reference -- gave a
wrong sign on 4,328 of 6,032 generated sentences on the round-7 template and
0 now:
- a colon split the sentence even inside a clock time, so "No vehicle between
  08:00 and 18:00 shall wait" -- the commonest TRO form -- scored +1;
- round 7's fix knew six abbreviations: "Schedule No. 3", "Art. 5", "(e.g.
  HGVs)", "Mr. Smith", "Jct. 5", "1 Jan. 2025", "6 p.m. Monday" still ended
  the sentence and cut the cue off from its *No* (+1). A full stop before a
  number or a day name never ends a sentence now; one that may or may not
  ("J. Smith" / "Zone A. No vehicle") is read both ways and refused if the
  readings differ;
- a negation in a LATER comma clause was never examined: "The applicant
  shall, in any event, never commence works" scored +1 and "Vehicles may,
  however, not wait" 0. A modal ending its comma clause now continues into the
  later ones (the mirror of round 7's rule);
- only *should* and *may* governed a bare *prohibited*: "could / might /
  would / can be prohibited" and "ought to be prohibited" scored a hard -1, and
  a hard clause worded "ought to be prohibited" compiled instead of being
  queued (*ought to* / *ought not to* are now soft cues);
- *mightn't*, *mayn't*, *oughtn't*, *ain't* were unread, and an apostrophe
  typed as a backtick, a left quote or U+02BC split *isn't*: "Parking isn`t
  prohibited" scored -1.

Outside it, one: a dict whose keys mix str and int (an OSM node id read as a
number) leaked a bare `TypeError` from `sorted()` in twelve entry points
(`Gazetteer`, `DesignSpace`, `weight_prior`, `agreement_score`,
`binding_agreement`, `detect_conflicts`, `merge_fixings`, `project_design`,
`DesignSpace.validate`, `check_schema`, `validate_json`, a dict of clauses):
keys were sorted before being validated.

Clean this round:
- `policy_check` against an independent oracle (each clause's own atoms
  minus the keys a conflict names, judged on the design; statuses, violated
  keys and the compliance ratio) on 22,400 random compilations and designs,
  both directions -- round 7's stripping of only the clashing keys holds;
- the `Gazetteer` existing / candidate overlap refusal (a caller editing an
  entry after construction bypasses it: *convention*);
- the wording gate: 6,032 hard clauses, every *should* one queued;
- 270,000 wording token soups with NULs, colons, dots and apostrophes (no
  leak, deterministic, every non-neutral reading consistent with its cue);
- clause-order invariance and the JSON round trip of the contract on 1,500
  random clause sets;
- scale invariance of `resolve`, `resolve_junction` and `disambiguate` at
  1e+-6 on 3,000 random gazetteers: 19 flips, every one an exact tie at the
  cap or a dead heat broken by one ulp (*inherent limit*, pinned);
- NaN / inf on 49 numeric entry points (147 cases);
- every optional parameter probed is live;
- no aliasing of inputs or results.

A 31-mutant run on the new and untouched code (sentence splitting, the
two-reading refusal, forward continuation, soft-modal governance,
contractions, key validation, violations, conflicts, merge, need(),
policy_check cuts, the gazetteer overlap, agreement) left six survivors; one
was a malformed mutant (rewritten, killed). Tier 43 kills the other five: a
negated modal ending its clause ("must not, in any case, be permitted"), the
nearest of two soft modals governing ("could, and should, be prohibited"), a
spurious missing_baseline on a target with no existing edges, policy_check
cutting a key from a clause no conflict names, and a NUL in the wording.
Fixing the sentence rule exposed one flaw in the fix itself (a day-name
prefix joined "street. Monument Square"); it is pinned with the
*Assoc.* limit.

**What the ninth bug hunt caught** (tiers 44 and 45; each tier-44 regression
test fails against the round-8 template). The confirmation round did not
converge: six template bugs, all in `classify_wording` scope, all silent sign
errors or a skipped wording check. An independent metamorphic generator --
subject x modal x negation, an aside (commas, brackets, dashes, a citation)
at every slot, and a preceding cue-free sentence ending in an ambiguous or
listed full stop, the bare sentence's reading as the reference -- gave a wrong
sign on 6,332 of 42,120 sentences on the round-8 template and 0 now:
- round 7's always-joined abbreviations also end sentences: in "No waiting on
  Fitzwilliam St. Vehicles must use the car park" (and after *Rd.*, *Ave.*,
  *Dept.*, *etc.*, *approx.*, *H.G.V.*) the first sentence's *No* negated the
  second's cue (-1 for an obligation). Before a capitalised word such a stop
  is now read both ways; *St. Stephen's*, *S.I. No. 274* and an ALL-CAPS
  next word still join;
- the day-name join matched *Thu* + *s*: "No objection was received. Thus
  the Council shall proceed" scored -1. Only whole day words join;
- brackets and dashes did not scope a cue: "The operator shall (not later
  than 14 days …) notify" read *shall not* (-1; round 6 fixed only the comma
  form), and "Vehicles must (other than buses, taxis and cycles) not enter"
  hid the negation from the forward rule (+1). An aside in `()`, `[]` or a
  dash pair is a comma clause of its own;
- the backward rule continued only a cue opening its clause: "In the Green
  Belt no new development, other than for agriculture, will be supported" --
  standard local-plan wording -- and "Never, in any case, will the Council
  support …" scored +0.5. A cue after auxiliaries only, and a lone negator
  clause, now continue;
- a comma interjection split the soft cue *ought to*: "Through traffic ought,
  wherever possible, to be prohibited" read a hard -1 and a hard clause so
  worded compiled a fixing instead of being queued;
- a colon or semicolon ended the sentence before a list: "The operator
  shall: (a) not obstruct the footway; and (b) not park on the verge" scored
  +1. A modal ending its sentence there continues up to the full stop.

Also fixed (not a wrong answer): the cue loop summed the preceding word
lengths for every cue match, so a long quote took quadratic time (12,000
words about 3 s on the round-8 template, 0.3 s now).

Clean this round:
- mixed str / int dict keys on 87 probes over every dict-taking entry point
  (the twelve round-8 ones, nested `from_dict` payloads, `Design` maps,
  `y_levels`, `relaxation_analysis`, payload builders): no leak;
- chains of up to ten ambiguous full stops in 60,000 random texts: whenever
  the wording commits to a reading, every one of the 2^n mixed split / join
  readings agrees (the two-reading rule is not fooled by mixtures);
- 150,000 wording token soups with brackets, dashes, NULs and abbreviations
  (no leak; modality, `p` and cue consistent);
- string metrics against independent DPs and difflib on 3,000 pairs (the
  integer Jaro transposition count is the documented *convention*),
  hypervolume against a grid count in 2-D and 3-D, the non-dominated filter
  against brute force, kappa against sklearn;
- span recovery on 2,000 random substrings, projection idempotent and
  feasible on every district design, the contract's JSON round trip;
- scale invariance at 1e+-6 (segment distance, bearing, polyline length,
  ambiguity, entropy, weight prior, hypervolume);
- NaN / inf on about 60 numeric entry points; every optional parameter probed is
  live; no aliasing of inputs or results.

A 33-mutant run on the new and untouched code (abbreviation joins, day
names, the modal exception, bracket / dash asides, auxiliaries, the lone
negator, *ought* interjections, offsets, geometry, hypervolume, time windows,
weight prior, idioms, comparatives, negation tables, the month *May*) left
ten survivors. Two exposed over-narrow rules in the fix itself, now widened:
a street *St.* after an ALL-CAPS name ("MAIN ST. Vehicles") and "Main St.
No. 5" are refused like "opp. No. 5". A second, 37-mutant run on the final
code (adding the colon-list continuation and the Jaro window) left two. Tier
45 kills all of them: *Rd.* / *etc.* before a bracket, the modal exception,
minimal dash and *ought* pairing, the governing modal's position, *May*
before a number, the colon list stopping at the full stop, and the Jaro
match window `floor(max/2) - 1` ("ab" / "ba" scores 0).

---

## References

- Levenshtein (1966); Damerau (1964); Jaro (1989); Winkler (1990); Ratcliff &
  Obershelp (1988); Russell & Odell — Soundex (US Census rules); Philips (1990)
  — Metaphone.
- Robertson & Zaragoza (2009) — BM25; Cormack, Clarke & Büttcher (2009) —
  reciprocal rank fusion.
- Allen (1983) — *Maintaining knowledge about temporal intervals*.
- Kahn (1962) — topological sorting.
- Zitzler & Thiele (1998) — hypervolume; While et al. (2006) — slicing
  (HSO).
- Cohen (1960) — kappa.
- UK Traffic Regulation Orders / Irish S.I. drafting conventions for
  amendment, revocation and supersession.

---

*Pure standard library. `policy_compiler_template.py` — `__version__ = "1.0.0"`,
all public names in `__all__`, 309 oracle-based tests.*
