

def fitness(chromosome, matrix):
    return sum(matrix[gene][slot] for slot, gene in enumerate(chromosome))


def tournament(population, matrix, rng, k=3):
    candidates = rng.sample(population, min(k, len(population)))
    return max(candidates, key=lambda x: fitness(x, matrix)).copy()


def crossover(a, b, rate, rng):
    if rng.random() < rate:
        cut = rng.randrange(1, len(a))
        return a[:cut] + b[cut:], b[:cut] + a[cut:]
    return a.copy(), b.copy()


def mutate(child, rate, n_programs, rng):
    for i in range(len(child)):
        if rng.random() < rate:
            child[i] = rng.randrange(n_programs)
    return child


def run_ga(matrix, crossover_rate, mutation_rate, generations=100,
           population_size=50, elite_count=2, seed=42):
    rng = random.Random(seed)
    n_programs = len(matrix)
    n_slots = len(matrix[0])
    population = [[rng.randrange(n_programs) for _ in range(n_slots)]
                  for _ in range(population_size)]
    history = []
    for _ in range(generations):
        population.sort(key=lambda c: fitness(c, matrix), reverse=True)
        history.append(fitness(population[0], matrix))
        next_population = [c.copy() for c in population[:elite_count]]
        while len(next_population) < population_size:
            a = tournament(population, matrix, rng)
            b = tournament(population, matrix, rng)
            c1, c2 = crossover(a, b, crossover_rate, rng)
            next_population.append(mutate(c1, mutation_rate, n_programs, rng))
            if len(next_population) < population_size:
                next_population.append(mutate(c2, mutation_rate, n_programs, rng))
        population = next_population
    best = max(population, key=lambda c: fitness(c, matrix))
    history.append(fitness(best, matrix))
    return best, fitness(best, matrix), history


def schedule_table(chromosome, programs, hour_columns, matrix):
    return pd.DataFrame({
        "Time Slot": [f"{int(h.split()[-1]):02d}:00" for h in hour_columns],
        "Program": [programs[g] for g in chromosome],
        "Rating": [matrix[g][i] for i, g in enumerate(chromosome)],
    })


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="TV Scheduling | Genetic Algorithm", page_icon="📺", layout="wide")
st.title("📺 TV Scheduling Using Genetic Algorithm")
st.caption("JIE42903 — Evolutionary Computing | Individual Assignment")
st.info("The brief conflicts on mutation rate: it specifies default 0.20 but also a 0.01–0.05 range. "
        "This app permits 0.00–1.00 for the default and experiments; confirm the lecturer's intended range.")

original, modified, change_log = prepare_data()
hours = [c for c in modified.columns if c.startswith("Hour ")]
programs = modified["Type of Program"].tolist()
matrix = modified[hours].astype(float).values.tolist()

with st.expander("1. Dataset and documented rating changes", expanded=False):
    st.write("Original data (10 programs × 18 hourly slots):")
    st.dataframe(original, use_container_width=True)
    st.write("Modified ratings (hypothetical viewer-preference assumptions):")
    st.dataframe(change_log, use_container_width=True)
    st.dataframe(modified, use_container_width=True)
    st.download_button("Download modified CSV", csv_bytes(modified),
                       "program_ratings_modified.csv", "text/csv")
    st.download_button("Download change log", csv_bytes(change_log),
                       "rating_changes.csv", "text/csv")

st.subheader("2. Genetic Algorithm settings")
with st.sidebar:
    st.header("GA configuration")
    generations = st.number_input("Generations", min_value=1, max_value=2000, value=100, step=10)
    pop_size = st.number_input("Population size", min_value=4, max_value=1000, value=50, step=2)
    elite_count = st.number_input("Elitism count", min_value=1, max_value=int(pop_size)-1, value=min(2, int(pop_size)-1))
    seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=42)
    st.caption("Same seed and configuration make results reproducible.")

co = st.slider("CO_R — crossover rate", 0.0, 0.95, 0.80, 0.01)
mut = st.slider("MUT_R — mutation rate", 0.0, 1.0, 0.20, 0.01)
if st.button("Run single schedule", type="primary"):
    best, score, hist = run_ga(matrix, co, mut, int(generations), int(pop_size), int(elite_count), int(seed))
    st.session_state["single"] = (best, score, hist, co, mut)

if "single" in st.session_state:
    best, score, hist, old_co, old_mut = st.session_state["single"]
    st.metric("Best schedule fitness", f"{score:.3f}")
    st.caption(f"Results for CO_R={old_co:.2f}, MUT_R={old_mut:.2f}")
    table = schedule_table(best, programs, hours, matrix)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.line_chart(pd.DataFrame({"Best fitness": hist}), x_label="Generation", y_label="Fitness")
    st.download_button("Download schedule CSV", csv_bytes(table), "single_schedule.csv", "text/csv")

st.subheader("3. Three experimental trials")
st.write("Set separate crossover and mutation rates for each trial, then run all three.")
defaults = [(0.20, 0.01), (0.50, 0.03), (0.80, 0.05)]
settings = []
cols = st.columns(3)
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Trial {i+1}**")
        t_co = st.number_input("CO_R", min_value=0.0, max_value=0.95,
                               value=defaults[i][0], step=0.01, key=f"co_{i}")
        t_mut = st.number_input("MUT_R", min_value=0.0, max_value=1.0,
                                value=defaults[i][1], step=0.01, format="%.2f", key=f"mut_{i}")
        settings.append((t_co, t_mut))

if st.button("Run all three trials"):
    results, schedules, histories = [], {}, {}
    for i, (t_co, t_mut) in enumerate(settings, 1):
        chromosome, score, hist = run_ga(
            matrix, t_co, t_mut, int(generations), int(pop_size), int(elite_count), int(seed))
        label = f"Trial {i}"
        results.append({"Trial": label, "CO_R": t_co, "MUT_R": t_mut, "Best Fitness": round(score, 4)})
        schedules[label] = schedule_table(chromosome, programs, hours, matrix)
        histories[label] = hist
    st.session_state["experiments"] = (pd.DataFrame(results), schedules, histories)

if "experiments" in st.session_state:
    results, schedules, histories = st.session_state["experiments"]
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("Trial")[["Best Fitness"]], y_label="Total fitness")
    st.line_chart(pd.DataFrame(histories), x_label="Generation", y_label="Best fitness")
    winner = results.loc[results["Best Fitness"].idxmax()]
    st.success(f"Best observed: {winner['Trial']} | Fitness {winner['Best Fitness']:.3f} | "
               f"CO_R={winner['CO_R']:.2f}, MUT_R={winner['MUT_R']:.2f}")
    st.caption("One run per trial is not proof of the generally optimal parameters. "
               "Try additional seeds for stronger analysis.")
    st.download_button("Download trial comparison", csv_bytes(results), "trial_comparison.csv", "text/csv")
    for name, schedule in schedules.items():
        with st.expander(f"{name} — 18-hour schedule", expanded=True):
            st.dataframe(schedule, use_container_width=True, hide_index=True)
            st.download_button(f"Download {name} CSV", csv_bytes(schedule),
                               f"{name.lower().replace(' ', '_')}_schedule.csv", "text/csv", key=f"dl_{name}")

st.caption("Model assumption: a program can repeat across slots; no program duration or broadcasting constraints are enforced.")
