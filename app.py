
"""JIE42903 Evolutionary Computing
TV Scheduling Using Genetic Algorithm
Complete single-file Streamlit application.
"""

import io
import random
import pandas as pd
import streamlit as st

# ==========================================
# 1. ORIGINAL CSV DATASET
# ==========================================

ORIGINAL_CSV = """Type of Program,Hour 6,Hour 7,Hour 8,Hour 9,Hour 10,Hour 11,Hour 12,Hour 13,Hour 14,Hour 15,Hour 16,Hour 17,Hour 18,Hour 19,Hour 20,Hour 21,Hour 22,Hour 23
news,0.1,0.1,0.4,0.3,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.5,0.5,0.4,0.3,0.2,0.1,0.2
live_soccer,0.0,0.0,0.0,0.2,0.1,0.3,0.2,0.1,0.4,0.3,0.4,0.5,0.4,0.6,0.4,0.3,0.4,0.3
movie_a,0.1,0.1,0.2,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.3,0.5,0.3,0.4
movie_b,0.2,0.1,0.1,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.5,0.4,0.3,0.4,0.5
reality_show,0.3,0.4,0.3,0.4,0.4,0.5,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.2,0.2,0.3
tv_series_a,0.2,0.3,0.2,0.1,0.1,0.2,0.2,0.4,0.4,0.3,0.3,0.3,0.5,0.6,0.4,0.5,0.4,0.3
tv_series_b,0.1,0.2,0.3,0.3,0.2,0.3,0.3,0.1,0.4,0.3,0.4,0.3,0.5,0.3,0.4,0.6,0.4,0.3
music_program,0.3,0.3,0.3,0.2,0.2,0.1,0.2,0.4,0.3,0.3,0.3,0.3,0.2,0.3,0.2,0.3,0.5,0.3
documentary,0.3,0.3,0.4,0.3,0.2,0.2,0.3,0.4,0.4,0.3,0.2,0.2,0.2,0.1,0.1,0.3,0.3,0.2
Boxing,0.4,0.3,0.3,0.2,0.2,0.1,0.1,0.1,0.1,0.3,0.3,0.3,0.2,0.3,0.4,0.3,0.4,0.6
"""

# ==========================================
# 2. MODIFY PROGRAM RATINGS
# ==========================================

DEFAULT_CHANGES = [
    ("news", "Hour 8", 0.6),
    ("live_soccer", "Hour 20", 0.8),
    ("movie_a", "Hour 21", 0.7),
    ("reality_show", "Hour 15", 0.6),
    ("documentary", "Hour 11", 0.4),
    ("Boxing", "Hour 23", 0.8)
]


def prepare_data():
    original = pd.read_csv(io.StringIO(ORIGINAL_CSV))
    modified = original.copy()
    log = []

    for program, hour, new_rating in DEFAULT_CHANGES:
        mask = modified["Type of Program"] == program

        if mask.sum() != 1 or hour not in modified.columns:
            raise ValueError(
                f"Invalid program or hour: {program}, {hour}"
            )

        old_rating = float(
            modified.loc[mask, hour].iloc[0]
        )

        modified.loc[mask, hour] = new_rating

        log.append({
            "Program": program,
            "Time Slot": hour,
            "Original Rating": old_rating,
            "Modified Rating": new_rating
        })

    return original, modified, pd.DataFrame(log)


# ==========================================
# 3. GENETIC ALGORITHM FUNCTIONS
# ==========================================

def fitness(chromosome, matrix):
    """Calculate the total rating of a schedule."""
    return sum(
        matrix[gene][slot]
        for slot, gene in enumerate(chromosome)
    )


def tournament(population, matrix, rng, k=3):
    """Select a parent using tournament selection."""
    candidates = rng.sample(
        population,
        min(k, len(population))
    )

    return max(
        candidates,
        key=lambda x: fitness(x, matrix)
    ).copy()


def crossover(parent1, parent2, rate, rng):
    """Perform one-point crossover."""
    if rng.random() < rate:
        point = rng.randrange(1, len(parent1))

        child1 = (
            parent1[:point] + parent2[point:]
        )

        child2 = (
            parent2[:point] + parent1[point:]
        )

        return child1, child2

    return parent1.copy(), parent2.copy()


def mutate(child, rate, n_programs, rng):
    """Randomly modify genes in a chromosome."""
    for i in range(len(child)):
        if rng.random() < rate:
            child[i] = rng.randrange(n_programs)

    return child


def run_ga(
    matrix,
    crossover_rate,
    mutation_rate,
    generations=100,
    population_size=50,
    elite_count=2,
    seed=42
):
    """Run the Genetic Algorithm."""
    rng = random.Random(seed)

    n_programs = len(matrix)
    n_slots = len(matrix[0])

    population = [
        [
            rng.randrange(n_programs)
            for _ in range(n_slots)
        ]
        for _ in range(population_size)
    ]

    history = []

    for generation in range(generations):

        population.sort(
            key=lambda c: fitness(c, matrix),
            reverse=True
        )

        history.append(
            fitness(population[0], matrix)
        )

        next_population = [
            c.copy()
            for c in population[:elite_count]
        ]

        while len(next_population) < population_size:

            parent1 = tournament(
                population, matrix, rng
            )

            parent2 = tournament(
                population, matrix, rng
            )

            child1, child2 = crossover(
                parent1,
                parent2,
                crossover_rate,
                rng
            )

            next_population.append(
                mutate(
                    child1,
                    mutation_rate,
                    n_programs,
                    rng
                )
            )

            if len(next_population) < population_size:
                next_population.append(
                    mutate(
                        child2,
                        mutation_rate,
                        n_programs,
                        rng
                    )
                )

        population = next_population

    best = max(
        population,
        key=lambda c: fitness(c, matrix)
    )

    best_fitness = fitness(best, matrix)
    history.append(best_fitness)

    return best, best_fitness, history


# ==========================================
# 4. OUTPUT FUNCTIONS
# ==========================================

def schedule_table(chromosome, programs, hours, matrix):
    return pd.DataFrame({
        "Time Slot": [
            f"{int(h.split()[-1]):02d}:00"
            for h in hours
        ],
        "Program": [
            programs[g] for g in chromosome
        ],
        "Rating": [
            matrix[g][i]
            for i, g in enumerate(chromosome)
        ]
    })


def csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8")


# ==========================================
# 5. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TV Scheduling | Genetic Algorithm",
    page_icon="📺",
    layout="wide"
)

st.title("📺 TV Scheduling Using Genetic Algorithm")

st.caption(
    "JIE42903 — Evolutionary Computing | Individual Assignment"
)

st.info(
    "Assignment note: The instructions specify a default "
    "mutation rate of 0.20, but also mention a range "
    "of 0.01–0.05. Confirm the intended range with "
    "your lecturer."
)

# ==========================================
# 6. LOAD DATASET
# ==========================================

original, modified, change_log = prepare_data()

hours = [
    c for c in modified.columns
    if c.startswith("Hour ")
]

programs = modified["Type of Program"].tolist()

matrix = modified[hours].astype(
    float
).values.tolist()


# ==========================================
# 7. DATASET DISPLAY
# ==========================================

st.header("1. Dataset and Rating Modifications")

tab1, tab2, tab3 = st.tabs([
    "Original Dataset",
    "Modified Dataset",
    "Rating Changes"
])

with tab1:
    st.subheader("Original Program Ratings")
    st.dataframe(
        original,
        use_container_width=True
    )

with tab2:
    st.subheader("Modified Program Ratings")
    st.dataframe(
        modified,
        use_container_width=True
    )

    st.download_button(
        "Download Modified CSV",
        csv_bytes(modified),
        "program_ratings_modified.csv",
        "text/csv"
    )

with tab3:
    st.subheader("Documented Rating Changes")
    st.dataframe(
        change_log,
        use_container_width=True
    )

    st.download_button(
        "Download Rating Change Log",
        csv_bytes(change_log),
        "rating_changes.csv",
        "text/csv"
    )

st.divider()


# ==========================================
# 8. GENETIC ALGORITHM PARAMETERS
# ==========================================

st.header("2. Genetic Algorithm Configuration")

with st.sidebar:
    st.header("GA Settings")

    generations = st.number_input(
        "Number of Generations",
        min_value=1,
        max_value=2000,
        value=100,
        step=10
    )

    pop_size = st.number_input(
        "Population Size",
        min_value=4,
        max_value=1000,
        value=50,
        step=2
    )

    elite_count = st.number_input(
        "Elitism Count",
        min_value=1,
        max_value=int(pop_size) - 1,
        value=min(2, int(pop_size) - 1)
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42
    )

col1, col2 = st.columns(2)

with col1:
    co_r = st.slider(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.8,
        step=0.01
    )

with col2:
    mut_r = st.slider(
        "Mutation Rate (MUT_R)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01
    )

# ==========================================
# 9. RUN SINGLE GA SCHEDULE
# ==========================================

if st.button(
    "Run Genetic Algorithm",
    type="primary"
):

    best, score, history = run_ga(
        matrix,
        co_r,
        mut_r,
        generations=int(generations),
        population_size=int(pop_size),
        elite_count=int(elite_count),
        seed=int(seed)
    )

    st.session_state["single"] = (
        best,
        score,
        history,
        co_r,
        mut_r
    )


if "single" in st.session_state:

    best, score, history, old_co, old_mut = (
        st.session_state["single"]
    )

    st.success("Genetic Algorithm completed!")

    st.metric(
        "Best Schedule Fitness",
        f"{score:.4f}"
    )

    st.caption(
        f"CO_R = {old_co:.2f} | MUT_R = {old_mut:.2f}"
    )

    st.subheader("Optimised TV Schedule")

    schedule = schedule_table(
        best,
        programs,
        hours,
        matrix
    )

    st.dataframe(
        schedule,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Best Schedule",
        csv_bytes(schedule),
        "best_tv_schedule.csv",
        "text/csv"
    )

    st.subheader("Fitness Progression")

    fitness_df = pd.DataFrame({
        "Best Fitness": history
    })

    fitness_df.index.name = "Generation"

    st.line_chart(fitness_df)

st.divider()


# ==========================================
# 10. THREE EXPERIMENTAL TRIALS
# ==========================================

st.header("3. Experimental Trials")

st.write(
    "Configure different crossover and mutation "
    "rates for each experiment."
)

defaults = [
    (0.20, 0.01),
    (0.50, 0.03),
    (0.80, 0.05)
]

trial_settings = []

columns = st.columns(3)

for i, column in enumerate(columns):

    with column:

        st.subheader(f"Trial {i + 1}")

        trial_co = st.number_input(
            "Crossover Rate",
            min_value=0.0,
            max_value=0.95,
            value=defaults[i][0],
            step=0.01,
            key=f"trial_co_{i}"
        )

        trial_mut = st.number_input(
            "Mutation Rate",
            min_value=0.01,
            max_value=0.05,
            value=defaults[i][1],
            step=0.01,
            format="%.2f",
            key=f"trial_mut_{i}"
        )

        trial_settings.append(
            (trial_co, trial_mut)
        )


# ==========================================
# 11. EXECUTE THREE TRIALS
# ==========================================

if st.button(
    "Run All Three Trials",
    type="primary"
):

    results = []
    schedules = {}
    histories = {}

    for i, (trial_co, trial_mut) in enumerate(
        trial_settings, start=1
    ):

        chromosome, score, history = run_ga(
            matrix,
            trial_co,
            trial_mut,
            generations=int(generations),
            population_size=int(pop_size),
            elite_count=int(elite_count),
            seed=int(seed)
        )

        trial_name = f"Trial {i}"

        results.append({
            "Trial": trial_name,
            "CO_R": trial_co,
            "MUT_R": trial_mut,
            "Best Fitness": round(score, 4)
        })

        schedules[trial_name] = schedule_table(
            chromosome,
            programs,
            hours,
            matrix
        )

        histories[trial_name] = history

    st.session_state["experiments"] = (
        pd.DataFrame(results),
        schedules,
        histories
    )


# ==========================================
# 12. EXPERIMENT COMPARISON
# ==========================================

if "experiments" in st.session_state:

    results_df, schedules, histories = (
        st.session_state["experiments"]
    )

    st.header("4. Experiment Results")

    st.subheader("Trial Comparison Table")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Fitness Score Comparison")

    st.bar_chart(
        results_df.set_index("Trial")[
            ["Best Fitness"]
        ]
    )

    st.subheader("Fitness Convergence")

    convergence_df = pd.DataFrame(histories)
    convergence_df.index.name = "Generation"

    st.line_chart(convergence_df)

    # Identify the best trial

    best_trial = results_df.loc[
        results_df["Best Fitness"].idxmax()
    ]

    st.success(
        f"Best Observed Trial: {best_trial['Trial']} | "
        f"Fitness: {best_trial['Best Fitness']:.4f} | "
        f"CO_R: {best_trial['CO_R']:.2f} | "
        f"MUT_R: {best_trial['MUT_R']:.2f}"
    )

    st.caption(
        "The best trial is based on these experimental "
        "runs. Additional random seeds are recommended "
        "for stronger conclusions."
    )

    st.download_button(
        "Download Experiment Comparison",
        csv_bytes(results_df),
        "experiment_comparison.csv",
        "text/csv"
    )

    # Display each trial schedule

    st.header("5. Generated TV Schedules")

    for name, schedule in schedules.items():

        with st.expander(
            f"{name} - TV Schedule",
            expanded=True
        ):

            st.dataframe(
                schedule,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                f"Download {name} Schedule",
                csv_bytes(schedule),
                f"{name.lower().replace(' ', '_')}_schedule.csv",
                "text/csv",
                key=f"download_{name}"
            )


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "TV Scheduling Using Genetic Algorithm | "
    "JIE42903 Evolutionary Computing"
)

st.caption(
    "Model assumption: Programs may repeat across "
    "different time slots. Program duration and "
    "broadcasting restrictions are not enforced."
)
