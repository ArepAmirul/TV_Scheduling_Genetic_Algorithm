"""JIE42903 Evolutionary Computing — TV Scheduling Genetic Algorithm.
Single-file Streamlit app. Original ratings are embedded below.
"""
import io
import random
import pandas as pd
import streamlit as st

ORIGINAL_CSV = 'Type of Program,Hour 6,Hour 7,Hour 8,Hour 9,Hour 10,Hour 11,Hour 12,Hour 13,Hour 14,Hour 15,Hour 16,Hour 17,Hour 18,Hour 19,Hour 20,Hour 21,Hour 22,Hour 23\nnews,0.1,0.1,0.4,0.3,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.5,0.5,0.4,0.3,0.2,0.1,0.2\nlive_soccer,0.0,0.0,0.0,0.2,0.1,0.3,0.2,0.1,0.4,0.3,0.4,0.5,0.4,0.6,0.4,0.3,0.4,0.3\nmovie_a,0.1,0.1,0.2,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.3,0.5,0.3,0.4\nmovie_b,0.2,0.1,0.1,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.5,0.4,0.3,0.4,0.5\nreality_show,0.3,0.4,0.3,0.4,0.4,0.5,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.2,0.2,0.3\ntv_series_a,0.2,0.3,0.2,0.1,0.1,0.2,0.2,0.4,0.4,0.3,0.3,0.3,0.5,0.6,0.4,0.5,0.4,0.3\ntv_series_b,0.1,0.2,0.3,0.3,0.2,0.3,0.3,0.1,0.4,0.3,0.4,0.3,0.5,0.3,0.4,0.6,0.4,0.3\nmusic_program,0.3,0.3,0.3,0.2,0.2,0.1,0.2,0.4,0.3,0.3,0.3,0.3,0.2,0.3,0.2,0.3,0.5,0.3\ndocumentary,0.3,0.3,0.4,0.3,0.2,0.2,0.3,0.4,0.4,0.3,0.2,0.2,0.2,0.1,0.1,0.3,0.3,0.2\nBoxing,0.4,0.3,0.3,0.2,0.2,0.1,0.1,0.1,0.1,0.3,0.3,0.3,0.2,0.3,0.4,0.3,0.4,0.6\n'
DEFAULT_CHANGES = [
    ("news", "Hour 8", 0.6),
    ("live_soccer", "Hour 20", 0.8),
    ("movie_a", "Hour 21", 0.7),
    ("reality_show", "Hour 15", 0.6),
    ("documentary", "Hour 11", 0.4),
    ("Boxing", "Hour 23", 0.8),
]


def prepare_data():
    original = pd.read_csv(io.StringIO(ORIGINAL_CSV))
    modified = original.copy()
    log = []
    for program, hour, new in DEFAULT_CHANGES:
        mask = modified["Type of Program"] == program
        if mask.sum() != 1 or hour not in modified.columns:
            raise ValueError(f"Invalid program or hour: {program}, {hour}")
        old = float(modified.loc[mask, hour].iloc[0])
        modified.loc[mask, hour] = new
        log.append({"Program": program, "Time Slot": hour,
                    "Original Rating": old, "Modified Rating": new})
    return original, modified, pd.DataFrame(log)


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


st.caption("Model assumption: a program can repeat across slots; no program duration or broadcasting constraints are enforced.")
