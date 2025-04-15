import time
import concurrent.futures

import cpuinfo
import msprime
import numpy as np
import click


@click.group()
def cli():
    pass

def run(L, N, num_samples, model):
    start = time.process_time()
    ts = msprime.sim_ancestry(
            samples=num_samples,
            population_size=N,
            sequence_length=L,
            recombination_rate=1e-8,
            model=model)
    end = time.process_time()
    return (max(ts.tables.nodes.time), ts.num_trees, end - start)

def csv(x):
    return ",".join(map(str, x)) + "\n"

def process_one_rep(params):
    L, N, ns, model = params
    h, n, t = run(L, N, ns, model)
    return [N, L, ns, h, n, t]

def run_sims_generic(outfile="data/ancestry-perf.csv",
             cpu_file="data/ancestry_perf_cpu.txt",
             model="hudson"):

    outfile = outfile
    num_reps = 3
    num_samples = [10, 100]
    num_samples = [1000, 100000]
    L_VALS = [1e6, 5e6, 1e7, 5e7, 1e8]
    N_VALS = [1000, 5000, 10000, 50000, 100000, 200000, 300000]
    ln_pairs = [(L, N) for L in L_VALS for N in N_VALS]
    ln_pairs.sort(key=lambda x: x[0] * x[1])

    cpu = cpuinfo.get_cpu_info()
    with open(cpu_file, "w") as f:
        for k, v in cpu.items():
            print(k, "\t", v, file=f)



    with open(outfile, "w") as f:
        f.write(csv(["N", "L", "num_samples", "height", "num_trees", "time"]))

        # Create a list of all parameter combinations
        all_tasks = []
        for L, N in ln_pairs:
            for ns in num_samples:
                if L * N * np.log10(ns) < 3e12:
                    for _ in range(num_reps):
                        all_tasks.append((L, N, ns, model))

        # Process tasks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(process_one_rep, params): params for params in all_tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result()
                    f.write(csv(result))
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

@click.command()
def run_sims(outfile="data/ancestry-perf.csv",
             cpu_file="data/ancestry_perf_cpu.txt",
             model="hudson"):
    run_sims_generic(outfile=outfile, cpu_file=cpu_file, model=model)

@click.command()
def run_sims_smc():
    outfile = "data/ancestry-perf-smc.csv"
    cpu_file = "data/ancestry_perf_cpu-smc.txt"
    model=msprime.SmcKApproxCoalescent()
    run_sims_generic(outfile=outfile, cpu_file=cpu_file, model=model)

def run_fixed_sample_size_sims_generic(seq_len_range=[10, 10e8],num_samples=10,
    model="hudson", N=1000, R=1e-5, outfile="data/ancestry-perf-fixed-sample-size.csv"):
    # Run a single simulation with fixed sample size and varying sequence length

    num_reps = 3

    with open(outfile, "w") as f:
        f.write(csv(["N", "L", "num_samples", "height", "num_trees", "time"]))

        all_tasks = []
        for seq_len in np.logspace(np.log10(seq_len_range[0]), np.log10(seq_len_range[1]), num=50):
            seq_len = int(seq_len)
            for _ in range(num_reps):
                all_tasks.append((seq_len, N, num_samples, model))

        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                # Submit all tasks and get futures
                future_results = {executor.submit(process_one_rep, params): params for params in all_tasks}

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_results):
                    try:
                        result = future.result()
                        f.write(csv(result))
                        f.flush()
                    except Exception as exc:
                        params = future_results[future]
                        print(f"Task {params} generated an exception: {exc}")

@click.command()
def run_fixed_sample_size_sims(outfile="data/ancestry-perf-fixed-sample-size.csv", model="hudson"):
    run_fixed_sample_size_sims_generic(outfile=outfile, model=model)

@click.command()
def run_fixed_sample_size_sims_smc():
    outfile = "data/ancestry-perf-fixed-sample-size-smc.csv"
    model=msprime.SmcKApproxCoalescent()
    run_fixed_sample_size_sims_generic(outfile=outfile, model=model)

cli.add_command(run_sims)
cli.add_command(run_sims_smc)
cli.add_command(run_fixed_sample_size_sims)
cli.add_command(run_fixed_sample_size_sims_smc)

if __name__ == "__main__":
    cli()