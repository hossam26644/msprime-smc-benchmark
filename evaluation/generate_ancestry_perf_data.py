import time
import concurrent.futures

import cpuinfo
import msprime
import numpy as np
import click


@click.group()
def cli():
    pass

def run(L, N, num_samples, r, model):
    start = time.process_time()
    ts = msprime.sim_ancestry(
            samples=num_samples,
            population_size=N,
            sequence_length=L,
            recombination_rate=r,
            model=model)
    end = time.process_time()
    return (max(ts.tables.nodes.time), ts.num_trees, end - start)

def csv(x):
    return ",".join(map(str, x)) + "\n"

def process_one_rep(params):
    L, N, ns, model = params
    h, n, t = run(L, N, ns, model)
    return [N, L, ns, h, n, t]

def process_one_rep_vary_k(params):
    L, N, ns, r, k = params
    if k == '''Hudson''':
        model = msprime.StandardCoalescent()
    elif type(k) == int:
        model=msprime.SmcKApproxCoalescent(hull_offset=k)
    elif '''Hybrid''' in k:
        duration = int(k.split('Hybrid')[-1])
        model = [msprime.StandardCoalescent(duration=duration),
                 msprime.SmcKApproxCoalescent()]
    else:
        raise ValueError("Unknown model")
    h, n, t = run(L, N, ns, r, model)
    return [N, L, r, ns, k, h, n, t]

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

def run_varying_seq_len_sims_generic(seq_len_range=[10, 10e9],samples_range=[10,1000],
    model="hudson", N=1000, R=1e-7, outfile="data/ancestry-perf-varying-seq-len.csv"):
    # Run a single simulation with fixed sample size and varying sequence length

    num_reps = 3

    with open(outfile, "w") as f:
        f.write(csv(["N", "L", "num_samples", "height", "num_trees", "time"]))

        all_tasks = []
        for seq_len in np.logspace(np.log10(seq_len_range[0]), np.log10(seq_len_range[1]), num=20):
            seq_len = int(seq_len)
            for samples in np.logspace(np.log10(samples_range[0]), np.log10(samples_range[1]), num=5):
                samples = int(samples)
                for _ in range(num_reps):
                    all_tasks.append((seq_len, N, samples, model))

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
def run_varying_seq_len_sims(outfile="data/ancestry-perf-varying-seq-len.csv", model="hudson"):
    run_varying_seq_len_sims_generic(outfile=outfile, model=model)

@click.command()
def run_varying_seq_len_sims_smc():
    outfile = "data/ancestry-perf-varying-seq-len-smc.csv"
    model=msprime.SmcKApproxCoalescent()
    run_varying_seq_len_sims_generic(outfile=outfile, model=model, seq_len_range=[10,10e9])

def run_varying_sample_size_genric(seq_len=10e7,num_samples_range=[10,10000],
    model="hudson", N=1000, R=1e-5, outfile="data/ancestry-perf-varying-seq-len.csv"):
    # Run a single simulation with fixed sample size and varying sequence length

    num_reps = 3

    with open(outfile, "w") as f:
        f.write(csv(["N", "L", "num_samples", "height", "num_trees", "time"]))

        all_tasks = []
        for num_samples in np.logspace(np.log10(num_samples_range[0]), np.log10(num_samples_range[1]), num=50):
            num_samples = int(num_samples)
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
def run_varying_sample_size():
    outfile = "data/ancestry-perf-varying-sample-size.csv"
    model="hudson"
    run_varying_sample_size_genric(outfile=outfile, model=model)

@click.command()
def run_varying_sample_size_smc():
    outfile = "data/ancestry-perf-varying-sample-size-smc.csv"
    model=msprime.SmcKApproxCoalescent()
    run_varying_sample_size_genric(outfile=outfile, model=model)

def run_varying_k_sims_inner(k_range=[1, 5e8],samples_range=[10,1000], seq_len=5e8,
     N=1000, r=1e-7, outfile="data/ancestry-perf-varying-k.csv", ksamples=25, num_reps=3, file_write=True):
    # Run a single simulation with fixed sample size and varying sequence length


    num_reps = 3

    file_mode = "w" if file_write else "a"
    with open(outfile, file_mode) as f:
        if file_write:
            f.write(csv(["N", "L", "r", "num_samples", "k", "height", "num_trees", "time"]))

        all_tasks = []

        for samples in np.logspace(np.log10(samples_range[0]), np.log10(samples_range[1]), num=5):
            samples = int(samples)
            k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), num=ksamples)

            if k_values[0] == 1: k_values[0] = 0
            for k in k_values:
                # Your code here
                k = int(k)

                for _ in range(num_reps):
                    all_tasks.append((seq_len, N, samples, r, k))

            for _ in range(num_reps):
                all_tasks.append((seq_len, N, samples, r, 'Hudson')) # for standard coalescent

        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                # Submit all tasks and get futures
                future_results = {executor.submit(process_one_rep_vary_k, params): params for params in all_tasks}

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
def run_varying_k_sims():
    # Run a single simulation with fixed sample size and varying sequence length
    k_range=[1, 5e8]
    samples_range=[10,1000]
    seq_len=5e8
    N=1000
    R=1e-7
    outfile="data/ancestry-perf-varying-k.csv"
    ksamples=25
    run_varying_k_sims_inner(k_range=k_range, samples_range=samples_range, seq_len=seq_len,
        N=N, R=R, outfile=outfile, ksamples=ksamples)

def run_hybrid_sims_inner(samples_range=[1,10000], seq_len=5e2,
     N=1000, r=1e-5, gens_range=[10,100000], outfile="data/ancestry-perf-hybrid.csv", num_reps=3, file_write=False):


    run_varying_k_sims_inner(samples_range=samples_range, seq_len=seq_len, outfile=outfile,
        N=N, r=r, k_range=[1, 1], ksamples=1, num_reps=num_reps, file_write=False)

    with open(outfile, "a") as f:
        all_tasks = []

        for samples in np.logspace(np.log10(samples_range[0]), np.log10(samples_range[1]), num=5):
            samples = int(samples)
            for gens in np.logspace(np.log10(gens_range[0]), np.log10(gens_range[1]), num=5):
                gens = int(gens)
                #print(gens)
                for _ in range(num_reps):
                    all_tasks.append((seq_len, N, samples, r, 'Hybrid' + str(gens)))

        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                # Submit all tasks and get futures
                future_results = {executor.submit(process_one_rep_vary_k, params): params for params in all_tasks}

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
def run_hybrid_sims():
    run_hybrid_sims_inner()

@click.command()
def run_panels():
    samples_range=[1,10000]
    num_reps=50
    r=1e-9
    seqlen_range=np.logspace(np.log10(10), np.log10(1e5), num=5)
    pop_range=np.logspace(np.log10(1), np.log10(1e6), num=7)
    file = "data/ancestry-perf-panels.csv"
    #clear file
    with open(file, "w") as f:
            f.write(csv(["N", "L", "r", "num_samples", "k", "height", "num_trees", "time"]))

    for L in seqlen_range:
        for N in pop_range:
            #if L==1e5 and N==10000.0: continue
            #if L*N>1e8: continue
            print(f"running simlution for N={N} and L={L}")
            run_hybrid_sims_inner(samples_range=samples_range, seq_len=L, N=N, r=r,
                gens_range=[10,100000], outfile="data/ancestry-perf-panels.csv", num_reps=num_reps, file_write=False)


@click.command()
def run_for_perf():
    run(10000, 1000000, 10000, msprime.SmcKApproxCoalescent())

cli.add_command(run_sims)
cli.add_command(run_sims_smc)
cli.add_command(run_varying_seq_len_sims)
cli.add_command(run_varying_seq_len_sims_smc)
cli.add_command(run_varying_sample_size)
cli.add_command(run_varying_sample_size_smc)
cli.add_command(run_varying_k_sims)
cli.add_command(run_hybrid_sims)
cli.add_command(run_panels)
cli.add_command(run_for_perf)


if __name__ == "__main__":
    cli()