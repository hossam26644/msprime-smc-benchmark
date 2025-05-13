import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import click

# Main text size is 9pt
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"legend.fontsize": 6})
plt.rcParams.update({"lines.markersize": 4})

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

@click.group()
def cli():
    pass


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


def two_panel_fig(**kwargs):
    # The columnwidth of the genetics format is ~250pt, which is
    # 3 15/32 inch, = 3.46
    width = 3.46
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, width / 2), **kwargs)
    ax1.set_title("(A)")
    ax2.set_title("(B)")
    return fig, (ax1, ax2)


def two_panel_fig_single_col(**kwargs):
    width = 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, width / 2), **kwargs)
    ax1.set_title("(A)")
    ax2.set_title("(B)")
    return fig, (ax1, ax2)


def ancestry_perf_plot(infile, outfile):
    """
    Plot the ancestry benchark.
    """

    df = pd.read_csv(infile, sep=",")

    # GRCh38, via stdpopsim
    human_rho = lambda L: int(4 * 10 ** 4 * L * 1e-8)
    human_chr1 = human_rho(248956422)

    # TAIR10, via stdpopsim
    r = 8.06e-10
    aratha_rho = lambda L: int(4 * 10 ** 4 * L * r)
    aratha_chr1 = aratha_rho(30427671)

    canfam_chr1 = 4 * 13000 * 122678785 * 7.636001498077e-09
    dromel_chr2l = 4 * 1720600 * 23513712 * 2.40462600791e-08
    print(f"Dromel rho {dromel_chr2l:.2g}")

    fig, axes = two_panel_fig_single_col()

    def annotate_rho(ax, rho, x_offset, y_offset, text):
        ax.axvline(rho / 4, color="0.8", linestyle="-.")
        ax.text(rho / 4 + x_offset, y_offset, text, fontstyle="italic")

    annotate_rho(axes[0], aratha_chr1, 100, 1.5, "Arabidopsis\nthaliana")
    annotate_rho(axes[1], human_chr1, -5000, 100, "Homo sapiens")
    annotate_rho(axes[1], canfam_chr1, 1000, 0, "Canis familiaris")
    # axes[1].axvline(dromel_chr2l)

    rgb = matplotlib.cm.get_cmap("Set1")(np.linspace(0.0, 1.0, len(set(df["N"]))))
    time_lims = [5, 1e12]
    for ax, tl in zip(axes, time_lims):
        legend_adds = [
            matplotlib.lines.Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                label=f"quadratic",
            )
        ]
        for ns, m in zip((1000, 100000), ("o", "v")):
            legend_adds.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color="grey",
                    marker=m,
                    linestyle="none",
                    label=f"n={ns}",
                )
            )
            ut = np.logical_and(df["time"] <= tl, df["num_samples"] == ns)
            # fit a qudratic with no intercept:
            # in R this would be lm(time ~ 0 + L : N + I((N*L)^2))
            rho = 4 * df["N"] * df["L"] / 1e8
            X = np.empty((sum(ut), 3))
            X[:, 0] = rho[ut]
            X[:, 1] = (rho ** 2)[ut]
            X[:, 2] = 1
            b, _, _, _ = np.linalg.lstsq(X, df["time"][ut], rcond=None)

            ax.set_xlabel("$N_e L$ (= scaled recombination rate $\\rho/4$)")

            def fitted_quadratic(x):
                return b[2] + b[0] * x + b[1] * (x ** 2)

            ax.set_ylabel("Time (seconds)")
            Nvals = sorted(list(set(df["N"])))
            for k, Nval in enumerate(Nvals):
                utN = np.logical_and(ut, df["N"] == Nval)
                if np.sum(utN) > 0:
                    sargs = {}
                    if m == "o":
                        sargs["label"] = f"$N_e={Nval}$"
                    ax.scatter(
                        rho[utN] / 4, df["time"][utN], color=rgb[k], marker=m, **sargs
                    )

            # xx = np.linspace(0, 1.05 * max(X[:, 0]), 51)
            try:
                xx = np.linspace(0, max(X[:, 0]), 51)
            except Exception as e:
                print(e) #when values are too large, I think
                continue

            # Note the two quadratic curves are not the same!
            pargs = {}
            if m == "o":
                pargs["label"] = "quadratic"
            ax.plot(xx / 4, fitted_quadratic(xx), color="black", **pargs)
            # print(
            #     f"Times less than {tl}: " f"{b[2]:.2f} + {b[0]} * rho + {b[1]} * rho^2"
            # )
            print(
                "Predicted time for DroMel chr2L with n =",
                ns,
                "=",
                # The quadratic is still fit to rho, so don't divide by 4 here
                fitted_quadratic(dromel_chr2l) / 3600,
                "hours",
            )
        max_x = np.max(rho[df["time"] <= tl] / 4)
        ticks = np.arange(
                0,
                1.1 * max_x,
                10 ** np.floor(np.log10(max_x)),
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(
                [f"{x:.0f}"for x in ticks]
        )

    prop = {"size": 7}
    axes[0].legend(handles=legend_adds, prop=prop)
    axes[1].legend(prop=prop)

    save(outfile)


def sequence_length_vs_time_combined(infile1, infile2, name1, name2, outfile):
    # Read the CSV files
    df1 = pd.read_csv(infile1)
    df2 = pd.read_csv(infile2)

    # Verify that the population size (N) is constant and the same in both files
    if len(df1['N'].unique()) != 1 or len(df2['N'].unique()) != 1:
        raise ValueError("Population size (N) is not consistent within one or both datasets")
    population_size1 = df1['N'].unique()[0]
    population_size2 = df2['N'].unique()[0]
    if population_size1 != population_size2:
        raise ValueError(f"Population size mismatch: {population_size1} != {population_size2}")

    # Get unique sample sizes from both datasets
    sample_sizes1 = sorted(df1['num_samples'].unique())
    sample_sizes2 = sorted(df2['num_samples'].unique())

    # Define a colormap for sample sizes
    cmap = plt.cm.get_cmap("viridis", max(len(sample_sizes1), len(sample_sizes2)))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot data from infile1 with solid lines
    for i, sample_size in enumerate(sample_sizes1):
        subset = df1[df1['num_samples'] == sample_size]
        grouped_data = subset.groupby('L')['time'].mean().reset_index()
        grouped_data = grouped_data.sort_values('L')
        plt.plot(grouped_data['L'], grouped_data['time'], linestyle='-', color=cmap(i),
                 linewidth=2, marker='o', markersize=5, label=f'{name1} (num_samples={sample_size})')

    # Plot data from infile2 with dashed lines
    for i, sample_size in enumerate(sample_sizes2):
        subset = df2[df2['num_samples'] == sample_size]
        grouped_data = subset.groupby('L')['time'].mean().reset_index()
        grouped_data = grouped_data.sort_values('L')
        plt.plot(grouped_data['L'], grouped_data['time'], linestyle='--', color=cmap(i),
                 linewidth=2, marker='o', markersize=5, label=f'{name2} (num_samples={sample_size})')

    # Set x-axis to log scale
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Sequence Length (L)')
    plt.ylabel('Time (seconds)')
    plt.title(f'Sequence Length vs Time: Population Size (N={population_size1})')

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    save(outfile)
    plt.close()

def sample_size_vs_time_combined(infile1, infile2, name1, name2, outfile):
 # Read the CSV files
    df1 = pd.read_csv(infile1)
    df2 = pd.read_csv(infile2)

    # Verify that all population sizes (N) are the same within each dataset
    if len(df1['N'].unique()) != 1 or len(df2['N'].unique()) != 1:
        raise ValueError("Population size (N) is not consistent within one or both datasets")

    # Get the population sizes
    population_size1 = df1['N'].unique()[0]
    population_size2 = df2['N'].unique()[0]

    seq_len1 = df1['L'].unique()[0]
    seq_len2 = df2['L'].unique()[0]
    assert seq_len1 == seq_len2
    assert population_size1 == population_size2
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Process first dataset
    grouped_data1 = df1.groupby('num_samples')['time'].mean().reset_index()
    grouped_data1 = grouped_data1.sort_values('num_samples')
    plt.plot(grouped_data1['num_samples'], grouped_data1['time'], 'r-', linewidth=2, marker='o',
             markersize=5, label=f'{name1}')

    # Process second dataset
    grouped_data2 = df2.groupby('num_samples')['time'].mean().reset_index()
    grouped_data2 = grouped_data2.sort_values('num_samples')
    plt.plot(grouped_data2['num_samples'], grouped_data2['time'], 'b-', linewidth=2, marker='o',
             markersize=5, label=f'{name2}')

    # Set x-axis to log scale
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('number of samples')
    plt.ylabel('Time (seconds)')
    plt.title(f'num_samples vs Time: popsize: ({population_size1}), sequence length: ({seq_len1})')

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    save(outfile)
    plt.close()

@click.command()
def k_vs_time(infile="data/ancestry-perf-varying-k.csv",
        outfile="k-vs-time"):
    # Read the CSV files
    df = pd.read_csv(infile)

    # Verify that the population size (N) is constant and the same in both files
    if len(df['N'].unique()) != 1:
        raise ValueError("Population size (N) is not consistent within the dataset")
    population_size = df['N'].unique()[0]

    if len(df['L'].unique()) != 1:
        raise ValueError("sequence length (L) is not consistent within the dataset")
    sequence_length = df['L'].unique()[0]

    # Get unique sample sizes from both datasets
    sample_sizes = sorted(df['num_samples'].unique())

    # Define a colormap for sample sizes
    cmap = plt.cm.get_cmap("viridis", len(sample_sizes))

    # Create the plot
    plt.figure(figsize=(10, 6))

    hudson_data = df[df['k'] == 'Hudson']
    smc_data = df[df['k'] != 'Hudson']

    smc_data['k'] = smc_data['k'].astype(int)

    for i, sample_size in enumerate(sample_sizes):
        subset = smc_data[smc_data['num_samples'] == sample_size]
        grouped_data = subset.groupby('k')['time'].mean().reset_index()
        grouped_data = grouped_data.sort_values('k')
        plt.plot(grouped_data['k'], grouped_data['time'], linestyle='-', color=cmap(i),
                 linewidth=2, marker='o', markersize=5, label=f'num_samples={sample_size}')

    # Plot data from Hudson with dashed horizontal lines
    for i, sample_size in enumerate(sample_sizes):
        subset = hudson_data[hudson_data['num_samples'] == sample_size]
        plt.axhline(y=subset['time'].mean(), linestyle='--', color=cmap(i),
            linewidth=2, label=f'Hudson (num_samples={sample_size})')


    # Set x-axis to log scale
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('k')
    plt.ylabel('Time (seconds)')
    plt.title(f'effect of K on time: Population Size (N={population_size}), Sequence Length (L={sequence_length})')

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    save(outfile)
    plt.close()

def hybrid_inner(infile="data/ancestry-perf-hybrid.csv",
        outfile="hybrid", df=None, givenax=None):
    # Read the CSV files
    if df is None: df = pd.read_csv(infile)

    # Verify that the population size (N) is constant and the same in both files
    if len(df['N'].unique()) != 1:
        raise ValueError("Population size (N) is not consistent within the dataset")
    population_size = df['N'].unique()[0]

    if len(df['L'].unique()) != 1:
        raise ValueError("sequence length (L) is not consistent within the dataset")
    sequence_length = df['L'].unique()[0]

    # Get unique sample sizes from both datasets
    sample_sizes = sorted(df['num_samples'].unique())

    # Define a colormap for sample sizes
    #cmap = plt.cm.get_cmap("viridis", len(sample_sizes))
    #cmap = matplotlib.cm.get_cmap("viridis", len(sample_sizes))
    hudson_data = df[df['k'] == 'Hudson']

    # Hybrid: k starts with 'Hybrid' or 'hybrid' (case-insensitive, if needed)
    hybrid_data = df[df['k'].str.lower().str.startswith('hybrid')]

    #smc_data = df[df['k'].str.match(r'^\d+$')]
    #smc_data['k'] = smc_data['k'].astype(int)
    smc_data = df[df['k']=='0']

    hybrid_ks = hybrid_data['k'].unique()
    group_names = ['Hudson', 'SMC'] + list(hybrid_ks)

    sample_sizes = sorted(
        set(hudson_data['num_samples']) |
        set(smc_data['num_samples']) |
        set(hybrid_data['num_samples'])
    )
    bar_width = 0.8 / len(group_names)  # To fit all bars neatly
    x = np.arange(len(sample_sizes))
    #cmap = plt.get_cmap('tab10')
    cmap = matplotlib.colormaps.get_cmap('tab10')

    # Prepare means: rows=groups, cols=sample_sizes
    means = np.zeros((len(group_names), len(sample_sizes)))

    for gi, group in enumerate(group_names):
        for si, sample_size in enumerate(sample_sizes):
            if group == 'Hudson':
                val = hudson_data[hudson_data['num_samples'] == sample_size]['time'].mean()
            elif group == 'SMC':
                val = smc_data[smc_data['num_samples'] == sample_size]['time'].mean()
            else:  # hybrid group
                val = hybrid_data[
          (hybrid_data['k'] == group) &
                    (hybrid_data['num_samples'] == sample_size)
                ]['time'].mean()
            means[gi, si] = val

    if givenax is None: fig, ax = plt.subplots(figsize=(12, 6))
    else: ax = givenax

    for gi, group in enumerate(group_names):
        ax.bar(
            x + gi*bar_width,
            means[gi],
            width=bar_width,
            color=cmap(gi % 10),
            label=group
        )
    ax.set_xticks(x + bar_width, sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Time')
    ax.set_title('Mean Coalescence Time by Sample Size and simulation method')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if givenax is None:

        ax.legend()
        plt.tight_layout()
        save(outfile)
        plt.close()

@click.command()
def hybrid():
    hybrid_inner()


@click.command()
def panels(infile="data/ancestry-perf-panels.csv",
        outfile="panels"):
    df = pd.read_csv(infile)
    Ns = sorted(df['N'].unique())
    ls = sorted(df['L'].unique())
    assert len(df['r'].unique())==1
    r = df['r'].unique()[0]
    fig, axes = plt.subplots(len(Ns), len(ls), figsize=(15, 10))
    for i, N in enumerate(Ns):
        for j, l in enumerate(ls):
            subset = df[(df['N'] == N) & (df['L'] == l)]
            if subset.empty: continue
            ax = axes[i, j]
            hybrid_inner(df=subset, givenax=ax)
            ax.set_title(f"N={N}, L={l}")
    fig.suptitle(f"simulations with r={r}", fontsize=16)
    plt.tight_layout()
    save(outfile)
    plt.close()


@click.command()
def ancestry_perf_hudson():
    """
    Run the ancestry benchmark.
    """
    ancestry_perf_plot(infile="data/ancestry-perf.csv", outfile="ancestry-perf")

@click.command()
def ancestry_perf_smc():
    """
    Run the ancestry benchmark.
    """
    ancestry_perf_plot(infile="data/ancestry-perf-smc.csv", outfile="ancestry-perf-smc")

@click.command()
def sequence_length_vs_time():
    """
    Run the sequence length vs time benchmark.
    """
    sequence_length_vs_time_combined(
        infile1="data/ancestry-perf-varying-seq-len.csv",
        infile2="data/ancestry-perf-varying-seq-len-smc.csv",
        name1="Hudson",
        name2="SMCK",
        outfile="sequence-length-vs-time",
    )


@click.command()
def sample_size_vs_time():
    """
    Run the sample size vs time benchmark.
    """
    sample_size_vs_time_combined(
        infile1="data/ancestry-perf-varying-sample-size.csv",
        infile2="data/ancestry-perf-varying-sample-size-smc.csv",
        name1="Hudson",
        name2="SMCK",
        outfile="sample-size-vs-time",
    )

with matplotlib.rc_context({"font.size": 7}):
    cli.add_command(ancestry_perf_hudson)
    cli.add_command(ancestry_perf_smc)
    cli.add_command(sequence_length_vs_time)
    cli.add_command(sample_size_vs_time)
    cli.add_command(k_vs_time)
    cli.add_command(hybrid)
    cli.add_command(panels)

if __name__ == "__main__":
    cli()

