from textwrap import dedent


def generate_phase5_summary() -> str:
    """
    Returns a markdown-formatted summary of Phase V
    suitable for inclusion in README.md or Zenodo.
    """
    text = r"""
    ## Phase V: Dynamical regime characterization

    Phase V builds on the Phase IV equilibrium pipeline by examining the
    detailed dynamics of the plasticity field $\langle w \rangle$ across
    the $(\eta,\lambda)$ plane.

    - For selected $(\eta,\lambda)$ points, we extract full episode
      trajectories of $\langle w \rangle$.
    - We classify the dynamics into:
        * equilibrium (stationary tail, small relative variance)
        * drift (persistent trend over episodes)
        * oscillatory (large variance around a mean)
    - We identify three key regimes:
        * a base equilibrium band at low $\lambda$
        * a non-equilibrium gap at intermediate $\lambda$
        * a re-entrant equilibrium island at higher $\lambda$ for
          specific $\eta$ values
    - These regimes are mapped back onto:
        * the binary phase diagram (equilibrated vs non-equilibrated)
        * the $T^*(\eta,\lambda)$ heatmap

    The resulting picture is a coherent dynamical interpretation of the
    Phase IV phase structure: the base band, instability gap, stability
    fingers, and floating equilibrium island all correspond to distinct
    qualitative behaviors of $\langle w \rangle$ over learning episodes.
    """
    return dedent(text).strip()
