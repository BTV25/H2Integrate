"""Fully continuous/differentiable electrolyzer performance model.

Separate from `ECOElectrolyzerPerformanceModel` (pem_electrolyzer.py) so the discrete,
production model is untouched. This model never rounds/floors n_clusters or dispatch, so it
has no derivative discontinuities in n_clusters, cluster_rating_MW, turndown_ratio, or
electricity_in -- at the cost of dropping degradation tracking (BOL only, matches
`_make_jax_h2_per_stack_fn`'s V_deg=0 assumption).

Combines three independently-picked smoothing candidates from the 2026-08-25 plot-first review
(see research-notes/hybridfarm/2026-08-25.md):
  - dispatch floor(): replaced by `n_active = clip(P / cluster_min_power, 0, n_clusters)`,
    exactly linear (zero curvature) in the sub-saturation range.
  - power-saturation min() clip: replaced by a LogSumExp smooth-min (`smooth_min2`), sharpness
    scaled to the per-stack kW range per [[feedback_logsumexp_sharpness_units]].
  - n_active's own upper clip at n_clusters (found while testing the above): also smoothed,
    same construction, sharpness scaled to the unitless cluster-count range.
n_clusters is never rounded, so the sizing `round()` jump is gone by construction.

turndown_ratio is exposed as a design variable, but ONLY its effect on the dispatch threshold
(cluster_min_power = turndown_ratio * cluster_rating_MW) is differentiated. Its real effect on
the electrochemical IV-curve fit (`PEM_H2_Clusters.iv_curve()`'s `scipy.optimize.curve_fit`, not
JAX) is NOT captured -- the curve is fit once at setup() using the tech_config value and held
fixed. This is a deliberate, known simplification (2026-08-25 decision: fast + fully analytic,
over refitting the curve per-iteration with an FD partial), matching the same tradeoff flagged
in the exploratory script's `build_stack_fn` docstring.

Degradation outputs (`uptime_degradation_v`, `cycling_degradation_v`, `fatigue_degradation_v`,
`total_degradation_v`), added 2026-08-25 session 8 (uptime/cycling) and session 9 (fatigue):
smooth proxies for all three turndown-sensitive mechanisms in the real discrete model
(`calc_uptime_degradation`, `calc_onoff_degradation`, `approx_fatigue_degradation`), validated
against the real values in `scripts/explore_electrolyzer_degradation_proxy.py` and
`scripts/explore_electrolyzer_fatigue_proxy.py`. Uptime/cycling share the same threshold as
dispatch (`cluster_min_power = turndown_ratio * cluster_rating_MW`), but as a sigmoid
`soft_on(t)` instead of a hard step; voltage is the real load-dependent operating voltage
(`_make_jax_voltage_fn`, ported from `cell_design`/`calc_V_act`/`calc_V_ohmic`), not a constant.
`cycling_deg` sums only the *decreases* in `soft_on` (ReLU), matching the real code's
off-transition-only cycle count -- this ReLU has the same kind of derivative kink as the
existing `jnp.maximum` uses elsewhere in this file (e.g. `n_active_raw`), not newly introduced.

Fatigue is a genuinely different mechanism (rainflow peak-valley hysteresis-loop counting on the
degraded voltage signal) with no closed-form smooth equivalent, so the proxy is a different
construction: `rate_fatigue * _FATIGUE_K_FIT * total_variation(voltage(t) * soft_on(t))`. Per
[[feedback_curvature_over_tight_fit]], this matches the real curve's overall increasing trend
(correlation 0.865 across a turndown sweep) but not its discrete plateaus/local jumps --
accepted 2026-08-25 as good enough (median rel. error 24%, worst case 158% at a plateau edge).
`_FATIGUE_K_FIT` is a single constant fit by least squares in the exploration script, not
re-fit per turndown_ratio.

NOT yet wired into any cost/LCOH output -- these are raw degradation quantities (Volts) only;
how they should feed into replacement cost / efficiency loss is still open.
"""

import numpy as np
import jax
import jax.numpy as jnp
from attrs import define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.converters.hydrogen.electrolyzer_baseclass import ElectrolyzerPerformanceBaseClass
from h2integrate.converters.hydrogen.pem_electrolyzer import (
    ECOElectrolyzerPerformanceModelConfig,
    _make_jax_h2_per_stack_fn,
)
from h2integrate.converters.hydrogen.pem_model.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as _PEMClusters,
)

jax.config.update("jax_enable_x64", True)

_STACK_RATING_KW = 1000.0
_HHV_KWH_PER_KG = 39.4
# Half-width of the saturation-clip smoothing, relative to a single stack's 1 MW rating.
_SAT_CLIP_S_KW = 1.0 / 1000.0
# Sharpness for the n_active -> n_clusters saturation (unitless, per cluster-count scale):
# transition width ~1/s = 0.2 clusters. Without this, n_active's hard upper clip leaves a real
# derivative kink exactly at full dispatch (n_active == n_clusters), where the model switches
# from "spread evenly across a growing active count" to "n_clusters stacks fixed, power/stack
# rising" -- these two regimes have different slopes in general, so the transition itself (not
# just the two saturation clips already handled) needs smoothing too.
_N_ACTIVE_CLIP_S = 5.0

# Scalar (per-timestep-independent) design variables, in argument order after power_kw.
_SCALAR_ARGS = ("n_clusters", "cluster_rating_mw", "turndown_ratio")


def smooth_max2(a, b, s):
    """Two-argument LogSumExp smooth max (same construction as Ard's smooth_max)."""
    m = jnp.maximum(a, b)
    return jnp.log(jnp.exp(s * (a - m)) + jnp.exp(s * (b - m))) / s + m


def smooth_min2(a, b, s):
    return -smooth_max2(-a, -b, s)


# Half-width of the soft on/off dispatch indicator used for degradation, as a fraction of the
# cluster's own power scale (turndown_ratio * cluster_rating_MW) -- same construction as
# _SAT_CLIP_S_KW/_N_ACTIVE_CLIP_S, validated in explore_electrolyzer_degradation_proxy.py.
_SOFT_ON_WIDTH_FRAC = 0.02

# Total-variation-to-rainflow scale constant for the fatigue proxy, least-squares fit in
# explore_electrolyzer_fatigue_proxy.py across a turndown_ratio sweep on the baseline wind
# profile (correlation 0.865 vs the real rainflow-counted values; not re-fit per input).
_FATIGUE_K_FIT = 0.928


def _make_jax_voltage_fn(curve_coeff):
    """JAX-differentiable cell voltage (V) for one 1 MW stack, given its power (kW).

    Ports PEM_H2_Clusters.cell_design (= calc_reversible_cell_voltage + calc_V_act +
    calc_V_ohmic) to JAX, at the model's fixed T_C=80 operating temperature. Current is
    computed the same way as `_make_jax_h2_per_stack_fn` (same curve_coeff).
    """
    p = jnp.array(curve_coeff[:5], dtype=jnp.float64)
    F = 96485.34
    R = 8.314
    cell_active_area = 1949.0
    T_C = 80.0
    T_K = T_C + 273.15
    membrane_thickness_cm = 0.0077

    A, B, C = 8.07131, 1730.63, 233.426
    p_h2o_sat_mmHg = 10 ** (A - (B / (C + T_C)))
    p_h2o_sat_atm = p_h2o_sat_mmHg * (133.322 / 101325.0)
    E_cell = 1.229 + ((R * T_K) / (2 * F)) * jnp.log(
        (1 - p_h2o_sat_atm) * jnp.sqrt(1 - p_h2o_sat_atm)
    )
    lambda_water = ((-2.89556 + 0.016 * T_K) + 1.625) / 0.1875
    sigma = (0.005139 * lambda_water - 0.00326) * jnp.exp(1268 * ((1 / 303) - (1 / T_K)))
    R_cell = membrane_thickness_cm / sigma

    def voltage_fn(power_per_stack_kw):
        pwr = power_per_stack_kw
        I = (
            p[0] * pwr**3 + p[1] * pwr**2 + p[2] * pwr
            + p[3] * jnp.sqrt(jnp.maximum(pwr, 0.0)) + p[4]
        )
        I = jnp.maximum(I, 0.0)
        i = I / cell_active_area
        V_anode = ((R * T_K) / (2 * F)) * jnp.arcsinh(i / (2 * 4e-7))
        V_cathode = ((R * T_K) / (0.5 * F)) * jnp.arcsinh(i / (2 * 4e-3))
        V_ohmic = i * R_cell
        return E_cell + V_anode + V_cathode + V_ohmic

    return voltage_fn


def _degradation_totals(
    power_kw, n_clusters, cluster_rating_mw, turndown_ratio,
    voltage_fn, steady_deg_rate, onoff_deg_rate, rate_fatigue,
):
    """Smooth proxies (V) for real-model uptime, on/off-cycling, and fatigue degradation.

    soft_on(t) in [0, 1] replaces the real model's hard cluster_status step at the same
    threshold (cluster_min_power = turndown_ratio * cluster_rating_mw). Voltage is evaluated at
    the same per-stack power split the real model uses when fully on (power_per_cluster /
    cluster_rating_mw -- 1 MW per stack), weighted by soft_on rather than gated to zero, so it
    stays smooth in electricity_in/turndown_ratio/cluster_rating_mw/n_clusters.

    fatigue_deg_v replaces the real model's rainflow cycle-counting with total variation of the
    same soft-gated voltage signal (see module docstring / _FATIGUE_K_FIT).
    """
    p_min_kw = turndown_ratio * 1000.0 * cluster_rating_mw
    width_kw = _SOFT_ON_WIDTH_FRAC * 1000.0 * cluster_rating_mw
    power_per_cluster = power_kw / n_clusters
    soft_on = jax.nn.sigmoid((power_per_cluster - p_min_kw) / width_kw)

    power_per_stack = power_per_cluster / cluster_rating_mw
    voltage = voltage_fn(power_per_stack)

    dt_sec = 3600.0
    uptime_deg_v = steady_deg_rate * dt_sec * jnp.sum(voltage * soft_on)
    decreases = jnp.maximum(soft_on[:-1] - soft_on[1:], 0.0)
    cycling_deg_v = onoff_deg_rate * jnp.sum(decreases)

    v_gated = voltage * soft_on
    tv = jnp.sum(jnp.abs(v_gated[1:] - v_gated[:-1]))
    fatigue_deg_v = rate_fatigue * _FATIGUE_K_FIT * tv

    return uptime_deg_v, cycling_deg_v, fatigue_deg_v


def _h2_per_timestep(power_kw, n_clusters, cluster_rating_mw, turndown_ratio, stack_fn):
    """Continuous H2 production (kg/hr) for one timestep of total input power (kW).

    At power_kw == 0 exactly (a routine occurrence for a real wind profile), naively dividing
    by n_stacks_active and evaluating stack_fn (whose sqrt() term has an infinite derivative at
    0) both produce a 0/0-style indeterminate gradient -- JAX's autodiff returns NaN there even
    though the true derivative is 0 (no power in, no H2 out, regardless of sizing). Both
    branches of every jnp.where below are kept finite (not just the selected one) so no NaN
    ever enters the graph -- `jnp.where` still backprops through the branch it didn't select,
    so guarding only the selected branch doesn't work (0 * NaN = NaN).
    """
    p_min_kw = turndown_ratio * 1000.0 * cluster_rating_mw
    n_active_raw = jnp.maximum(power_kw / p_min_kw, 0.0)
    n_active = smooth_min2(n_active_raw, n_clusters, _N_ACTIVE_CLIP_S)
    n_stacks_active = n_active * cluster_rating_mw
    is_on = n_stacks_active > 1e-6
    safe_n_stacks = jnp.where(is_on, n_stacks_active, 1.0)
    capacity_kw = safe_n_stacks * _STACK_RATING_KW
    power_capped = smooth_min2(power_kw, capacity_kw, _SAT_CLIP_S_KW)
    power_per_stack = jnp.maximum(power_capped / safe_n_stacks, 1e-6)
    h2 = safe_n_stacks * stack_fn(power_per_stack)
    return jnp.where(is_on, h2, 0.0)


@define(kw_only=True)
class SmoothElectrolyzerPerformanceModelConfig(ECOElectrolyzerPerformanceModelConfig):
    """Same fields as ECOElectrolyzerPerformanceModelConfig. H2 production is BOL-only by
    construction (V_deg=0); degradation is tracked separately as raw diagnostic outputs
    (uptime_degradation_v/cycling_degradation_v/total_degradation_v) that do NOT feed back into
    hydrogen_out/efficiency/replacement_schedule."""


class SmoothElectrolyzerPerformanceModel(ElectrolyzerPerformanceBaseClass):
    """Fully continuous electrolyzer performance model for gradient-based optimization.

    Drop-in alternative to ECOElectrolyzerPerformanceModel for testing/optimization use where
    smoothness in n_clusters/cluster_rating_MW/turndown_ratio/electricity_in matters more than
    degradation fidelity. No on/off dispatch simulation, no rounding for H2 production -- a
    closed-form JAX expression, differentiated with jax.grad/vmap. Degradation
    (uptime_degradation_v/cycling_degradation_v/total_degradation_v) is a separate smooth proxy,
    also JAX-differentiated, that reports V but does not affect H2 production (see module
    docstring).
    """

    def setup(self):
        self.config = SmoothElectrolyzerPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_output("efficiency", val=0.0, units="unitless", desc="HHV efficiency (BOL)")
        self.add_output(
            "time_until_replacement", val=0.0, units="h", desc="Time until replacement"
        )
        self.add_input(
            "n_clusters",
            val=self.config.n_clusters,
            units="unitless",
            desc="number of electrolyzer clusters in the system (continuous)",
        )
        self.add_input(
            "cluster_rating_MW",
            val=self.config.cluster_rating_MW,
            units="MW",
            desc="rating of a single cluster (continuous)",
        )
        self.add_input(
            "turndown_ratio",
            val=self.config.turndown_ratio,
            units="unitless",
            desc="dispatch turndown floor, as a fraction of cluster_rating_MW (continuous; "
            "does NOT refit the electrochemical IV curve -- see module docstring)",
        )
        self.add_output("electrolyzer_size_mw", val=0.0, units="MW")
        self.add_output(
            "uptime_degradation_v", val=0.0, units="V", desc="Smooth uptime degradation proxy"
        )
        self.add_output(
            "cycling_degradation_v", val=0.0, units="V", desc="Smooth on/off cycling degradation proxy"
        )
        self.add_output(
            "fatigue_degradation_v", val=0.0, units="V", desc="Smooth rainflow fatigue degradation proxy"
        )
        self.add_output(
            "total_degradation_v", val=0.0, units="V",
            desc="uptime + cycling + fatigue degradation proxy",
        )

        _pem_tmp = _PEMClusters(
            1,
            1,
            eol_eff_percent_loss=self.config.eol_eff_percent_loss,
            uptime_hours_until_eol=self.config.uptime_hours_until_eol,
            include_degradation_penalty=False,
            turndown_ratio=self.config.turndown_ratio,
        )
        self._stack_fn = _make_jax_h2_per_stack_fn(_pem_tmp.curve_coeff)
        self._rated_stack_h2 = float(self._stack_fn(jnp.asarray(_STACK_RATING_KW)))

        def _f(power_kw, n_clusters, cluster_rating_mw, turndown_ratio):
            return _h2_per_timestep(
                power_kw, n_clusters, cluster_rating_mw, turndown_ratio, self._stack_fn
            )

        self._h2_fn = _f
        self._h2_fn_v = jax.vmap(_f, in_axes=(0, None, None, None))
        self._d_power = jax.vmap(jax.grad(_f, argnums=0), in_axes=(0, None, None, None))
        self._d_scalar = {
            name: jax.vmap(jax.grad(_f, argnums=i + 1), in_axes=(0, None, None, None))
            for i, name in enumerate(_SCALAR_ARGS)
        }

        self._voltage_fn = _make_jax_voltage_fn(_pem_tmp.curve_coeff)
        self._steady_deg_rate = _pem_tmp.steady_deg_rate
        self._onoff_deg_rate = _pem_tmp.onoff_deg_rate
        self._rate_fatigue = _pem_tmp.rate_fatigue

        def _f_uptime(power_kw, n_clusters, cluster_rating_mw, turndown_ratio):
            uptime_v, _, _ = _degradation_totals(
                power_kw, n_clusters, cluster_rating_mw, turndown_ratio,
                self._voltage_fn, self._steady_deg_rate, self._onoff_deg_rate, self._rate_fatigue,
            )
            return uptime_v

        def _f_cycling(power_kw, n_clusters, cluster_rating_mw, turndown_ratio):
            _, cycling_v, _ = _degradation_totals(
                power_kw, n_clusters, cluster_rating_mw, turndown_ratio,
                self._voltage_fn, self._steady_deg_rate, self._onoff_deg_rate, self._rate_fatigue,
            )
            return cycling_v

        def _f_fatigue(power_kw, n_clusters, cluster_rating_mw, turndown_ratio):
            _, _, fatigue_v = _degradation_totals(
                power_kw, n_clusters, cluster_rating_mw, turndown_ratio,
                self._voltage_fn, self._steady_deg_rate, self._onoff_deg_rate, self._rate_fatigue,
            )
            return fatigue_v

        # Scalar in, scalar out (already reduced over time) -- no vmap needed, jax.grad on the
        # vector arg (power_kw) directly returns the per-timestep gradient vector.
        self._deg_fns = {"uptime": _f_uptime, "cycling": _f_cycling, "fatigue": _f_fatigue}
        self._deg_grad_power = {k: jax.grad(f, argnums=0) for k, f in self._deg_fns.items()}
        self._deg_grad_scalar = {
            k: {name: jax.grad(f, argnums=i + 1) for i, name in enumerate(_SCALAR_ARGS)}
            for k, f in self._deg_fns.items()
        }

    def setup_partials(self):
        n_ts = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        plant_life = self.options["plant_config"]["plant"]["plant_life"]
        arange = np.arange(n_ts)
        zeros = np.zeros(n_ts, int)

        self.declare_partials("hydrogen_out", "electricity_in", rows=arange, cols=arange)
        self.declare_partials("total_hydrogen_produced", "electricity_in", rows=zeros, cols=arange)
        self.declare_partials("efficiency", "electricity_in", rows=zeros, cols=arange)
        ann_rows = np.repeat(np.arange(plant_life), n_ts)
        ann_cols = np.tile(arange, plant_life)
        self.declare_partials("annual_hydrogen_produced", "electricity_in", rows=ann_rows, cols=ann_cols)
        self.declare_partials("capacity_factor", "electricity_in", rows=ann_rows, cols=ann_cols)
        deg_outputs = [
            "uptime_degradation_v", "cycling_degradation_v", "fatigue_degradation_v",
            "total_degradation_v",
        ]
        for deg_name in deg_outputs:
            self.declare_partials(deg_name, "electricity_in", rows=zeros, cols=arange)

        scalar_inputs = ["n_clusters", "cluster_rating_MW", "turndown_ratio"]
        for name in scalar_inputs:
            self.declare_partials("hydrogen_out", name, rows=arange, cols=zeros)
            self.declare_partials("total_hydrogen_produced", name)
            self.declare_partials("annual_hydrogen_produced", name)
            self.declare_partials("capacity_factor", name)
            self.declare_partials("rated_hydrogen_production", name)
            self.declare_partials("electrolyzer_size_mw", name)
            self.declare_partials("efficiency", name)
            for deg_name in deg_outputs:
                self.declare_partials(deg_name, name)

        self._n_ts = n_ts
        self._plant_life = plant_life

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        n_clusters = float(inputs["n_clusters"][0])
        cluster_rating_mw = float(inputs["cluster_rating_MW"][0])
        turndown_ratio = float(inputs["turndown_ratio"][0])
        power_kw = jnp.asarray(inputs["electricity_in"], dtype=jnp.float64)

        hydrogen_out = np.array(self._h2_fn_v(power_kw, n_clusters, cluster_rating_mw, turndown_ratio))
        outputs["hydrogen_out"] = hydrogen_out
        total_h2 = float(hydrogen_out.sum())
        outputs["total_hydrogen_produced"] = total_h2

        rated_h2 = n_clusters * cluster_rating_mw * self._rated_stack_h2
        outputs["rated_hydrogen_production"] = rated_h2

        annual_h2 = total_h2 / self.fraction_of_year_simulated
        outputs["annual_hydrogen_produced"] = np.full(self._plant_life, annual_h2)
        outputs["replacement_schedule"] = np.zeros(self._plant_life)
        cf = annual_h2 / (rated_h2 * 8760.0) if rated_h2 > 0 else 0.0
        outputs["capacity_factor"] = np.full(self._plant_life, cf)

        outputs["electrolyzer_size_mw"] = n_clusters * cluster_rating_mw
        outputs["time_until_replacement"] = self.config.uptime_hours_until_eol

        uptime_v, cycling_v, fatigue_v = _degradation_totals(
            power_kw, n_clusters, cluster_rating_mw, turndown_ratio,
            self._voltage_fn, self._steady_deg_rate, self._onoff_deg_rate, self._rate_fatigue,
        )
        outputs["uptime_degradation_v"] = float(uptime_v)
        outputs["cycling_degradation_v"] = float(cycling_v)
        outputs["fatigue_degradation_v"] = float(fatigue_v)
        outputs["total_degradation_v"] = float(uptime_v + cycling_v + fatigue_v)

        total_elec_kwh = float(np.sum(inputs["electricity_in"])) * self.dt / 3600.0
        outputs["efficiency"] = (
            total_h2 * _HHV_KWH_PER_KG / total_elec_kwh if total_elec_kwh > 0 else 0.0
        )

        self._total_h2 = total_h2
        self._rated_h2 = rated_h2
        self._annual_h2 = annual_h2
        self._n_clusters = n_clusters
        self._cluster_rating_mw = cluster_rating_mw

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n_clusters = float(inputs["n_clusters"][0])
        cluster_rating_mw = float(inputs["cluster_rating_MW"][0])
        turndown_ratio = float(inputs["turndown_ratio"][0])
        power_kw = jnp.asarray(inputs["electricity_in"], dtype=jnp.float64)
        args = (power_kw, n_clusters, cluster_rating_mw, turndown_ratio)

        jac_power = np.array(self._d_power(*args))
        partials["hydrogen_out", "electricity_in"] = jac_power
        partials["total_hydrogen_produced", "electricity_in"] = jac_power
        d_annual_d_power = jac_power / self.fraction_of_year_simulated
        partials["annual_hydrogen_produced", "electricity_in"] = np.tile(
            d_annual_d_power, self._plant_life
        )

        rated_h2 = n_clusters * cluster_rating_mw * self._rated_stack_h2
        annual_h2 = getattr(self, "_annual_h2", 0.0)
        total_h2 = getattr(self, "_total_h2", 0.0)
        total_elec_kwh = float(np.sum(inputs["electricity_in"])) * self.dt / 3600.0

        if rated_h2 > 0:
            d_cf_d_power = d_annual_d_power / (rated_h2 * 8760.0)
        else:
            d_cf_d_power = np.zeros_like(d_annual_d_power)
        partials["capacity_factor", "electricity_in"] = np.tile(d_cf_d_power, self._plant_life)

        if total_elec_kwh > 0:
            d_elec_kwh_d_power = self.dt / 3600.0
            partials["efficiency", "electricity_in"] = (
                _HHV_KWH_PER_KG / total_elec_kwh * jac_power
                - total_h2 * _HHV_KWH_PER_KG / total_elec_kwh**2 * d_elec_kwh_d_power
            )
        else:
            partials["efficiency", "electricity_in"] = np.zeros(self._n_ts)

        # d(rated_h2)/d(scalar): rated_h2 = n_clusters * cluster_rating_mw * rated_stack_h2,
        # independent of turndown_ratio (rated production assumes full utilization).
        d_rated = {
            "n_clusters": cluster_rating_mw * self._rated_stack_h2,
            "cluster_rating_mw": n_clusters * self._rated_stack_h2,
            "turndown_ratio": 0.0,
        }
        # d(electrolyzer_size_mw)/d(scalar): size = n_clusters * cluster_rating_mw.
        d_size = {
            "n_clusters": cluster_rating_mw,
            "cluster_rating_mw": n_clusters,
            "turndown_ratio": 0.0,
        }
        om_names = {"n_clusters": "n_clusters", "cluster_rating_mw": "cluster_rating_MW", "turndown_ratio": "turndown_ratio"}

        for key, om_name in om_names.items():
            jac_scalar = np.array(self._d_scalar[key](*args))
            sum_jac = float(jac_scalar.sum())

            partials["hydrogen_out", om_name] = jac_scalar
            partials["total_hydrogen_produced", om_name] = sum_jac

            d_annual_d_s = sum_jac / self.fraction_of_year_simulated
            partials["annual_hydrogen_produced", om_name] = d_annual_d_s

            if rated_h2 > 0:
                d_cf_d_s = (d_annual_d_s * rated_h2 - annual_h2 * d_rated[key]) / (rated_h2**2 * 8760.0)
            else:
                d_cf_d_s = 0.0
            partials["capacity_factor", om_name] = d_cf_d_s

            partials["rated_hydrogen_production", om_name] = d_rated[key]
            partials["electrolyzer_size_mw", om_name] = d_size[key]

            if total_elec_kwh > 0:
                partials["efficiency", om_name] = _HHV_KWH_PER_KG / total_elec_kwh * sum_jac
            else:
                partials["efficiency", om_name] = 0.0

            d_uptime_s = float(self._deg_grad_scalar["uptime"][key](*args))
            d_cycling_s = float(self._deg_grad_scalar["cycling"][key](*args))
            d_fatigue_s = float(self._deg_grad_scalar["fatigue"][key](*args))
            partials["uptime_degradation_v", om_name] = d_uptime_s
            partials["cycling_degradation_v", om_name] = d_cycling_s
            partials["fatigue_degradation_v", om_name] = d_fatigue_s
            partials["total_degradation_v", om_name] = d_uptime_s + d_cycling_s + d_fatigue_s

        d_uptime_power = np.array(self._deg_grad_power["uptime"](*args))
        d_cycling_power = np.array(self._deg_grad_power["cycling"](*args))
        d_fatigue_power = np.array(self._deg_grad_power["fatigue"](*args))
        partials["uptime_degradation_v", "electricity_in"] = d_uptime_power
        partials["cycling_degradation_v", "electricity_in"] = d_cycling_power
        partials["fatigue_degradation_v", "electricity_in"] = d_fatigue_power
        partials["total_degradation_v", "electricity_in"] = (
            d_uptime_power + d_cycling_power + d_fatigue_power
        )
