"""Fully continuous sibling of SingliticoCostModel (singlitico_cost_model.py), separate so the
original is untouched. Only change: the hardcoded 100 MW economies-of-scale cap (a real
derivative kink, `jnp.where(P >= 0.1, 0.1, P)`) is replaced by a LogSumExp smooth-min, same
construction/sharpness convention as pem_electrolyzer_smooth.py. The separate 10 MW SF_elec
scale-factor switch is left as-is -- it sits at a fixed, small power level unlikely to be
straddled by a design search whose baseline is 100 MW, and was out of scope for the
2026-08-25 plot-first review that picked this candidate.
"""

import numpy as np
import jax
import jax.numpy as jnp

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.converters.hydrogen.electrolyzer_baseclass import ElectrolyzerCostBaseClass
from h2integrate.converters.hydrogen.singlitico_cost_model import SingliticoCostModelConfig

jax.config.update("jax_enable_x64", True)

# Half-width of the 100 MW cap smoothing (GW): ~5 MW, matching the 2026-08-25 plot-first review.
SINGLITICO_CAP_S_GW = 1.0 / 0.005


def smooth_max2(a, b, s):
    m = jnp.maximum(a, b)
    return jnp.log(jnp.exp(s * (a - m)) + jnp.exp(s * (b - m))) / s + m


def smooth_min2(a, b, s):
    return -smooth_max2(-a, -b, s)


def _make_jax_singlitico_compute_smooth(location, electrolyzer_capex, s=SINGLITICO_CAP_S_GW):
    loc = 0.0 if location == "onshore" else 1.0

    def compute(x):
        P = x[0] * 1e-3  # GW
        SF = jnp.where(P < 10.0 / 1e3, -0.21, -0.14)
        P_cost = smooth_min2(P, jnp.asarray(0.1), s)

        unit_capex_musd = electrolyzer_capex * (1.0 + 0.33 * loc) * (P_cost * 1e3 / 10.0) ** SF
        capital_musd = unit_capex_musd * P
        CapEx = capital_musd * 1e6

        P_opex = smooth_min2(P, jnp.asarray(0.1), s)
        opex_eq = capital_musd * (1.0 - 0.33 * (1.0 + loc)) * 0.0344 * (P_opex * 1e3) ** -0.155
        opex_neq = 0.04 * capital_musd * 0.33 * (1.0 + loc)
        OpEx = (opex_eq + opex_neq) * 1e6

        return jnp.stack([CapEx, OpEx])

    return compute


class SmoothSingliticoCostModel(ElectrolyzerCostBaseClass):
    """Continuous alternative to SingliticoCostModel, for use alongside
    SmoothElectrolyzerPerformanceModel when electrolyzer_size_mw is a gradient-optimized DV
    that may cross the real model's 100 MW cap.
    """

    def setup(self):
        self.config = SingliticoCostModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()
        self.add_input("electrolyzer_size_mw", val=0, units="MW")

    def setup_partials(self):
        self.declare_partials("CapEx", "electrolyzer_size_mw")
        self.declare_partials("OpEx", "electrolyzer_size_mw")
        _fn = _make_jax_singlitico_compute_smooth(
            self.config.location, float(self.config.electrolyzer_capex)
        )
        self._jax_fn = jax.jit(_fn)
        self._jax_jac = jax.jit(jax.jacobian(_fn))

    def compute(self, inputs, outputs):
        x = jnp.array([float(inputs["electrolyzer_size_mw"][0])])
        capex, opex = np.array(self._jax_fn(x))
        outputs["CapEx"] = capex
        outputs["OpEx"] = opex

    def compute_partials(self, inputs, partials):
        x = jnp.array([float(inputs["electrolyzer_size_mw"][0])])
        jac = self._jax_jac(x)
        partials["CapEx", "electrolyzer_size_mw"] = np.array(jac[0])
        partials["OpEx", "electrolyzer_size_mw"] = np.array(jac[1])
