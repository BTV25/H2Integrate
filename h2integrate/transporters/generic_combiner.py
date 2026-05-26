import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs


@define(kw_only=True)
class GenericCombinerPerformanceConfig(BaseConfig):
    """Configuration class for a generic combiner.

    Fields include `commodity`, `commodity_rate_units`, and `in_streams`.
    """

    commodity: str = field(converter=(str.lower, str.strip))
    commodity_rate_units: str = field()
    in_streams: int = field(default=2)


class GenericCombinerPerformanceModel(om.ExplicitComponent):
    """
    Combine any commodity or resource from multiple sources into one output without losses.

    This component is purposefully simple; a more realistic case might include
    losses or other considerations from system components.

    The combined output capacity factor is computed as a weighted average of the
    input stream capacity factors, weighted by each stream's rated production:

    .. math::

        CF_{out} = \\frac{\\sum_i CF_i \\cdot S_i}{\\sum_i S_i}

    where :math:`CF_i` is the capacity factor and :math:`S_i` is the rated
    commodity production of input stream *i*. If the total rated production is
    zero, the output capacity factor is set to zero.

    The total rated production is the sum of all input rated productions, and
    the output commodity profile is the element-wise sum of all input profiles.
    """

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = GenericCombinerPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        plant_life = int(self.options["plant_config"]["plant"]["plant_life"])
        self._n_timesteps = n_timesteps
        self._plant_life = plant_life

        for i in range(1, self.config.in_streams + 1):
            self.add_input(
                f"{self.config.commodity}_in{i}",
                val=0.0,
                shape=n_timesteps,
                units=self.config.commodity_rate_units,
            )
            self.add_input(
                f"rated_{self.config.commodity}_production{i}",
                val=0.0,
                units=self.config.commodity_rate_units,
            )
            self.add_input(
                f"{self.config.commodity}_capacity_factor{i}",
                val=0.0,
                shape=plant_life,
                units="unitless",
            )

        self.add_output(
            f"{self.config.commodity}_out",
            val=0.0,
            shape=n_timesteps,
            units=self.config.commodity_rate_units,
        )
        self.add_output(
            "capacity_factor",
            val=0.0,
            shape=plant_life,
            units="unitless",
        )
        self.add_output(
            f"rated_{self.config.commodity}_production",
            val=0.0,
            units=self.config.commodity_rate_units,
        )

    def setup_partials(self):
        commodity = self.config.commodity
        n = self._n_timesteps
        p = self._plant_life

        for i in range(1, self.config.in_streams + 1):
            # commodity_out[t] = sum_i commodity_in_i[t]: identity per stream
            self.declare_partials(
                f"{commodity}_out", f"{commodity}_in{i}",
                rows=np.arange(n), cols=np.arange(n), val=1.0,
            )
            # rated_production = sum_i rated_production_i: scalar 1.0 per stream
            self.declare_partials(
                f"rated_{commodity}_production",
                f"rated_{commodity}_production{i}",
                val=1.0,
            )
            # capacity_factor = sum(CF_i*S_i) / sum(S_i): nonlinear wrt CF_i and S_i
            self.declare_partials(
                "capacity_factor", f"{commodity}_capacity_factor{i}",
                rows=np.arange(p), cols=np.arange(p),
            )
            self.declare_partials(
                "capacity_factor", f"rated_{commodity}_production{i}",
            )

    def compute_partials(self, inputs, partials):
        commodity = self.config.commodity

        total_rated = sum(
            inputs[f"rated_{commodity}_production{i}"].item()
            for i in range(1, self.config.in_streams + 1)
        )

        if total_rated > 0:
            combined_production = sum(
                inputs[f"{commodity}_capacity_factor{i}"]
                * inputs[f"rated_{commodity}_production{i}"].item()
                for i in range(1, self.config.in_streams + 1)
            )
            capacity_factor = combined_production / total_rated

            for i in range(1, self.config.in_streams + 1):
                S_i = inputs[f"rated_{commodity}_production{i}"].item()
                CF_i = inputs[f"{commodity}_capacity_factor{i}"]
                # d(cf)/d(CF_i)[t] = S_i / total_rated  (diagonal)
                partials["capacity_factor", f"{commodity}_capacity_factor{i}"] = (
                    S_i / total_rated
                )
                # d(cf)/d(S_i)[t] = (CF_i[t] - cf[t]) / total_rated
                partials["capacity_factor", f"rated_{commodity}_production{i}"] = (
                    (CF_i - capacity_factor) / total_rated
                )
        else:
            for i in range(1, self.config.in_streams + 1):
                partials["capacity_factor", f"{commodity}_capacity_factor{i}"] = 0.0
                partials["capacity_factor", f"rated_{commodity}_production{i}"] = 0.0

    def compute(self, inputs, outputs):
        total_out = 0.0
        combined_production = 0.0
        total_rated = 0.0
        for key, value in inputs.items():
            if "_in" in key:
                # add the commodity_in profile
                total_out = total_out + value
            if key.startswith("rated_"):
                # add the rated_commodity_production
                total_rated = total_rated + value
            if "_capacity_factor" in key:
                # get the stream number so we can get the proper rated capacity
                stream_number = key.split("capacity_factor")[-1]
                rated_capacity = inputs[f"rated_{self.config.commodity}_production{stream_number}"]
                # weight the capacity factor with the rated capacity to get the combined production
                combined_production += value * rated_capacity

        outputs[f"{self.config.commodity}_out"] = total_out
        outputs[f"rated_{self.config.commodity}_production"] = total_rated
        if total_rated > 0:
            # weighted CF = (CF1*S1 + CF2*S2)/(S1 + S2) = combined production/combined capacity
            # Where S is the rated commodity production of input stream i
            # and CF is the capacity factor of input stream i
            weighted_cf = combined_production / total_rated
            outputs["capacity_factor"] = weighted_cf
        else:
            outputs["capacity_factor"] = 0.0
