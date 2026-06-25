import numpy as np
import openmdao.api as om


class CablePerformanceModel(om.ExplicitComponent):
    """
    Pass-through cable with no losses.
    """

    def initialize(self):
        self.options.declare("transport_item", values=["electricity"])
        self.options.declare("plant_config", default=None)

    def setup(self):
        self.input_name = self.options["transport_item"] + "_in"
        self.output_name = self.options["transport_item"] + "_out"
        self.add_input(
            self.input_name,
            val=-1.0,
            shape_by_conn=True,
            copy_shape=self.output_name,
            units="kW",
        )
        self.add_output(
            self.output_name,
            val=-1.0,
            shape_by_conn=True,
            copy_shape=self.input_name,
            units="kW",
        )

    def setup_partials(self):
        plant_config = self.options["plant_config"]
        if plant_config is not None:
            n = plant_config["plant"]["simulation"]["n_timesteps"]
            arange = np.arange(n)
            self.declare_partials(self.output_name, self.input_name,
                                  rows=arange, cols=arange, val=1.0)
        else:
            self.declare_partials(self.output_name, self.input_name, method="cs")

    def compute(self, inputs, outputs):
        outputs[self.output_name] = inputs[self.input_name]
