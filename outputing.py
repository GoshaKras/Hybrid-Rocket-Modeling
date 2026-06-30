class Output_template:
    def __init__(self,label,value,unit,precision):
        self.name= label
        self.value= value
        self.unit= unit
        self.precision= precision
    def __str__(self):
        if isinstance(self.value, float):
            return f"{self.name}: {self.value:.{self.precision}f} {self.unit}"
        else:
            return f"{self.name}: {self.value} {self.unit}"


class OutputGroup:
    def __init__(self, name):
        self.name = name
        self.outputs = []

    def add_output(self, label, value, unit, precision):
        self.outputs.append(Output_template(label, value, unit, precision))

    def print_outputs(self):
        print(f"--- {self.name} Outputs ---")
        for output in self.outputs:
            if isinstance(output.value, float):
                print(f"{output.name:<30} {output.value:.{output.precision}f} {output.unit}")
            else:
                print(f"{output.name:<30} {output.value} {output.unit}")

    def get_output_by_label(self, label):
        for output in self.outputs:
            if output.name == label:
                return output
        raise KeyError(f"Output label '{label}' not found in {self.name}")


class RocketOutputs:
    def __init__(self):
        self.cea = OutputGroup("CEA Properties")
        self.nozzle = OutputGroup("Nozzle")
        self.oxtank = OutputGroup("Oxtank")
        self.fuelgrain = OutputGroup("Fuelgrain")
        # You can add more groups here, e.g. self.oxtank, self.fuelgrain

    def print_all_outputs(self):
        self.cea.print_outputs()
        self.nozzle.print_outputs()
        self.oxtank.print_outputs()
        self.fuelgrain.print_outputs()

    def print_ox_tank_starting_outputs(self):
        self.nozzle.print_outputs()

    def print_out_fuelgrain_outputs(self):
        self.fuelgrain.print_outputs()

    def print_certain_cea_values(self):
        self.cea.print_outputs()

    def print_nozzle_outputs(self):
        self.nozzle.print_outputs()
