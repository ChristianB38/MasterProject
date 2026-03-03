from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import strict_discrete_set

class SR850(Instrument):
    """
    Driver for the Stanford Research SR850 Lock-in Amplifier.
    Supports GPIB or serial RS-232 through a VISA resource.
    """

    def __init__(self, adapter, name="SR850 Lock-in Amplifier", **kwargs):
        super().__init__(adapter, name, **kwargs)

        # Terminations:
        self.adapter.connection.write_termination = "\n"
        self.adapter.connection.read_termination = "\n"

    # -----------------------
    # Core instrument queries
    # -----------------------

    id = Instrument.measurement(
        "*IDN?",
        """The identity query."""
    )

    frequency = Instrument.measurement(
        "FREQ?",
        """Reference frequency in Hz."""
    )

    sensitivity = Instrument.control(
        "SENS %d",
        "SENS?",
        """Sensitivity setting index (0–26).""",
        validator=strict_discrete_set,
        values=range(27)
    )

    time_constant = Instrument.control(
        "TC %d",
        "TC?",
        """Time constant index (0–19).""",
        validator=strict_discrete_set,
        values=range(20)
    )

    x = Instrument.measurement("OUTP? 1", "X output")
    y = Instrument.measurement("OUTP? 2", "Y output")
    r = Instrument.measurement("OUTP? 3", "R output")
    theta = Instrument.measurement("OUTP? 4", "Theta output")


