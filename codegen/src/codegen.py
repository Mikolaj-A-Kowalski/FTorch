"""For now all is in a Megamodule. Perhaps will be split later."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> Any:
    """
    Load a yaml file into Python objects.

    Following PyYAML it can return almost anything.
    """
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


class TorchType(Enum):
    """
    Represents one of the types in torch operations.

    The list is incomplete. Will expand on need.
    """

    Tensor = "at::Tensor"
    Scalar = "at::Scalar"

    @staticmethod
    def parse(st: str) -> "TorchType":
        """Parse a string into a TorchType."""
        st = st.strip()
        if st == "at::Tensor":
            return TorchType.Tensor
        elif st in {"at::Scalar", "const at::Scalar &"}:
            # The difference between const and non const in not very clear
            # output should determine if it is mutable...?
            return TorchType.Scalar
        else:
            msg = f"Unknown type: {st}"
            raise ValueError(msg)


@dataclass
class Argument:
    """
    Represents an operator argument or return value.

    We ignore a lot of keywords at the moment:
     - annotation
     - is_nullable (!)

    """

    name: str
    """Argument name"""

    type: TorchType
    """Corresponds to the dynamic_type"""

    optional: bool = False
    """Is an optional argument"""

    default: Any | None = None
    """Default value of the argument if provided."""

    output: bool = False
    """If true this argument should be provided as an output (?)"""

    def __init__(self, args: dict) -> None:
        """Parse a dictionary from the yaml into an Argument."""
        self.name = args["name"]
        self.type = TorchType.parse(args["dynamic_type"])
        self.optional = args.get("optional", False)
        self.default = args.get("default")
        self.output = args.get("output", False)

    @property
    def cpp_declaration(self) -> str:
        """Get the C++ declaration of this argument."""
        if self.output:
            return f"{self.type.value} & {self.name}"
        else:
            return f"const {self.type.value} & {self.name}"


@dataclass
class ReturnValue:
    """Represents a return value of the operator."""

    name: str
    """
    Points to either the name of an argument or is 'result'
    """
    type: TorchType
    """
    Dynamic type of the return value.

    Perhaps the `type` C++ declaration should be used instead?
    """

    def __init__(self, args: dict) -> None:
        """Parse a dictionary from the yaml into a ReturnValue."""
        self.name = args["name"]
        self.type = TorchType.parse(args["dynamic_type"])


@dataclass
class Operator:
    """Collects the data about a pytorch operator."""

    name: str
    """
    Name of the operator.
    """

    arguments: list[Argument]

    return_value: Argument

    def __init__(self, args: dict) -> None:
        """Parse a dictionary from the yaml into an Operator."""
        self.name = args["name"]
        self.arguments = [Argument(arg) for arg in args["arguments"]]

        return_values = args["returns"]
        if len(return_values) != 1:
            msg = f"Multiple returns not supported yet: {return_values}"
            raise NotImplementedError(msg)

        self.return_value = ReturnValue(return_values[0])
