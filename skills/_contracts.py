"""The shared records of prompts/interfaces.md, re-exported from their owner, executor/contracts.py.

Kept as a module so `from skills._contracts import ...` (and the improvement loop's protected-path list) keep working.
`_contracts_vendored.py` is a byte-identical copy of executor/contracts.py, used only where the executor package is
absent (the improvement loop's minimal sandbox repo); executor/tests/test_wiring.py fails if the two differ.
World frame: metres, radians, z up, origin at the follower base on the table (prompts/scene-contract.md).
"""
try:
    from executor.contracts import (ActionProposal, ExecutionResult, FrameRef, JointState, Observation,  # noqa: F401
                                    ObjectEstimate, Outcome, SetupId, Status, ValidationResult, proposal_from_dict, to_dict)
except ImportError:      # standalone copy of skills/ without the executor
    from ._contracts_vendored import (ActionProposal, ExecutionResult, FrameRef, JointState, Observation,  # noqa: F401
                                      ObjectEstimate, Outcome, SetupId, Status, ValidationResult, proposal_from_dict, to_dict)

__all__ = ["ActionProposal", "ExecutionResult", "FrameRef", "JointState", "Observation", "ObjectEstimate", "Outcome", "SetupId",
           "Status", "ValidationResult", "proposal_from_dict", "to_dict"]
