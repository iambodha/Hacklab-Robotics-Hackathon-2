"""The frozen records of prompts/interfaces.md, re-exported from their owner, executor/contracts.py.

Kept as a module so `from models._contracts import ...` keeps working; there is only one definition of each record.
"""
from executor.contracts import (ActionProposal, ExecutionResult, FrameRef, JointState, Observation, ObjectEstimate,  # noqa: F401
                                Outcome, SetupId, Status, ValidationResult, proposal_from_dict, to_dict)

__all__ = ["ActionProposal", "ExecutionResult", "FrameRef", "JointState", "Observation", "ObjectEstimate", "Outcome", "SetupId",
           "Status", "ValidationResult", "proposal_from_dict", "to_dict"]
