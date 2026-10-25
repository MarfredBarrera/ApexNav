# The planner's next action in sync_step mode, tied to the step it was planned from.

int32 action
# last StepObservation.step processed before planning this action
int32 step
# ObjectFusionState.update_seq published before this action; the evaluator waits for it
uint32 fusion_seq
# FINAL_RESULT and EXPL_RESULT in force when the action was published (-1 = none yet)
int32 final_state
int32 expl_result
