class StatefulHierarchicalPlanner:
    def __init__(self, global_planner, local_planner, global_replan_freq=10):
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.global_replan_freq = global_replan_freq
        self.cached_obs_plan = None
        self.steps_since_last_global_plan = self.global_replan_freq

    def __call__(self, cond_dict, replan_freq=5):
        # decide whether to run the expensive global planner
        if (
            self.cached_obs_plan is None
            or self.steps_since_last_global_plan >= self.global_replan_freq
        ):
            self.cached_obs_plan, _ = self.global_planner(cond_dict)
            self.steps_since_last_global_plan = 0

        # always run the cheap local planner logic
        lookahead_steps = self.steps_since_last_global_plan + replan_freq
        waypoint_idx = min(lookahead_steps, len(self.cached_obs_plan) - 1)
        subgoal = self.cached_obs_plan[waypoint_idx]

        # low-level plan in action space conditioned on the waypoint
        start_obs = cond_dict["start_obs_goal"][0]
        local_cond = {"start_obs_goal": (start_obs, subgoal)}
        _, act_plan = self.local_planner(local_cond)

        # update step counter
        self.steps_since_last_global_plan += replan_freq

        return self.cached_obs_plan, act_plan
