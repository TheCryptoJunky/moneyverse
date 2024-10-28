# Full file path: /moneyverse/managers/goal_manager.py

import logging

# Set up centralized logging
logger = logging.getLogger(__name__)

class GoalManager:
    """
    Manages trading and strategy goals, including adding, removing, and retrieving structured goals.
    """

    def __init__(self, config):
        self.config = config
        self.goals = []

    def add_goal(self, goal, priority="medium", status="pending"):
        """
        Add a structured goal to the manager.

        Parameters:
            goal (str): Description of the goal.
            priority (str): Priority of the goal ("low", "medium", "high").
            status (str): Status of the goal ("pending", "in_progress", "completed").
        """
        structured_goal = {
            "description": goal,
            "priority": priority,
            "status": status
        }
        self.goals.append(structured_goal)
        logger.info(f"Added goal: {structured_goal}")

    def remove_goal(self, goal_description):
        """
        Remove a goal from the manager based on the description.

        Parameters:
            goal_description (str): Description of the goal to remove.
        """
        for goal in self.goals:
            if goal["description"] == goal_description:
                self.goals.remove(goal)
                logger.info(f"Removed goal: {goal}")
                return
        logger.warning(f"Goal not found: {goal_description}")

    def update_goal_status(self, goal_description, new_status):
        """
        Update the status of a specified goal.

        Parameters:
            goal_description (str): Description of the goal to update.
            new_status (str): New status of the goal ("pending", "in_progress", "completed").
        """
        for goal in self.goals:
            if goal["description"] == goal_description:
                old_status = goal["status"]
                goal["status"] = new_status
                logger.info(f"Updated goal '{goal_description}' status from '{old_status}' to '{new_status}'")
                return
        logger.warning(f"Goal not found: {goal_description}")

    def get_goals(self, status=None, priority=None):
        """
        Retrieve goals, optionally filtering by status and priority.

        Parameters:
            status (str, optional): Filter goals by status.
            priority (str, optional): Filter goals by priority.
        
        Returns:
            List[Dict]: List of goals matching the specified filters.
        """
        filtered_goals = [
            goal for goal in self.goals
            if (status is None or goal["status"] == status) and (priority is None or goal["priority"] == priority)
        ]
        logger.info(f"Retrieved goals with status={status} and priority={priority}: {filtered_goals}")
        return filtered_goals
