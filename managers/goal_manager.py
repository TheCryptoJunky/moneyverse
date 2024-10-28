# goal_manager.py
import logging

class GoalManager:
    def __init__(self, config):
        self.config = config
        self.goals = []

    def add_goal(self, goal):
        self.goals.append(goal)

    def remove_goal(self, goal):
        self.goals.remove(goal)

    def get_goals(self):
        return self.goals

# Usage:
goal_manager = GoalManager(config)
goal_manager.add_goal('Maximize profit')
goal_manager.add_goal('Minimize risk')
