# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
import numpy as np


class GameStateFeatures:

    def __init__(self, state: GameState):
        self.state = state

    def walls(self, state):
        
        wallList= []
        wallGrid = state.getWalls()
        width = wallGrid.width
        height = wallGrid.height
        for i in range(width):
            for j in range(height):
                if wallGrid[i][j]:
                    wallList.append((i, j))
        return wallList

    def inFront(self, obj, facing, state):

        pacman = state.getPacmanPosition()
        pacman_x = pacman[0]
        pacman_y = pacman[1]
        wallList = self.walls(state)

        # Check north until we hit a wall, looking for the object
        if facing == Directions.NORTH:
            nxt = (pacman_x, pacman_y + 1)
            while nxt not in wallList:
                if nxt == obj:
                    return True
                nxt = (pacman_x, nxt[1] + 1)
            return False

        # Check south until we hit a wall, looking for the object
        if facing == Directions.SOUTH:
            nxt = (pacman_x, pacman_y - 1)
            while nxt not in wallList:
                if nxt == obj:
                    return True
                nxt = (pacman_x, nxt[1] - 1)
            return False

        # Check east until we hit a wall, looking for the object
        if facing == Directions.EAST:
            nxt = (pacman_x + 1, pacman_y)
            while nxt not in wallList:
                if nxt == obj:
                    return True
                nxt = (nxt[0] + 1, pacman_y)
            return False

        # Check west until we hit a wall, looking for the object
        if facing == Directions.WEST:
            nxt = (pacman_x - 1, pacman_y)
            while nxt not in wallList:
                if nxt == obj:
                    return True
                nxt = (nxt[0] - 1, pacman_y)
            return False

        return False

        # using this from coursework 1 with slight modifications
    def getFeatureVector(self):
        # Returns local information about the environment in the form of a feature vector
        features = []
        xLoc, yLoc = self.state.getPacmanPosition()

        # Walls around Pacman: N, E, S, W
        wallGrid = self.state.getWalls()
        features.append(1 if wallGrid[xLoc][yLoc + 1] else 0) # North wall
        features.append(1 if wallGrid[xLoc + 1][yLoc] else 0) # East wall
        features.append(1 if wallGrid[xLoc][yLoc - 1] else 0) # South wall
        features.append(1 if wallGrid[xLoc - 1][yLoc] else 0) # West wall

        ghosts = self.state.getGhostPositions()
        facing = self.state.getPacmanState().configuration.direction
        # is there a ghost in front of pacman
        visibleGhost = any(self.inFront(g, facing, self.state) for g in ghosts)
        features.append(1 if visibleGhost else 0)

        # for calculating if ghosts are around pacman
        # reduced this feature space to 4 (up, down, left, right) rather than 8 (including diagonals)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            features.append(1 if (xLoc + dx, yLoc + dy) in ghosts else 0)

        # Ghost nearby (Manhattan distance <= 2)
        nearby = any(abs(g[0] - xLoc) + abs(g[1] - yLoc) <= 2 for g in ghosts)
        features.append(1 if nearby else 0)

        # replaced food metric in cw1 to find nearest food rather than "is there food around pacman"
        foodGrid = self.state.getFood()
        food_list = foodGrid.asList()
        if food_list:
            nearest = min(food_list, key=lambda f: abs(f[0] - xLoc) + abs(f[1] - yLoc)) # nearest food in any direction including diagonals
            fx, fy = nearest
            features.append(1 if fy > yLoc else 0) # food North
            features.append(1 if fx > xLoc else 0) # food East
            features.append(1 if fy < yLoc else 0) # food South
            features.append(1 if fx < xLoc else 0) # food West
        else:
            features.extend([0, 0, 0, 0])

        # feature for the number of remaining foodcrumbs
        features.append(int(np.sum(np.array(self.state.getFood().data))))

        return features



class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        super().__init__()
        
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        
        self.episodesSoFar = 0
        self.Q = {}
        self.N = {}
        self.prevState = None
        self.prevAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        
        # score is calculated as difference between current score and previous score
        return endState.state.getScore() - startState.state.getScore()

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        
        # get Q value for this state and action, if note available, we return 0
        key = (tuple(state.getFeatureVector()), action)
        return self.Q.get(key, 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:

        legal = state.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if not legal:
            return 0.0

        # among the legal actions for this state, we find the highest q value, then we assign return that max Q value
        return max(self.getQValue(state, action) for action in legal)

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        
        ##Apply the QLearning function to the given state-action pair with the associated reward and nextState
        key = (tuple(state.getFeatureVector()), action)
        oldQ = self.Q.get(key, 0.0)
        sample = reward + self.gamma * self.maxQValue(nextState)
        self.Q[key] = oldQ + self.alpha * (sample - oldQ)

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        
        ##Add 1 to the Count of the given state-action pair
        key = (tuple(state.getFeatureVector()), action)
        self.N[key] = self.N.get(key, 0) + 1

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        
        ##Return the Count of the given state-action pair
        key = (tuple(state.getFeatureVector()), action)
        return self.N.get(key, 0)

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        
        ##When returning utility of state-action pairs with count lesser than maxAttempts, return it with a slight incraese
        if counts < self.maxAttempts:
            return utility + 5.0 / (counts + 1)
        return utility

    def getAction(self, state: GameState) -> Directions:

        ##Obtain the stateFeatures from the state
        stateFeatures = GameStateFeatures(state)

        # Learn from previous transition if any (Compute reward -> Update count -> Apply QLearning)
        if self.prevState is not None and self.prevAction is not None:
            reward = self.computeReward(self.prevState, stateFeatures)
            self.updateCount(self.prevState, self.prevAction)
            self.learn(self.prevState, self.prevAction, reward, stateFeatures)

        ##Obtain the legal actions (STOP not included)
        legal = stateFeatures.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if not legal:
            return Directions.STOP

        # Choose next action via Epsilon Greedy mechanism
        if random.random() < self.epsilon:
            ##Random directions are selected among legal actions (STOP not included)
            action = random.choice(legal)
        else:
            ##If direction is not randomly chosen
            ##go through all viable actions for the current state 
            ##and select the one with best value
            scored_actions = []
            best_value = float('-inf')
            for a in legal:
                value = self.explorationFn(self.getQValue(stateFeatures, a),
                                           self.getCount(stateFeatures, a))
                scored_actions.append((a, value))
                if value > best_value:
                    best_value = value

            best_actions = [a for a, v in scored_actions if v == best_value]
            action = random.choice(best_actions)

        ##Save the current state and chosen action as previous State and Actions, for use in the next iteration. 
        self.prevState = stateFeatures
        self.prevAction = action

        ##Return the selected action
        return action

    def final(self, state: GameState):

        """Called at the end of each game"""

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        stateFeatures = GameStateFeatures(state)

        # Final terminal update
        if self.prevState is not None and self.prevAction is not None:
            reward = self.computeReward(self.prevState, stateFeatures)
            key = (tuple(self.prevState.getFeatureVector()), self.prevAction)
            oldQ = self.Q.get(key, 0.0)
            self.Q[key] = oldQ + self.alpha * (reward - oldQ)
            self.N[key] = self.N.get(key, 0) + 1

        self.incrementEpisodesSoFar()

        # Check if we are done training
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

        self.prevState = None
        self.prevAction = None
    