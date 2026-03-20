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
from pacman_utils import util
import numpy as np


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        self.state = state

    def __eq__(self, other):
        return hasattr(other, 'state') and self.state == other.state

    def __hash__(self):
        return hash(self.state)
    
    def walls(self, state):
        # Returns a list of (x, y) pairs of wall positions
        #
        # This version just returns all the current wall locations
        # extracted from the state data.  In later versions, this will be
        # restricted by distance, and include some uncertainty.
        
        wallList= []
        wallGrid = state.getWalls()
        width = wallGrid.width
        height = wallGrid.height
        for i in range(width):
            for j in range(height):
                if wallGrid[i][j] == True:
                    wallList.append((i, j))            
        return wallList
    
    def inFront(self, object, facing, state):
        # Returns true if the object is along the corridor in the
        # direction of the parameter "facing" before a wall gets in the
        # way.
        
        pacman = state.getPacmanPosition()
        pacman_x = pacman[0]
        pacman_y = pacman[1]
        wallList = self.walls(state)

        # If Pacman is facing North
        if facing == Directions.NORTH:
            # Check if the object is anywhere due North of Pacman before a
            # wall intervenes.
            next = (pacman_x, pacman_y + 1)
            while not next in wallList:
                if next == object:
                    return True
                else:
                    next = (pacman_x, next[1] + 1)
            return False

        # If Pacman is facing South
        if facing == Directions.SOUTH:
            # Check if the object is anywhere due North of Pacman before a
            # wall intervenes.
            next = (pacman_x, pacman_y - 1)
            while not next in wallList:
                if next == object:
                    return True
                else:
                    next = (pacman_x, next[1] - 1)
            return False

        # If Pacman is facing East
        if facing == Directions.EAST:
            # Check if the object is anywhere due East of Pacman before a
            # wall intervenes.
            next = (pacman_x + 1, pacman_y)
            while not next in wallList:
                if next == object:
                    return True
                else:
                    next = (next[0] + 1, pacman_y)
            return False
        
        # If Pacman is facing West
        if facing == Directions.WEST:
            # Check if the object is anywhere due West of Pacman before a
            # wall intervenes.
            next = (pacman_x - 1, pacman_y)
            while not next in wallList:
                if next == object:
                    return True
                else:
                    next = (next[0] - 1, pacman_y)
            return False
    
    def getFeatureVector(self):
        # Returns local information about the environment in the form of a
        # feature vector
        features = []
        xLoc = self.state.getPacmanPosition()[0]
        yLoc = self.state.getPacmanPosition()[1]

        #Are there walls around Pacman?
        wallGrid = self.state.getWalls()
        if wallGrid[xLoc][yLoc+1] == True:
            features.append(1)
        else:
            features.append(0)
        if wallGrid[xLoc+1][yLoc] == True:
            features.append(1)
        else:
            features.append(0)
        if wallGrid[xLoc][yLoc-1] == True:
            features.append(1)
        else:
            features.append(0)
        if wallGrid[xLoc-1][yLoc] == True:
            features.append(1)
        else:
            features.append(0)

        # Is there food around Pacman?
        # foodGrid = self.state.getFood()
        # if foodGrid[xLoc][yLoc+1] == True:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if foodGrid[xLoc+1][yLoc] == True:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if foodGrid[xLoc][yLoc-1] == True:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if foodGrid[xLoc-1][yLoc] == True:
        #     features.append(1)
        # else:
        #     features.append(0)

        # Are there ghosts in any of the eight squares around Pacman
        ghosts = self.state.getGhostPositions()
        facing = self.state.getPacmanState().configuration.direction
        visibleGhost = False

        # Is there a ghost in front of Pacman?
        for i in range(len(ghosts)):
            if self.inFront(ghosts[i], facing, self.state):
                visibleGhost = True

        if visibleGhost:
            features.append(1)
        else:
            features.append(0)

        # encoded rather than sparse
        for x, y in [(0,1),(1,0),(0,-1),(-1,0)]:
            features.append(1 if (xLoc+x, yLoc+y) in ghosts else 0)

        # ghost nearby (within two spaces)
        nearby = any(abs(g[0]-xLoc)+abs(g[1]-yLoc) <= 2 for g in ghosts)
        features.append(1 if nearby else 0)

        foodGrid = self.state.getFood()
        food_list = foodGrid.asList()
        if food_list:
            nearest = min(food_list, key=lambda f: abs(f[0]-xLoc) + abs(f[1]-yLoc)) # nearest food in any direction
            fx, fy = nearest
            features.append(1 if fy > yLoc else 0)  # food is North
            features.append(1 if fx > xLoc else 0)  # food is East
            features.append(1 if fy < yLoc else 0)  # food is South
            features.append(1 if fx < xLoc else 0)  # food is West
        else:
            features.extend([0, 0, 0, 0])

        features.append(sum(np.array(self.state.getFood().data).flatten()))
            
        return features


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.5,
                 epsilon: float = 0.05,
                 gamma: float = 1.0,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.Q = {}
        self.N = {}
        self.prevState = None
        self.prevAction = None
        self.prevScore = None
        # self.learned = []

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

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(self,
                      startState: GameState,
                      endState: GameState):
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        features = startState.getFeatureVector()
        # print(f"features: {features}")
        score = endState.state.getScore()
        prevScore = startState.state.getScore()
        # print("computeReward Score: ", score)

        key = (
            tuple(features),
            self.prevAction
        )
        reward = score if prevScore is None else score - prevScore
        reward -= 10 if score < prevScore else 0
        if key not in self.Q:
            # self.Q[key] = reward
            self.N[key] = 1
        else:
            # self.Q[key] += reward
            self.N[key] += 1
        return reward

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        stateFeatures = GameStateFeatures(state)

        legal = stateFeatures.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # 1. Compute feature vector for current state
        current_features = tuple(stateFeatures.getFeatureVector())

        reward = 0
        # 2. Update Q-value from previous step, if this is not the first move
        if self.prevState is not None:
            reward = self.computeReward(self, self.prevState, stateFeatures)
            max_q_next = max([self.Q.get((current_features, a), 0) for a in legal])
            prev_key = (tuple(self.prevState.getFeatureVector()), self.prevAction)
            if self.N.get(prev_key, 0) < self.maxAttempts:
                reward += 10
            self.Q[prev_key] = self.Q.get(prev_key, 0) + self.alpha * (reward + self.gamma * max_q_next - self.Q.get(prev_key, 0))

        # 3. Choose next action using current state
        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action = max(legal, key=lambda a: self.Q.get((current_features, a), 0))
            # action = max(legal, key=lambda a: (self.Q.get((current_features, a), 0) + 10 if self.N.get((current_features, a), 0) < self.maxAttempts else 0))
        # else:
        #     max_q = max([self.Q.get((current_features, a), 0) for a in legal])
        #     # Find all actions that result in that maximum Q-value
        #     best_actions = [a for a in legal if self.Q.get((current_features, a), 0) == max_q]
        #     # Break ties randomly
        #     action = random.choice(best_actions)
        
        # 4. Remember current state and chosen action for next update
        self.prevState = stateFeatures
        self.prevAction = action
        return self.prevAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        if self.prevState is not None:
            # compute reward for final transition
            stateFeatures = GameStateFeatures(state)
            reward = stateFeatures.state.getScore() - self.prevState.state.getScore()

            prev_key = (tuple(self.prevState.getFeatureVector()), self.prevAction)

            # no future Q term because terminal state
            self.Q[prev_key] = self.Q.get(prev_key, 0) + self.alpha * (
                reward - self.Q.get(prev_key, 0)
            )
            if prev_key not in self.N:
                self.N[prev_key] = 0
            self.N[prev_key] += 1
        
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

        self.prevState = None
        self.prevAction = None
