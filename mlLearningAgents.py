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
    
    # using this from coursework 1
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
    
    # using this from coursework 1
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
    
    # using this from coursework 1 with slight modifications
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

        # reduced this feature space to 4 (up, down, left, right) rather than 8 (including diagonals)
        for x, y in [(0,1),(1,0),(0,-1),(-1,0)]:
            features.append(1 if (xLoc+x, yLoc+y) in ghosts else 0)

        # ghost nearby (within two spaces)
        nearby = any(abs(g[0]-xLoc)+abs(g[1]-yLoc) <= 2 for g in ghosts)
        features.append(1 if nearby else 0)

        # replaced food metric in cw1 to find nearest food rather than "is there food around pacman"
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
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
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
        
        self.episodesSoFar = 0
        self.Q = {}
        self.N = {}
        self.prevState = None
        self.prevAction = None
        self.prevScore = None

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
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"

        features = startState.getFeatureVector()
        score = endState.state.getScore()
        prevScore = startState.state.getScore()

        reward = score if prevScore is None else score - prevScore
        reward -= 10 if score < prevScore else 0

        return reward



    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        
        features = state.getFeatureVector()
        key = (
            tuple(features),
            action
        )
        if key not in self.Q:
            self.Q[key] = 0
            self.N[key] = 0

        return self.Q[key]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        features = state.getFeatureVector()
        legal = state.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        maxQValue = -float('inf')

        for action in legal:
            key = (
                tuple(features),
                action
            )
            qValue = self.getQValue(state, action)
            if (qValue>maxQValue):
                maxQValue = qValue
        
        return maxQValue


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        features = tuple(state.getFeatureVector())
        key = (features, action)

        maxQNext = self.maxQValue(nextState)
        
        oldQ = self.getQValue(state, action)

        # Q-learning update
        self.Q[key] = oldQ + self.alpha * (reward + self.gamma * maxQNext - oldQ)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        features = tuple(state.getFeatureVector())
        key = (
            tuple(features),
            action
        )
        if key not in self.Q:
            self.N[key] = 1
        else:
            self.N[key] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        
        features = tuple(state.getFeatureVector())
        key = (
            tuple(features),
            action
        )
        if key not in self.Q:
            self.Q[key] = 0
            self.N[key] = 0
        
        return self.N[key]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        if(counts==0):
            return float('inf')
        else:
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        # print(f"state: {stateFeatures.state.data}")

        # 1. Learn from previous action (If Any)
        if self.prevState is not None:
            prev_key = (tuple(self.prevState.getFeatureVector()), self.prevAction)
            reward = self.computeReward(self.prevState, stateFeatures)
            self.updateCount(self.prevState, self.prevAction)
            if self.getCount(self.prevState, self.prevAction) < self.maxAttempts:
                reward += 10
            self.learn(self.prevState, self.prevAction, reward, stateFeatures)


        # 3. Choose next action using current state
        
        legal = stateFeatures.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action = max(legal, key=lambda a: 
                self.explorationFn(self.getQValue(stateFeatures, a), self.getCount(stateFeatures, a)))

        # 4. Store current state and action for the next learning step
        self.prevState = stateFeatures
        self.prevAction = action

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        stateFeatures = GameStateFeatures(state)
        # print(f"state: {stateFeatures.state.data}")

        # 1. Learn from previous action (If Any)
        if self.prevState is not None:
            prev_key = (tuple(self.prevState.getFeatureVector()), self.prevAction)
            reward = self.computeReward(self.prevState, stateFeatures)
            # no future Q term because terminal state
            self.Q[prev_key] = self.Q.get(prev_key, 0) + self.alpha * (
                reward - self.Q.get(prev_key, 0)
            )
            if prev_key not in self.N:
                self.N[prev_key] = 0
            self.N[prev_key] += 1
        
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
    
