# -*- coding: utf-8 -*-
"""
@author: Rolf van Lieshout
"""

import numpy as np
import math   
import os


class Point2D: 
    """Class for representing a point in 2D space"""
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    
    #method that computes the rounded euclidian distance between two 2D points
    def getDistance(c1,c2): 
        dx = c1.x-c2.x
        dy = c1.y-c2.y
        return math.sqrt(dx**2+dy**2)

class TSP:
    """
    Class for representing a Traveling Salesman Problem
    
    Attributes
    ----------
    nCities : int
        the number of cities
    cities : list of ints
        the cities, all represented by integers
    distMatrix : 2D array
        matrix with all distances between cities. Distance between city i and city j is distMatrix[i-1][j]
    
    """
    def __init__(self,tspFileName):
        """
        Reads a .tsp file and constructs an instance. 
        We assume that it is an Euclidian TSP

        Parameters
        ----------
        tspFileName : str
            name of the file
        """
        points = list() #add all points to list
        f = open(tspFileName)
        for line in f.readlines()[6:-1]: #start reading from line 7, skip last line
            asList = line.split()
            floatList = list(map(float,asList))

            id = int(floatList[0])-1 #convert to int, subtract 1 because Python indices start from 0
            x = floatList[1]
            y = floatList[2]

            c = Point2D(id,x,y)
            points.append(c)
        f.close()
        
        print("Read in all points, start computing distance matrix")

        
        self.nCities = len(points)
        self.cities = list(range(self.nCities))
        
        #compute distance matrix, assume Euclidian TSP
        self.distMatrix = np.zeros((self.nCities,self.nCities)) #init as nxn matrix
        for i in range(self.nCities):
            for j in range(i+1,self.nCities):
                distItoJ = Point2D.getDistance(points[i], points[j])
                self.distMatrix[i,j] = distItoJ
                self.distMatrix[j,i] = distItoJ
        
        print("Finished computing distance matrix")


    def getTour_NN(self,start):
        """
        Performs the nearest neighbour algorithm

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
    
        print("Start computing NN tour")
        for i in range(self.nCities-1):
            curCity = tour[i]
            closestDist = -1 #initialize with -1
            closestCity = None #initialize with None

            #find closest city not yet in tour
            for j in notInTour:
                dist = self.distMatrix[curCity][j]
                if dist<closestDist or closestCity is None:
                    #update the closest city and distance
                    closestDist = dist
                    closestCity = j
            
            tour.append(closestCity)
            notInTour.remove(closestCity)
        
        print("Finished computing NN tour")

        return tour
    
    # ========= helpers (reused by both variants) =========
    def nearest_dist_to_tour(self, city, tour):
        """Distance from 'city' to its nearest city in 'tour'."""
        best = None
        for i in tour:
            d = self.distMatrix[city][i]
            if best is None or d < best:
                best = d
        return best

    def furthest_city(self, candidates, from_set):
        """
        Among 'candidates', return the city whose distance to the *nearest*
        city in 'from_set' is maximal (i.e., the 'outlier').
        """
        furthestCity = None
        furthestDist = -1.0
        for c in candidates:
            # distance to *nearest* city already in from_set
            nearest = None
            for i in from_set:
                d = self.distMatrix[c][i]
                if nearest is None or d < nearest:
                    nearest = d
            if nearest > furthestDist:
                furthestDist = nearest
                furthestCity = c
        return furthestCity

    def best_insertion_position(self, tour, city):
        """
        Return (bestPos, bestIncrease) for inserting 'city' into 'tour'.
        Insert between tour[k] and tour[(k+1)%m] at position k+1.
        """
        bestPos = 0
        bestIncrease = None
        m = len(tour)
        for k in range(m):
            i = tour[k]
            j = tour[(k + 1) % m]
            increase = (self.distMatrix[i][city]
                        + self.distMatrix[city][j]
                        - self.distMatrix[i][j])
            if bestIncrease is None or increase < bestIncrease:
                bestIncrease = increase
                bestPos = k + 1
        return bestPos, bestIncrease

    # ========= deterministic Outlier Insertion =========
    def getTour_OutlierInsertion(self, start):
        """
        Performs the (deterministic) Outlier Insertion heuristic:
        - add the city furthest from 'start'
        - repeatedly add the city furthest from the current tour,
          inserted at the position with smallest increase.
        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print("Start computing Outlier Insertion tour")

        # first extension: furthest from start
        first = self.furthest_city(notInTour, [start])
        tour.append(first)
        notInTour.remove(first)

        # iterative insertions
        while notInTour:
            candidate = self.furthest_city(notInTour, tour)
            pos, _ = self.best_insertion_position(tour, candidate)
            tour.insert(pos, candidate)
            notInTour.remove(candidate)

        print("Finished computing Outlier Insertion tour")
        return tour

    # ========= GRASP'ed Outlier Insertion (randomized both steps) =========
    def getTour_GRASPedInsertion(self, start, alpha_pct=10.0, seed=None):
        """
        GRASP version of Outlier Insertion that randomizes *both* steps using an α% rule.

        Parameters
        ----------
        start : int
            starting city
        alpha_pct : float
            α in percent (e.g., 10 means 'within 10% of the best'):
            - City selection (maximize): choose uniformly at random among cities whose
              'distance to nearest tour city' ≥ (1-α) * bestScore.
            - Position selection (minimize): choose uniformly among positions whose
              insertion increase ≤ (1+α) * bestIncrease.
        seed : int | None
            RNG seed for reproducibility.

        Returns
        -------
        tour : list[int]
        """
        import random
        rng = random.Random(seed)
        alpha = float(alpha_pct) / 100.0

        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print("Start computing GRASPed Outlier Insertion tour")

        # ---- first extension: furthest from start with α-RCL ----
        # scores = distance to start
        scores = {c: self.distMatrix[start][c] for c in notInTour}
        bestScore = max(scores.values())
        thr_city = (1.0 - alpha) * bestScore
        rcl_cities = [c for c, s in scores.items() if s >= thr_city]
        first = rng.choice(rcl_cities)
        tour.append(first)
        notInTour.remove(first)

        # ---- iterative insertions: randomize city & position with α-RCLs ----
        while notInTour:
            # CITY RCL (maximize nearest-to-tour distance)
            scores = {c: self.nearest_dist_to_tour(c, tour) for c in notInTour}
            bestScore = max(scores.values())
            thr_city = (1.0 - alpha) * bestScore
            rcl_city = [c for c, s in scores.items() if s >= thr_city]
            city = rng.choice(rcl_city)

            # POSITION RCL (minimize insertion increase)
            # gather all positions and best increase
            m = len(tour)
            opts = []
            bestInc = None
            for k in range(m):
                i, j = tour[k], tour[(k + 1) % m]
                inc = (self.distMatrix[i][city]
                       + self.distMatrix[city][j]
                       - self.distMatrix[i][j])
                opts.append((k + 1, inc))
                if bestInc is None or inc < bestInc:
                    bestInc = inc
            thr_pos = (1.0 + alpha) * bestInc
            rcl_pos = [pos for (pos, inc) in opts if inc <= thr_pos]
            pos = rng.choice(rcl_pos)

            tour.insert(pos, city)
            notInTour.remove(city)

        print("Finished computing GRASPed Outlier Insertion tour")
        return tour
    
    def isTwoOpt(self, tour):
        """
        Checks whether a tour is 2-optimal.
        A tour is 2-optimal if no 2-exchange (edge swap) improves the total length.

        Parameters
        ----------
        tour : list[int]
            Current tour.

        Returns
        -------
        bool
            True if 2-optimal, False otherwise.
        """
        n = len(tour)
        best_cost = self.computeCosts(tour)

        # try all 2-opt moves
        for i in range(n - 1):
            for j in range(i + 2, n):  
                if i == 0 and j == n - 1:  # skip breaking the start/end edge
                    continue

                # perform 2-opt swap
                new_tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                new_cost = self.computeCosts(new_tour)

                if new_cost < best_cost:
                    return False  # found an improving move

        return True  # no improvement found

    def makeTwoOpt(self, tour):
        """
        Applies 2-exchanges until the tour is 2-optimal.

        Parameters
        ----------
        tour : list[int]
            Current tour.

        Returns
        -------
        list[int]
            Improved 2-optimal tour.
        """
        n = len(tour)
        improved = True
        best_tour = tour
        best_cost = self.computeCosts(best_tour)

        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):  
                    if i == 0 and j == n - 1:  # skip invalid swap
                        continue

                    # try 2-opt swap
                    new_tour = best_tour[:i + 1] + best_tour[i + 1:j + 1][::-1] + best_tour[j + 1:]
                    new_cost = self.computeCosts(new_tour)

                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break  # restart search after improvement
                if improved:
                    break

        return best_tour
    
    def getTour_GRASP2Opt(self, start: int, alpha_pct: float = 10.0, seed: int | None = None):
        """
        Construct a tour with GRASPed Outlier Insertion and improve it with 2-opt.

        Parameters
        ----------
        start : int
            Starting city.
        alpha_pct : float, optional
            α percentage for GRASP restricted candidate lists (default 10.0).
        seed : int | None, optional
            RNG seed forwarded to the GRASP builder.

        Returns
        -------
        list[int]
            2-opt–improved tour.
        """
        tour_before = self.getTour_GRASPedInsertion(start=start, alpha_pct=alpha_pct, seed=seed)
        tour_after = self.makeTwoOpt(tour_before)
        return tour_after
    
    def getCitiesCopy(self): 
        return self.cities.copy()
        
    def evaluateSolution(self,tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            print("The solution is feasible with costs "+str(costs))
        else: 
            print("The solution is infeasible")
    
    def isFeasible(self,tour):
        """
        Checks if tour is feasible

        Parameters
        ----------
        tour : list of integers
            order in which cities are visited. For a 4-city TSP, an example tour is [3, 1, 4, 2]

        Returns
        -------
        bool
            TRUE if feasible, FALSE if infeasible.

        """
        #first check if the length of the tour is correct
        if len(tour)!=self.nCities:
            print("Length of tour incorrect")
            return False
        else: 
            #check if all cities in the tour
            for city in self.cities:
                if city not in tour:
                    return False
        return True
    
    def computeCosts(self,tour):
        """
        Computes the costs of a tour

        Parameters
        ----------
        tour : list of integers
            order of cities.

        Returns
        -------
        costs : int
            costs of tour.

        """
        costs = 0
        for i in range(len(tour)-1):
            costs += self.distMatrix[tour[i],tour[i+1]]
            
        # add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1],tour[0]]
        return costs
    

